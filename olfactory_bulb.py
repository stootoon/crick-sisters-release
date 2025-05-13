import numpy as np
from cmath import sqrt as csqrt
import logging, json
import time, datetime
import cvxpy as cvx
import sisters_util as util
import autograd.numpy as anp
import pymanopt

from sisters_util import TimedBlock

logger = util.create_logger("olfactory_bulb")
INFO   = logger.info
WARN   = logger.warning
DEBUG  = logger.debug

def linearly_rectify(x,th):
    return (x > th)*(x - th)

def smooth_rectify(x,th,f=10):
    return np.log(1 + np.exp(f*(x - th)))/f

def get_x_true(N, k, spread = 0):
    vals     = np.linspace(1-spread/2, 1+spread/2, k)
    ind_x_on = {int(round(i*float(N)/(k+1))) : vals[i-1] for i in range(1,k+1)}
    x_true   = np.zeros((N,))
    for ind, val in ind_x_on.items():
        x_true[ind] = val
        
    return x_true
    
class OlfactoryBulb:
    @staticmethod
    def compute_affinity_and_covariance(A):
        M = len(A)
        S = [Ai.shape[0] for Ai in A]
        N = A[0].shape[1]
        
        Ci = [np.zeros((N,N)) for Ai in A]
        C  =  np.zeros((N,N))
        a  =  np.zeros((M,N))
    
        for i, (Si, Ai) in enumerate(zip(S,A)):
            a[i] = np.mean(Ai, axis=0)
            Ci   = np.cov(Ai.T, bias=True)
            C   += len(Ai)*Ci
                
        return a, C, Ci

    @staticmethod
    def gen_random_weights(M, N, Srange, A_mean=None, scale=1e-3):
        # M: number of glomeruli
        # N: number of odours
        # Srange: range of number of sisters
        assert Srange[0] <= Srange[1], "Srange[0] must be less than or equal to Srange[1]."
        
        S = np.random.randint(Srange[0], Srange[1], M) if Srange[1] > Srange[0] else [Srange[0]] * M
        
        A = [np.random.rand(Si, N) * scale for Si in S]

        if A_mean is None:
            A_mean = np.random.rand(M, N)
        
        for i, (Ai, Ami) in enumerate(zip(A, A_mean)):
            A[i] = Ai -np.mean(Ai, axis=0) + Ami
            assert np.allclose(np.mean(A[i], axis=0), Ami), "Mean of Ai is not equal to Ami."

        return A, S

    @staticmethod
    def generate_connectivity(M, N, Svals, A_mean, C, random_rotation = False, return_details = False, sparsify = 0, penalty = 1, verbosity = 0):
        # We need to specify the elements of A
        # A is a list of [Si x N] matrices, where Si is the number of sister cells for glomerulus i
        # We want the mean of the A[i] to be A_mean[i]
        # We also want sum Si Ci to equal C

        # We will generate the A[i] by first assuming the means are all zero and setting the values to
        # get the correct covariance matrix. Then we will adjust the means to get the correct mean.

        S = sum(Svals)
        A = [np.outer(np.ones(Sg,), Am_g) for Sg, Am_g in zip(Svals, A_mean)]

        # If penalty is a scalar, then expand it to size M
        # If not, assert that it has length M
        if np.isscalar(penalty):
            penalty = np.ones((M,)) * penalty
        else:
            assert len(penalty) == M, f"Sparsity penalty must be a scalar, or a vector of length {M}, not {len(penalty)}."
        
        # First determine how many odours have covariance constraints
        ind_nz = np.where(np.diag(C) > 1e-8)[0]
        n = len(ind_nz)
        
        if n == 0:
            WARN(f"Desired covariance had no non-zero diagonal elements.")
            return A
                    
        # See if that's too many
        n1 = np.sqrt(2 * (S - M) * N)
        n2 = M * (S - 1)
        n3 = S - M
        n_max = np.min([n1, n2, n3])
        assert n <= n_max, f"Too many odours with covariance constraints: {n} > {n_max}."
        DEBUG(f"n = {n} <= n_max = {n_max}.")

        C_n = C[ind_nz][:, ind_nz]
        mags = np.sqrt(np.diag(C_n))
        corrs = C_n / (mags[:, None] * mags[None, :])

        W = np.zeros((S, n))
        W[0,0] = 1
        for i in range(1, n):
            for j in range(i):
                W[j,i] = (corrs[i,j] - sum(W[:j,j]*W[:j,i])) / W[j,j]
            W[i,i] = np.sqrt(1 - sum(W[:i,i]**2))

        assert np.allclose(W.T @ W, corrs), "W'W is not equal to corrs."
        DEBUG("W'W is equal to corrs.")

        # Create the matrix that will enforce the zero-mean constraints
        B = np.zeros((M,S))
        starts = np.cumsum([0] + list(Svals))
        for i in range(M):
            B[i, starts[i]:starts[i+1]] = 1

        penalty_vector = penalty @ B

        _, _, VBt = np.linalg.svd(B, full_matrices = True)
        VB_perp = VBt.T[:, M:]

        # Rotate the weights into the nullspace of B
        _, sW, VWt = np.linalg.svd(W, full_matrices = False)
        SV_W = np.diag(sW) @ VWt
        # Generate a random n x n rotation
        n_perp = VB_perp.shape[1]
        R = np.linalg.qr(np.random.randn(n_perp,n_perp))[0] if random_rotation else np.eye(n_perp)
        random_rotation and DEBUG("Applying random rotation.")

        W1 = (VB_perp @ R)[:, :n] @ SV_W        

        if sparsify:
            manifold = pymanopt.manifolds.SpecialOrthogonalGroup(n_perp)
            
            @pymanopt.function.autograd(manifold)
            def loss(R):
                W1 = (VB_perp @ R)[:, :n] @ SV_W
                return anp.sum(anp.abs(penalty_vector[:, None] * W1))

            best_norm = loss(R)
            best_R = R
            
            print(f"Optimizing sparsity with {sparsify} iterations.")
            print(f"Initial norm: {best_norm}")

            problem = pymanopt.Problem(manifold=manifold, cost=loss)
            solver = pymanopt.optimizers.SteepestDescent(verbosity = verbosity)

            R_init = R
            for i in range(sparsify):
                res = solver.run(problem, initial_point=R_init)
                R = res.point
                new_best = False
                if res.cost < best_norm:
                    best_norm = res.cost
                    best_R    = R
                    new_best = True
                print(f"Iteration {i+1}: {res.cost} {'*' if new_best else ''}")
                    
                R_init = np.linalg.qr(np.random.randn(n_perp,n_perp))[0]
                
            W1 = (VB_perp @ best_R)[:, :n] @ SV_W

        assert np.allclose(W1.T @ W1, corrs), "W1'W1 does not equal corrs."
        DEBUG("W1'W1 is equal to corrs.")

        assert np.allclose(B @ W1, 0), "B W1 != 0: Not all constraints were met."
        DEBUG("B W1 == 0: Constraints were met")

        Wsol = W1 @ np.diag(mags)
        for g, Sg, Am_g in zip(range(M), Svals, A_mean):
            for ni, ii in zip(range(n), ind_nz):
                A[g][:, ii] += Wsol[starts[g]:starts[g+1],ni]

        # Check the means
        assert all([np.allclose(Ai.mean(axis=0), Ami) for Ai, Ami in zip(A, A_mean)]), "Means didn't match."
        INFO(f"Means matched in each glomerulus.")

        AtA = sum([Si * np.cov(Ai.T, bias=True) for Si, Ai in zip(Svals, A)])
        assert np.allclose(AtA, C), "AtA != C"
        INFO(f"Weighted sum of weighted covariances matched target.")

        details = {"W":W, "R":R, "W1":W1, "Wsol":Wsol, "B":B, "ind_nz":ind_nz}

        return (A, details) if return_details else A
        
    def __init__(self, A, # Should be a list of [Si x N] matrices, where Si is the number of sister cells for glomerulus i
                 sd=1e-2, be=100, ga=100,

                 # Whether to adjust the γ values so that the net effect,
                 # Including the coactivation penalty, achieves the
                 # desired gamma value.
                 enforce_ga = False,
                 tau_mc=0.025, tau_gc=0.050, 
                 active_gcs = "all", noisy=False,
                 verbosity=0, seed = 0,
                 **kwargs):

        self.seed = seed
        np.random.seed(self.seed)

        self.M  = len(A)

        # assert that all A have the same number of columns
        assert all([A[i].shape[1] == A[0].shape[1] for i in range(1, self.M)])
        self.N  = A[0].shape[1]

        self.S  = [A[i].shape[0] for i in range(self.M)]

        self.A = A
        self.A_flat = np.vstack(self.A)
        # Compute the mean connectivity to the GCs per glomerulus
        # This encodes the affinity of that glomerulus for each odour
        self.A_mean, self.C, self.C_sis = self.compute_affinity_and_covariance(A)

        
        self.sd = sd
        self.be = be        

        c = np.diag(self.C) / self.sd**2
        self.ga     = ga
        self.enforce_ga = enforce_ga
        self.ga_vec = ga * np.ones(self.N) if not enforce_ga else (ga - c)
                
        # Combine the L2 and precision loss into one quadratic form
        # Otherwise CVX complains that C is not positive definite, when it is.
        self.Q = np.diag(self.ga_vec) + self.C / self.sd**2        
        assert np.all(np.linalg.eigvals(self.Q) > 0), "Q is not positive definite."
        if enforce_ga:
            assert np.allclose(np.diag(self.Q), ga), "Diagonal of Q is not equal to ga."
        
        self.tau_mc  = tau_mc
        self.tau_gc  = tau_gc
        
        self.noisy   = noisy
        self.verbosity = verbosity
        self.active_gcs = active_gcs

        if self.verbosity>0:
            INFO("Created new olfactory bulb.")
            self.print_params()

    def print_params(self):
        INFO("OB Parameters:")
        INFO("  seed = %d." % self.seed)
        INFO("  M (# glom) = %d, N (# GCs) = %d" % (self.M,  self.N))
        INFO("  S (# sister cells per glom): {}".format(self.S))

        INFO("  First 3 values of A[0]: {}".format(self.A[0][0,:min(3,self.N)]))
        INFO("  Active gcs: {}.".format(self.active_gcs))
        INFO("  sd = %g, be = %g ga = %g." % (self.sd, self.be, self.ga))
        INFO("  enforce_ga = %s." % self.enforce_ga)
        INFO("  min(ga_vec) = %g, max(ga_vec) = %g." % (np.min(self.ga_vec), np.max(self.ga_vec)))
        INFO("  tau(mc) = %g, tau(gc) = %g." % (self.tau_mc, self.tau_gc))

        Q_eig = np.linalg.eigvalsh(self.Q)
        INFO(f"  min eig(Q) = {np.min(Q_eig):g}.")
        INFO(f"  max eig(Q) = {np.max(Q_eig):g}.")
        q     = np.diag(self.Q)
        INFO(f"  min diag(Q) = {np.min(q):g}.")
        INFO(f"  max diag(Q) = {np.max(q):g}.")
        
        INFO("  Noisy inputs = %g." % (self.noisy))

    def loss(self, x, y, version = "sis"):
        phi = self.be * np.linalg.norm(x,1)        
        lklhd = 0

        version = version.lower()
        assert version in ["sis", "cov", "q"], f"Unknown version {version}."
        
        if version == "sis":
            # L(x) = β ||x||_1 + 1/2 γ ||x||_2^2 + 1/2 sum_i sum_s (y_i - Ais . x)^2 / σ^2
            phi += 0.5 * np.sum(self.ga_vec * x**2)    
            for i, Si in enumerate(self.S):
                for s in range(Si):
                    lklhd += 0.5 * (y[i] - np.dot(self.A[i][s,:], x))**2
            lklhd /= self.sd**2
        elif version == "cov":
            phi += 0.5 * np.sum(self.ga_vec * x**2)
            phi += 0.5 * np.dot(x, np.dot(self.C, x)) / self.sd**2            
            lklhd = 0.5 * np.array(self.S) @ ((y - self.A_mean @ x)**2)/ self.sd**2
        elif version == "q":
            phi += 0.5 * np.dot(x, np.dot(self.Q, x))
            lklhd = 0.5 * np.array(self.S) @ ((y - self.A_mean @ x)**2)/ self.sd**2
        else:
            raise ValueError(f"Invalid {version=}.")
            
        return phi + lklhd 
    
    def run_exact(self, y, solver=cvx.SCS, **kwargs):
        """ Uses CVXOPT to infer the odor by solving
        the convex optimization problem directly.
        """
        INFO("Running exact model.")
        
        with TimedBlock(INFO, "RUNNING EXACT MODEL"):
            x = cvx.Variable(self.N)            
            r = cvx.Variable(self.M)
            objective  = cvx.Minimize(
                self.be * cvx.norm(x,1)
#                + 0.5 * self.ga * cvx.sum_squares(x)
#                + 0.5 * cvx.quad_form(x, self.C) / self.sd**2
                + 0.5 * cvx.quad_form(x, self.Q)
                + 0.5 * cvx.sum(cvx.multiply(self.S, cvx.square(r))) / self.sd**2  # Need to weght by Si
            )
            constraint = [r == y - self.A_mean @ x, x>=0]
            problem = cvx.Problem(objective, constraint)
            problem.solve(solver=solver, **kwargs)
            x_MAP = np.reshape(np.array(x.value),(self.N,))
            l_MAP = -constraint[0].dual_value
            r_MAP = l_MAP*self.sd*self.sd
        return {"x":x_MAP, "la": l_MAP, "r": r_MAP, "status":problem.status}
        
    def run_sister(self, odor, t_end, dt, keep_every = 1, Y_init = None, X_init = None, V_init = None, La_init = None, XFUN = linearly_rectify, report_frequency = 10, keep_till = np.inf):

        INFO(f"Run Parameters:")
        INFO(f"  Running to t={t_end}.")
        INFO(f"  Stepsize {dt=}.")
        INFO(f"  Keeping every {keep_every} step(s).")        
        t_all  = np.arange(0, t_end, dt)
        nt_all = len(t_all)
        INFO(f"  {nt_all} total time steps.")
        # We often want the sim to run for some large time interval T.
        # This is to e.g. get the final values of the inference.
        # But we only want it to periodically report within some shorter interval
        # in which we e.g. examine the dynamics.
        INFO(f"  Keeping times t < {keep_till}.")
        ind_keep = np.arange(0, min(keep_till, t_end), dt)[::keep_every]
        nt_keep  = len(ind_keep)
        INFO(f"  ({nt_keep} time steps to keep.)")

        # Generate the input
        # - Y has every time step because we'll need this throughout the integration.
        # - 18 August 2020: Changed this to keeping only t_keep timesteps because
        #   the memory foot print is too large.
        
        with TimedBlock(INFO, "PREPARING INPUTS"):
            Y  = [np.zeros((nt_keep, Si), order = "C") for Si in self.S]
            if Y_init is not None:
                for iglom, y_glom in enumerate(Y_init):
                    Y[iglom][0,:] = y_glom

        with TimedBlock(INFO, "SETTING INITIAL CONDITIONS"):
            X  = np.zeros((nt_keep, self.N), order="C")
            x  = X_init if X_init else np.zeros((self.N,))
            X[0,:] = x
                
            La = [np.zeros((nt_keep, Si), order="C") for Si in self.S]
            la = La_init if La_init else [np.zeros(Si, order="C") for Si in self.S]

            for La_glom, la_glom in zip(La, la):
                La_glom[0, :] = la_glom
            
        self.gc_mask = np.ones(self.N,) if self.active_gcs == "all" else np.array([i in self.active_gcs for i in range(self.N)])
        INFO(f"Set gc_mask according to active_gcs = {self.active_gcs}: {int(sum(self.gc_mask))}/{self.N} gcs active.")
        
        status    = "FAILED."
        success   = False
        last_iter = 0
        t_keep = [0]
        with TimedBlock(INFO, "MAIN LOOP"):
            report_interval = nt_all//report_frequency
            t_report = 0
            i_keep   = 1
            for ti, t in enumerate(t_all):
                if ti == 0:
                    continue

                Ax_per_glom = [Ai @ x for Ai in self.A]

                la_flat     = np.concatenate(la)
                if np.any(np.isnan(la_flat)):
                    status  = f"Overflow at step {t}."
                    success = False
                    break
                
                AtLa        = self.A_flat.T @ la_flat

                if ti == 1:
                    Yt = [y_glom[0,:] for y_glom in Y]
                else:
                    y  = odor.value_at(t_all[ti-1], A = self.A_mean) if hasattr(odor, "value_at") else odor
                    #Yt = np.outer(y, np.ones(self.S))
                    Yt = [yi * np.ones(Si) + self.noisy * np.random.randn(Si) * self.sd * np.sqrt(dt) for yi, Si in zip(y, self.S)]
                    
                dladt = [-la_glom + (yt_glom - Ax_glom)/self.sd**2 for la_glom, yt_glom, Ax_glom in zip(la, Yt, Ax_per_glom)]

                dxdt  = -self.be * np.sign(x) - self.ga_vec * x + AtLa

                for i in range(self.M):
                    la[i] += dladt[i]/self.tau_mc * dt

                x +=  dxdt/self.tau_gc * dt * self.gc_mask
                x[x<0] = 0
                
                last_iter = ti

                t_report += 1
                if t_report == report_interval:
                    n_prog = ti//report_interval
                    INFO("[" + "*" * n_prog + " "*(report_frequency - n_prog) + "]")
                    t_report = 0

                if (t < keep_till) and (not np.mod(ti, keep_every)):
                    X[ i_keep, :] = x
                    for iglom, (la_glom, yt_glom) in enumerate(zip(la, Yt)):
                        La[iglom][i_keep, :] = la_glom
                        Y[iglom][ i_keep, :] = yt_glom
                    i_keep += 1
                    t_keep.append(t)

            
        if last_iter == nt_all-1: # All iterations ran
            status = "OK."
            success = True
            
        return {"T":np.array(t_keep), "X":X, "La":La, "Y":Y, "A10":self.A[0][:min(self.M,10)],
                "t_final":last_iter*dt,
                "x_final":x,
                "y_final":y,
                "la_final":la,
                "last_iter":last_iter, "success":success, "status":status}            

            
    
def get_x_MAP(run_params_json):
    with open(run_params_json, "rb") as in_file:
        p = json.load(in_file)

    ob = OlfactoryBulb(**p)
    y = np.dot(ob.A_mean, get_x_true(p["N"],p["k"], spread=p["spread"]))
    res = ob.run_exact(y, eps=5e-13)
    return res["x"]

    
