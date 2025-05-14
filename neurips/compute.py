import os, sys, pickle, logging
import numpy as np
import pandas as pd

git_path = os.environ["GIT"]
project_path = os.path.join(git_path, "crick-sisters-release")
sys.path.append(project_path)


import olfactory_bulb as ob
import odours as od
import sisters_util as util

class Computation:
    def __init__(self, *args, **kwargs):
        self.computed = False
        
    def compute(self, *args, **kwargs):
        raise NotImplementedError("compute() method not implemented")

class ConnectivitySchematic(Computation):
    def compute(self, *args, **kwargs):
        print("COMPUTING CONNECTIVITY SCHEMATIC DATA")
        np.random.seed(5)
        M, N = 50, 200
        n = 5
        C = util.gen_cov2(N, n, -0.24)
        np.random.seed(3)
        sd = 20
        A_mean = np.random.rand(M, N)*3
        Smin, Smax = 2, 6
        # Pick S values randomly from the range
        Svals = [np.random.randint(Smin, Smax) for i in range(M)];
        print(Svals);
        cv = [0] + list(np.cumsum(Svals))
        parent_glom = np.concatenate([[i]*s for i, s in enumerate(Svals)])
        sis_id = np.arange(sum(Svals))-np.concatenate([[cvi]*s for cvi, s in zip(cv[:-1], Svals)])
        glom_cols = [cm.hsv(0.5*(i%2) + 0.5*np.random.rand()) for i in range(M)]
        sis_cols = np.concatenate([[gc]*s for gc, s in zip(glom_cols, Svals)])
        odour = od.Odour(np.arange(N)<n, A_mean)
        A1, details = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd**2 * C * 1e-1, random_rotation=True, return_details=True)

        self.Svals = Svals
        self.A_mean = A_mean
        self.A1 = A1
        self.details = details
        self.parent_glom = parent_glom
        self.sis_id = sis_id
        self.sis_cols = sis_cols
        self.glom_cols = glom_cols

        self.computed = True
        print("DONE COMPUTING CONNECTIVITY SCHEMATIC DATA.")
    
class ConnectivityDynamics(Computation):
    @staticmethod
    def gen_sparse_cov(N, n, rho = 0.9, sp = 0.1):
        Cnn = (np.random.rand(n,n) <= 0.1) * rho
        # Set the lower triangular indices to zero
        Cnn[np.tril_indices(n)] = 0
        Cnn += Cnn.T
        Cnn[np.diag_indices(n)] = 1
        C = np.eye(N)
        C[:n, :n] = Cnn
        return C

    @staticmethod
    def compute_affinity_and_correlation_from_weights(A):
        """
        Given a list of Si x N sister cell weights,
        computes the implied affinity and prior correlation.
        """
        S   = [len(Ai) for Ai in A]
        aff = np.array([Ai.mean(axis=0) for Ai in A]) # Affinity is the mean across sister cells
        Ams = [Ai - affi for Ai,affi in zip(A, aff)]        
        Ci  = [np.cov(Ai.T, bias=True) for Ai in A]
        C   = np.average(Ci, weights = S, axis = 0) * np.sum(S) # Weighted sum of covariances from each input channel
        Ams_= np.vstack(Ams)
        assert np.allclose(C, Ams_.T @ Ams_), "Covariance computed as weighted sum of input channels does not match covariance computed directly from weights."
        return aff, C
            
    def compute(self, force = False):
        print("PREPPING CONNECTIVITY DYNAMICS DATA...")
        print("\tGenerating sparse covariance matrix...")
        C1 = self.gen_sparse_cov(50, 50, rho=0.2, sp = 0.01)
        
        tau_gc = 0.1
        tau_mc = 0.05
        sd_inf =20
        sd_n = 0.5 
        be = 0.1 
        ga = 0.1

        np.random.seed(0)
        M, N = 20, 50
        Smin, Smax = 10, 20
        Svals = [np.random.randint(Smin, Smax) for i in range(M)]; print(Svals, sum(Svals))
        A_mean = np.random.rand(M, N)*3
        data_file = f"conn.p"        
        if not os.path.exists(data_file) or force:
            print(f"Generating random, sparse and weighted connectivity matrices...")
            np.random.seed(1); A1_0, details1_0 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=0, penalty=0, verbosity = 0)
            np.random.seed(1); A1_1, details1_1 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=5, penalty=10+0*np.arange(M), verbosity = 0)
            np.random.seed(1); A1_w, details1_w = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=5, penalty=np.arange(M), verbosity = 0)
            ob_arr = [ob.OlfactoryBulb(A, sd_inf, be, ga, tau_gc=tau_gc, tau_mc = tau_mc, verbosity=0, enforce_ga=True) for A in [A1_0, A1_1, A1_w]]
            
            data = {"A1_0": A1_0, "A1_1": A1_1, "A1_w": A1_w, "details1_0": details1_0, "details1_1": details1_1, "details1_w": details1_w, 
            "Svals": Svals, "sd_inf": sd_inf, "be": be, "ga": ga, "tau_gc": tau_gc, "tau_mc": tau_mc, "ob_arr": ob_arr}
    
            with open(data_file, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to {data_file}.")
        else:
            print(f"Loading data from {data_file}...")
            with open(data_file, "rb") as f:
                data_read = pickle.load(f)

            ob_arr = data_read["ob_arr"]
            A1_0 = data_read["A1_0"]
            A1_1 = data_read["A1_1"]
            A1_w = data_read["A1_w"]
            details1_0 = data_read["details1_0"]
            details1_1 = data_read["details1_1"]
            details1_w = data_read["details1_w"]
                        
        # Load the runs for the different bulbs and odours
        print("Loading data from run_odours sweep...")
        resp = {}
        for which_bulb in range(3):
            for which_odour in range(50):
                data_file = f"{project_path}/run_odours/bulb{which_bulb}/odour{which_odour}.p"
                with open(data_file, "rb") as f:
                    resp[(which_bulb, which_odour)] = pickle.load(f)

        print("Calculating correlations...")
        corrs = []
        for (bi, oi) in resp.keys():
            for si in range(20):
                ind_cmp = np.where((resp[bi,oi]["T"] > 0.5) * (resp[bi,oi]["T"] < 2))[0]
                R = resp[bi, oi]["La"][si][ind_cmp, :]
                rho = np.corrcoef(R.T)
                # Take the upper triangle of the correlation matrix
                rho = rho[np.triu_indices(rho.shape[0], k=1)]
                corrs.append(np.mean(rho))
        corrs = np.array(corrs).reshape((3, 50, 20))                    
        
        from run_odours import run_odours        
        which_odours = [[0,1,2], [0,1,2], [0,1,2]]

        data_file = "out1_.p"
        if not os.path.exists(data_file) or force:
            print("Running bulbs for a few odours...")        
            out1_0, out1_1, out1_w = [run_odours(which_ob, od_inds) for which_ob, od_inds in zip(ob_arr, which_odours)]
            with open(data_file, "wb") as f:
                pickle.dump((out1_0, out1_1, out1_w), f)
        else:
            print(f"Loading data from {data_file}...")
            with open(data_file, "rb") as f:
                out1_0, out1_1, out1_w = pickle.load(f)

        self.out1_0 = out1_0
        self.out1_1 = out1_1
        self.out1_w = out1_w

        self.ob_arr = ob_arr
        self.which_odours = which_odours
        self.corrs = corrs

        self.Svals = Svals
        self.N = N

        self.A1_0 = A1_0
        self.A1_1 = A1_1
        self.A1_w = A1_w
        
        self.details1_0 = details1_0
        self.details1_1 = details1_1
        self.details1_w = details1_w

        self.computed = True
        print("DONE PREPPING CONNECTIVITY DYNAMICS DATA.")
    

class InferenceDynamics(Computation):
    @staticmethod
    def gen_cov(N, n, rho):
        C = np.zeros((N,N))
        C[:n, :n] = rho
        for i in range(n):
            C[i, i] = 1
        #print(np.linalg.eigvalsh(C[:n, :n]))
        return C

    @staticmethod
    def single_run(A_mean, C, n = 5, S_min_max = [4,9], sd_inf = 20, sd_n = 0.1, be = 0.1, ga = 0.1, seed = 2):
        np.random.seed(seed)
        M, N = A_mean.shape
        Smin, Smax = S_min_max
        Svals = [np.random.randint(Smin, Smax) for i in range(M)]; 
        odour = od.Odour(np.arange(N)<n, A_mean)
        #y   = sum(A_mean[:, :5],axis=1) + randn(M,) * 0
        y  = odour.value_at(0)
        A0 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, 0* C)
        A1 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C * 1e-1)
        tau_gc = 0.1
        tau_mc = 0.05
        ob0 = ob.OlfactoryBulb(A0, sd_inf, be, ga, tau_gc=tau_gc, tau_mc = tau_mc, verbosity=0, enforce_ga=True)
        ob1 = ob.OlfactoryBulb(A1, sd_inf, be, ga, tau_gc=tau_gc, tau_mc = tau_mc, verbosity=0, enforce_ga=True)
        noise = sd_n * np.random.randn(*y.shape)
        res0 = ob0.run_exact(y + noise); res1 = ob1.run_exact(y+noise)
        return res0, res1, odour, noise, ob0, ob1
                
    def compute(self, force = False):
        print("PREPPING INFERENCE DYNAMICS FIGURE.")
        np.random.seed(5)
        M, N = 50, 200
        n = 5
        C = self.gen_cov(N, n, -0.24)
        ob.logger.setLevel(logging.ERROR)
        od.logger.setLevel(logging.ERROR)
        A_mean = np.random.randn(M, N)
        sd_inf_vals = [1, 2, 5, 10, 20, 50, 100]
        n_trials = 5
        results = []
        err_fun = lambda x: np.linalg.norm(x, 2)

        if not os.path.exists("inf_dyn.p") or force:
            keep = []
            for j, sd_inf in enumerate(sd_inf_vals):
                for i in range(n_trials):
                    res0, res1, odour, noise, ob0, ob1 = self.single_run(A_mean, C, S_min_max = [4,9], sd_inf = sd_inf, sd_n = 0.5, be = 0.1, ga = 0.1, seed = i)
                    x0 = res0["x"]
                    x1 = res1["x"]
                    x_true = np.arange(N) < n
                    err0 = err_fun(x0 - x_true)
                    err1 = err_fun(x1 - x_true)
                    results.append({"sd_inf": sd_inf, "trial": i, "x0err": err0, "x1err": err1})
                    print(f"sd_inf = {sd_inf}, trial = {i}, |x0 - x_true| = {err0:.3f}, |x1 - x_true| = {err1:.3f}")
                    if sd_inf == 20 and i == 0:
                        keep = [res0, res1, odour, noise, ob0, ob1]
    
            df = pd.DataFrame(results)
            res0, res1, odour, noise, ob0, ob1 = keep
            out1 = ob1.run_sister(odour.value_at(0.5) +noise, t_end = 3, dt=2e-4, keep_every=10)

            keep = keep[:2] # only keep res0 and res1
            
            with open("inf_dyn.p", "wb") as f:
                pickle.dump((keep, df, out1), f)

            print(f"Results saved to inf_dyn.p.")

        else:
            print(f"Loading results from inf_dyn.p")
            with open("inf_dyn.p", "rb") as f:
                keep, df, out1 = pickle.load(f)                                                    
    
        self.N = N
        self.n = n
        self.keep = keep
        self.df = df
        self.out1 = out1

        self.computed = True
        print("DONE PREPPING INFERENCE DYNAMICS FIGURE.")


class InferringThePrior(Computation):
    def compute(self):
        print("PREPPING PRIOR INFERENCE FIGURE...")
        from sisters_experiments import Experiment
        from essential_oils import EssentialOils

        sisters = Experiment()
        sisters.compute_responses()
        
        eo = EssentialOils(
            essential_oils_file = project_path + "/data/essential_oil_composition.csv", 
            smiles_file = project_path + "/data/smiles_names.csv",
        )

        sm_db = eo.smiles_db
        # Any overlap of the essential oils with the odours?
        sm_od = {od:sm_db.get_smiles(od) for mo,od in sisters.odour_names.items()}
        sm_od_list = [k[0] for k in sm_od.values() if k is not None]
        first = lambda x: x if x is None else x[0]
        sisters.odour_smiles = [first(sm_db.get_smiles(od)) for od in sisters.odour_names.values()]        

        # Compute the intersection of the two smiles lists
        overlap = sorted(list(set(sm_od_list).intersection(set(eo.smiles))))
        print("Overlap between essential oils and odours of",len(overlap), "items:", "\n".join(overlap))

        pairs_ods = []
        for i1, sm1 in enumerate(overlap[:-1]):
            for sm2 in overlap[i1+1:]:
                oils = eo.has_all_components([sm1, sm2])
                if len(oils) == 0:
                    continue
                ind1 = sisters.odour_smiles.index(sm1)
                ind2 = sisters.odour_smiles.index(sm2)
                pairs_ods.append((sm1, sm2, ind1, ind2, oils))
                if len(oils):
                    print(sm1, sm_db.get_name(sm1),  sm2, sm_db.get_name(sm2),  "co-occur in", oils)

        odours = sisters.odours
        vp = [sisters.vapour_pressures[od] for od in odours]

        C, C_glom = sisters.compute_covariance()
        Q = np.array([Si * np.abs(Ci) for Si, Ci in zip(sisters.S_vals, C_glom)]).sum(axis=0)
        ee,rr = np.linalg.eigh(Q)
        assert np.allclose(Q @ rr[:, -1], ee[-1] * rr[:, -1])
        r = rr[:, -1]
        
        self.r = r
        self.sisters = sisters
        self.vp = vp
        self.overlap = overlap
        self.pairs_ods = pairs_ods

        self.computed = True
        print("DONE PREPPING PRIOR INFERENCE FIGURE.")

    
