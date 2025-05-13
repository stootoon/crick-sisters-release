import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd

git_path = os.environ["GIT"]
sys.path.append(git_path)
from label_axes import label_axes

project_path = os.path.join(git_path, "crick-sisters-release")
sys.path.append(project_path)

art_path     = os.path.join(project_path, "art")
fig_path     = os.path.join(art_path, "figs")

plt.rcParams['figure.figsize']    = [8, 3]
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.style.use("default")

from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import olfactory_bulb as ob
import odours as od
import sisters_util as util

class ConnectivitySchematic:
    def __init__(self, args):
        self.args = args
        self.prep(args)

    def prep(self, args):
        print("Preparing connectivity schematic")
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

    def plot(self):
        print("Plotting connectivity schematic")
        n_rows, n_cols = 20, 13
        gs = GridSpec(n_rows, n_cols)
        plt.figure(figsize=(24, 10))
        
        ax_mean = plt.subplot(gs[:12,:4])
        im_file = art_path + "/sister_conn_mean.jpg"
        ax_mean.imshow(plt.imread(im_file))
        ax_mean.axis("off")
            
        ax_cov  = plt.subplot(gs[12:,:4])
        # Paste the svg image in ax_cov
        im_file = art_path + "/sister_conn_cov.jpg"
        ax_cov.imshow(plt.imread(im_file))
        ax_cov.axis("off")
        
        ax_angle = plt.subplot(gs[:10,4:8])
        im_file = art_path + "/angles.jpg"
        ax_angle.imshow(plt.imread(im_file))
        ax_angle.axis("off")
            
        ax_rot  = plt.subplot(gs[10:,4:8])
        im_file = art_path + "/rotated.jpg"
        ax_rot.imshow(plt.imread(im_file))
        ax_rot.axis("off")
        
        nr = [7,14]
        
        ax_conn1 = plt.subplot(gs[:nr[0]-1,  8:11])
        ax_conn2 = plt.subplot(gs[nr[0]:nr[1]-1, 8:11])
        ax_mean2 = plt.subplot(gs[nr[1]:,  8:11])
        
        ax_cov1  = plt.subplot(gs[:nr[0]-1,11:])
        ax_cov2  = plt.subplot(gs[nr[0]:nr[1]-1,11:])

        Svals = self.Svals
        details = self.details
        parent_glom = self.parent_glom
        sis_id = self.sis_id
        A1 = self.A1
        A_mean = self.A_mean
        sis_cols = self.sis_cols
        glom_cols = self.glom_cols
                
        n_sis_plot = sum(Svals[:3])
        n_sis = sum(Svals)
        # Plot the connectivity matching angles
        ax_conn1.matshow(details["W"].T[:,:n_sis_plot]); ax_conn1.axis("auto")  
        ax_conn1.set_ylabel("Feature")
        #ax_conn1.set_xlabel("Parent Glom. : Sister")
        ax_conn1.set_xticks(np.arange(n_sis_plot))
        ax_conn1.set_xticklabels([(f"{p}" if s==0 else "") + f":{s}" for p,s in zip(parent_glom, sis_id)][:n_sis_plot], fontsize=8)
        # Put the xtick labels at the bottom
        ax_conn1.xaxis.set_label_position('bottom')
        # Put the xticks at the bottom
        ax_conn1.xaxis.set_ticks_position('bottom')
        
        ax_cov1.matshow(np.cov(details["W"].T, bias=True)); ax_cov1.set_xticks([]); ax_cov1.set_yticks([])
        ax_cov1.set_xlabel("Feature"); #ax_cov1.set_ylabel("Odour")
        
        # Plot the output connectivity
        ax_conn2.matshow(details["Wsol"].T[:,:n_sis_plot]); ax_conn2.axis("auto")
        ax_conn2.set_ylabel("Feature")
        ax_conn2.set_xlabel("Input Channel : Sister")
        ax_conn2.set_xticks(np.arange(n_sis_plot))
        ax_conn2.set_xticklabels([(f"{p}" if s==0 else "") + f":{s}" for p,s in zip(parent_glom, sis_id)][:n_sis_plot], fontsize=8)
        # Put the xtick labels at the bottom
        ax_conn2.xaxis.set_label_position('bottom')
        # Put the xticks at the bottom
        ax_conn2.xaxis.set_ticks_position('bottom')
        
        ax_cov2.matshow(np.cov(details["Wsol"].T, bias=True)); ax_cov2.set_xticks([]); ax_cov2.set_yticks([])
        ax_cov2.set_xlabel("Feature"); #ax_cov2.set_ylabel("Odour")
        
        A1flat = np.vstack(A1).T
        ax_mean2.scatter(np.arange(n_sis), A1flat[0] - A_mean[parent_glom,0], c = sis_cols, alpha=0.25); 
        x1 = np.cumsum(Svals)
        x0 = [0] + list(x1[:-1])
        y0 = [np.mean(A1flat[0,x0i:x1i]) - A_mean[i,0] for i,(x0i,x1i) in enumerate(zip(x0,x1))]
        for x0i,x1i,y0i,ci in zip(x0,x1,y0,glom_cols):
            ax_mean2.plot([x0i,x1i-1],[y0i,y0i],color=ci, lw=2)
        ax_mean2.set_xticks(x0 + (np.array(Svals)-1)/2, labels=np.arange(1,len(Svals)+1))
        ax_mean2.set_xlim(-0.5, n_sis_plot-0.5)
        ax_mean2.set_xlabel("Input Channel")
        ax_mean2.set_ylabel("Weights to GC0", labelpad=-5)
        
        fontsize = 12
        for ax in [ax_conn1, ax_conn2, ax_cov1, ax_cov2, ax_mean2]:  # Replace with your axes
            ax.tick_params(labelsize=fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
            #ax.title.set_fontsize(fontsize)
        
        label_axes.label_axes([ax_mean, ax_cov, ax_angle, ax_rot, ax_conn1, ax_conn2, ax_mean2], "ABCDEFG", fontsize=24, fontweight="bold")
        
        plt.tight_layout()
        #plt.show()
        output_file = os.path.join(args.output_dir, "connectivity.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")

class InferenceDynamics:
    def __init__(self, args):
        self.args = args
        self.prep(args)

    @staticmethod
    def mystem(x, y, *args, **kwargs):
        return np.plot([x, x], [0*np.array(y), y], *args, **kwargs)

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
        random.seed(seed)
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
        noise = sd_n * randn(*y.shape)
        res0 = ob0.run_exact(y + noise); res1 = ob1.run_exact(y+noise)
        return res0, res1, odour, noise, ob0, ob1
        
        
    def prep(self, args, force = False):
        np.random.seed(5)
        M, N = 50, 200
        n = 5
        C = self.gen_cov2(N, n, -0.24)
        ob.logger.setLevel(logging.ERROR)
        od.logger.setLevel(logging.ERROR)
        A_mean = np.random.randn(M, N)
        sd_inf_vals = [10,20,50] #[1, 2, 5, 10, 20, 50, 100]
        n_trials   = 2#5
        results = []
        err_fun = lambda x: np.linalg.norm(x, 2)
        if not os.path.exists("results.p") or force:
            keep = []
            for j, sd_inf in enumerate(sd_inf_vals):
                for i in range(n_trials):
                    res0, res1, odour, noise, ob0, ob1 = self.single_run(A_mean, C, S_min_max = [4,9], sd_inf = sd_inf, sd_n = 0.5, be = 0.1, ga = 0.1, seed = i)
                    x0 = res0["x"]
                    x1 = res1["x"]
                    x_true = arange(N) < n
                    err0 = err_fun(x0 - x_true)
                    err1 = err_fun(x1 - x_true)
                    results.append({"sd_inf": sd_inf, "trial": i, "x0err": err0, "x1err": err1})
                    print(f"sd_inf = {sd_inf}, trial = {i}, |x0 - x_true| = {err0:.3f}, |x1 - x_true| = {err1:.3f}")
                    if sd_inf == 20 and i == 0:
                        keep = [res0, res1, odour, noise, ob0, ob1]
    
            df = pd.DataFrame(results)

            out1 = ob1.run_sister(odour.value_at(0.5) +noise, t_end = 3, dt=2e-4, keep_every=10)

            with open("results.p", "wb") as f:
                pickle.dump((df, keep, out1), f)
            print("Results saved to results.p.")
        else:
            with open("results.p", "rb") as f:
                df, keep, out1 = pickle.load(f)

            print("Results loaded from results.p.")
                    
        res0, res1, odour, noise, ob0, ob1 = keep        

    def plot(self):
        plt.figure(figsize=(8, 3))
        gs = GridSpec(1, 6)
        ax_inf = plt.subplot(gs[0,:2])
        x_true = np.arange(N) < n
        x0 = keep[0]["x"]
        x1 = keep[1]["x"]
        # Make a plot where we show the first 10 elements of the true x, the first 10 elements of x0, and the first 10 elements of x1
        # We do these as stem plots, and staggered to avoid overlap
        h0 = self.mystem(np.arange(10), x_true[:10],   "o-", label = "x_true", color = "gray", markersize=2, lw=2)
        h1 = self.mystem(np.arange(10) + 0.2, x1[:10], "o-", label = "x1", color = "C1", markersize=2, lw=2)
        h2 = self.mystem(np.arange(10) + 0.4, x0[:10], "o-", label = "x0", color = "C0", markersize=2, lw=2)
        plt.plot(plt.xlim(),[0,0], "k--", lw=0.5)
        plt.legend([h0[0], h1[0], h2[0]], ["True", "Coact.", "Indep."], loc = "upper right", fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("Feature Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        
        ax_err = plt.subplot(gs[0,2:4])
        # Make a plot whose x-axis is σ_inf, and whose y-axis is the mean of the errors over the trials
        # Make it a line plot, connecting x0err, and a line plot connecting x1err, and use log scale on the y-axis
        # Add error bars to show the standard deviation of the errors over the trials
        df.groupby("sd_inf").mean().plot(y = ["x0err", "x1err"], logx = True, logy = True, yerr = df.groupby("σ_inf").std(), marker = "o", ax = ax_err)
        plt.gca().legend(["Indep.", "Coact."],  fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("$\sigma_{inf}$", fontsize=14)
        plt.ylabel("Error", fontsize=14, labelpad=-10)
        # Get the y-axis ticks and convert them to decimal notation
        plt.tight_layout()
        
        ax_tc = plt.subplot(gs[0,-2:])
        plt.plot(out1["T"], out1["X"][:,:10])
        plt.xlim(out1["T"][0], out1["T"][-1])
        x_exact = keep[1]["x"][:10]
        # Put red triangles < at the right most edge of the plot at the target values
        plt.plot(out1["T"][-1]*np.ones(10), x_exact, "r<", markersize=12)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.ylim([-0.025,1]); grid(True)
        label_axes.label_axes([ax_inf, ax_err, ax_tc], "ABC", fontsize=12,fontweight="bold")
        plt.savefig(os.path.join(fig_path , "inference.pdf"), bbox_inches="tight")        



from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Make figures for the paper.")
    parser.add_argument("-f", "--fig", type=str, default="all", help="Figure to make")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    fig_name = args.fig.lower()
    # Fig name should be either all, or a comma-separted list of integers
    which_figs = map(int, fig_name.split(",")) if fig_name != "all" else range(1, 5)
    # Check that all requested figures are valid
    assert all([f in range(1, 5) for f in which_figs]), f"Invalid figure number(s): {fig_name}"
    # Check that the output directory exists
    assert os.path.exists(args.output_dir), f"Output directory does not exist: {args.output_dir}"
    # Check that the output directory is a directory
    assert os.path.isdir(args.output_dir), f"Output path is not a directory: {args.output_dir}"
    # Check that the output directory is writable
    assert os.access(args.output_dir, os.W_OK), f"Output directory is not writable: {args.output_dir}"

    for fig_num in which_figs:
        print(f"Making Figure {fig_num}...")
        if fig_num == 1:            
            cs = ConnectivitySchematic(args)
            cs.plot()
        elif fig_num == 2:
            obj = InferenceDynamics(args)
            obj.plot()
            
            
            
            
    
        
    

