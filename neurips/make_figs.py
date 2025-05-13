import os, sys, logging, pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
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

class BaseFigure:
    def __init__(self):
        pass

    def prep(self):
        raise NotImplementedError("prep() method not implemented")

    def plot(self):
        raise NotImplementedError("plot() method not implemented")

class ConnectivitySchematic(BaseFigure):
    def __init__(self, args):
        self.args = args
        self.prep(args)

    def prep(self):
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
        output_file = os.path.join(self.args.output_dir, "schematic.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")

class ConnectivityDynamics(BaseFigure):
    def __init__(self, args):
        self.args = args
        self.prep()

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
    def spines_off(ax = gca(), which=["top", "right"]):
        for w in which:
            ax.spines[w].set_visible(False)
        return ax

    @staticmethod
    def plot_diversity_cumfrac(c, cols = ["tan", "wheat", "chocolate"][::-1], thresh =[0, 0.7, 0.9], ax = None, **kwargs):
        ax = plt.gca() if ax is None else ax
        r = c.flatten()
        x = np.arange(len(r))/len(r)
        y = np.array(sorted(r))
        for i, t in enumerate(thresh):
            ax.plot(x[y > t], y[y > t], color = cols[i], **kwargs)

    @staticmethod
    def plot_diversity_bars(c, cols = ["tan", "wheat", "chocolate"][::-1], thresh = [0,0.7,0.9], ax = None):
        ord = np.argsort(np.mean(c,axis=0))
        ax = plt.gca() if ax is None else ax
        for i in range(len(ord)):
            ci = c[:, ord[i]]
            fracs = [np.mean(ci >= th)*100 for th in thresh]
            for fi, f in enumerate(fracs):
                ax.plot([i, i],[0,f], color=cols[fi], lw=5)
        xlabel("input channel (sorted)")
        ylabel("stimulus %")

    @staticmethod
    def plot_W1(d, vlim = 1/4, cmap="bwr", ls = "k", ax = None):
        ax = gca() if ax is None else ax
        ax.matshow(d["W1"].T, vmin=-1/4,vmax=1/4, cmap="bwr")
        yl = plt.ylim()
        xl = plt.xlim()
        xx = np.cumsum(Svals)
        plt.plot([xx, xx], [-1+0*xx, (N)+0*xx], ls, lw=1)
        plt.xlim(xl); plt.ylim(yl)
                
    def prep(self, force = False):
        print("Preparing figure...")
        print("\tGenerating sparse covariance matrix...")
        C1 = gen_sparse_cov(50, 50, rho=0.2, sp = 0.01)
        
        tau_gc = 0.1
        tau_mc = 0.05
        sd_inf =20
        sd_n = 0.5 
        be = 0.1 
        ga = 0.1

        random.seed(0)
        M, N = 20, 50
        Smin, Smax = 10, 20
        Svals = [random.randint(Smin, Smax) for i in range(M)]; print(Svals, sum(Svals))
        A_mean = rand(M, N)*3
        data_file = f"conn.p"        
        if not os.path.exists(data_file) or force:
            print(f"Generating random, sparse and weighted connectivity matrices...")
            random.seed(1); A1_0, details1_0 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=0, penalty=0, verbosity = 0)
            random.seed(1); A1_1, details1_1 = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=5, penalty=10+0*arange(M), verbosity = 0)
            random.seed(1); A1_w, details1_w = ob.OlfactoryBulb.generate_connectivity(M, N, Svals, A_mean, sd_inf**2 * C1 * 1e-1, random_rotation=True, return_details=True, sparsify=5, penalty=arange(M), verbosity = 0)
            ob_arr = [ob.OlfactoryBulb(A, sd_inf, be, ga, tau_gc=tau_gc, tau_mc = tau_mc, verbosity=0, enforce_ga=True) for A in [A1_0, A1_1, A1_w]]
            
            data = {"A1_0": A1_0, "A1_1": A1_1, "A1_w": A1_w, "details1_0": details1_0, "details1_1": details1_1, "details1_w": details1_w, 
            "Svals": Svals, "σ_inf": σ_inf, "β": β, "γ": γ, "tau_gc": tau_gc, "tau_mc": tau_mc, "ob_arr": ob_arr}
    
            with open(data_file, "wb") as f:
                pickle.dump(data, f)

            print(f"Saved data to {data_file}.")
        else:
            print(f"Loading data from {data_file}...")
            with open(data_file, "rb") as f:
                data_read = pickle.load(f)

            ob_arr = data_read["ob_arr"]
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

        from run_odours import run_odours        

        which_odours = [[0,1,2], [0,1,2], [0,1,2]]

        print("Running bulbs for a few odours...")
        self.out1_0, self.out1_1, self.out1_w = [run_odours(which_ob, od_inds) for which_ob, od_inds in zip(ob_arr, which_odours)]
        self.details1_0 = details1_0
        self.details1_1 = details1_1
        self.details1_w = details1_w
        print("Done prepping for figure.")

    def plot(self):
        out1_0 = self.out1_0
        out1_1 = self.out1_1
        out1_w = self.out1_w

        ob_arr = self.ob_arr

        details1_0 = self.details1_0
        details1_1 = self.details1_1
        details1_r = self.details1_r
        details1_w = self.details1_w
                
        n_rows_per_conn = 1
        n_rows_per_resp = 2
        n_rows_per_spacer = 1
        n_rows   = n_rows_per_resp * (4 * n_rows_per_resp + 3 * n_rows_per_conn) + n_rows_per_spacer# 1 row for real data, 3 for the resps from the different conn models
        n_odours = len(out1_0)
        n_cols   = n_odours + 1 + 1 # 1 for the distribution histogram
        plt.figure(figsize=(12, 20))
        gs = GridSpec(n_rows, n_cols)
        lab_ax = []
        ax = {} 
        ax["resp"] = {"real":[]}
        # First row is for the real data
        new_ax = plt.subplot(gs[0:2,:3])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_resps.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)
        
        new_ax = plt.subplot(gs[0:2,3])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_cumfrac.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)
        
        new_ax = plt.subplot(gs[0:2,4])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_bars.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)
        
        row_offset = 1 * n_rows_per_resp
        names = ["Random", "Sparse", "Weighted"]
        
        for i, (name, out, (glom, sis_inds), od_inds) in enumerate(zip(names, [out1_0, out1_1, out1_r],[(0,[0,1]), (0,[0,1]), (0,[0,1])], which_odours)):
            # sis_inds are in (glom, sis1, sis2) format
            # od_inds are in ind1,ind2 format
            rows = list(row_offset + arange(n_rows_per_resp))
            ax["resp"][f"sim{i}"] = []
            for j, (od_resp, od_ind) in enumerate(zip(out, od_inds)):
                new_ax = plt.subplot(gs[slice(rows[0], rows[1]+1), j])
                ax["resp"][f"sim{i}"].append(new_ax)
                t = od_resp["T"]
                la= od_resp["La"]
                new_ax.plot(t-0.5, la[glom][:, sis_inds[0]] * 1000, label=f"g{glom}s{sis_inds[0]}")
                new_ax.plot(t-0.5, la[glom][:, sis_inds[1]] * 1000, label=f"g{glom}s{sis_inds[1]}")
                i==0 and new_ax.legend(fontsize=8, frameon=False, labelspacing=0)
                i==0 and new_ax.set_title(f"Odour {od_ind}")
                #i != 2 and new_ax.set_xticklabels([])
                new_ax.set_xlim(-0.5,1.5)
                yl = gca().get_ylim()
                yl_m = np.mean(yl)
                new_ax.set_ylim(yl_m-2.5,yl_m + 2.5)
                new_ax.add_patch(patches.Rectangle((0,yl_m-2.5), 1, 5, linewidth=0, facecolor=(0,0,0,0.1)))
                (i == len(out)-1) and new_ax.set_xlabel("Time (sec)")
                (j == 0) and new_ax.set_ylabel(name, fontsize=14, labelpad=-5)
                new_ax.set_xticks([0, 0.5, 1])
                spines_off(new_ax)
                if j == 0:
                    lab_ax.append(new_ax)
            
            new_ax = plt.subplot(gs[slice(rows[0], rows[1]+1), j+1])
            ax["resp"][f"sim{i}"].append(new_ax)
            self.plot_diversity_cumfrac(corrs[i], ax=new_ax, lw=3)
            (i == 2) and new_ax.set_xlabel("temporal similarity index")
            new_ax.set_ylabel("cumu. frac. glom-odour", fontsize=8, labelpad=-1)
            new_ax.tick_params(axis='both', labelsize=10)
            new_ax.set_xticks([0, 0.3, 0.7, 1])
            new_ax.set_yticks([0, 0.31, 0.69, 1])
            self.spines_off(new_ax)
            lab_ax.append(new_ax)
        
            new_ax = plt.subplot(gs[slice(rows[0], rows[1]+1), j+2])
            ax["resp"][f"sim{i}"].append(new_ax)
            self.plot_diversity_bars(corrs[i], ax=new_ax)
            # The yticks are in fractions of 1, so we need to convert them to percentages
            new_ax.set_yticks([0, 25, 50, 75, 100])
            new_ax.set_ylabel("glom %", labelpad=-5)
            # Set the tick fontsize to 8
            new_ax.tick_params(axis='both', labelsize=10)
            (i < 2) and new_ax.set_xlabel("")
            self.spines_off(new_ax)
            lab_ax.append(new_ax)
            
        
            row_offset = rows[-1]+1
        
        row_offset += n_rows_per_spacer
        
        ax["conn"] = []
        for i, (name, details) in enumerate(zip(["Random", "Sparse", "Weighted"], [details1_0, details1_1, details1_w])):
            rows = list(row_offset + arange(n_rows_per_conn))
            new_ax = plot.subplot(gs[slice(rows[0], rows[-1]+1), :-1])
            ax["conn"].append(new_ax)
            self.plot_W1(details, ax=new_ax, ls = "k:")
            new_ax.axis("auto")
            new_ax.set_xlim(0, sum(ob_arr[0].S))
            new_ax.set_ylim(0, ob_arr[0].N-1)
            new_ax.set_xticks([]); new_ax.set_yticks([])
            row_offset = rows[-1]+1
            i == 2 and new_ax.set_xlabel("Mitral cell")
            new_ax.set_ylabel(f"GC", fontsize=10)
            # Left align the title
            new_ax.set_title(name, fontsize=14, loc="left")
            lab_ax.append(new_ax)
        
        tight_layout(h_pad=-0.25, w_pad = 0.5)    
        align_x = [[0,3,6,9,12,13,14], [1, 4,7,10], [2, 5,8,11]]
        align_y = [[0,1,2], [3,4,5], [6,7,8],[9,10,11]]
        label_axes.label_axes(lab_ax, "ABCDEFGHIJKLMNOP", fontsize=12,fontweight="bold", align_x = align_x, align_y = align_y, dx=-0.001, dy=+0.001)

        output_file = os.path.join(self.args.output_dir, "conn_dynamics.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        
                                    
class InferenceDynamics(BaseFigure):
    def __init__(self, args):
        self.args = args
        self.prep()

    @staticmethod
    def mystem(x, y, *args, **kwargs):
        return plt.plot([x, x], [0*np.array(y), y], *args, **kwargs)

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
        
        
    def prep(self, force = False):
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

    def plot(self):
        N = self.N
        n = self.n
        keep = self.keep
        df   = self.df
        out1 = self.out1
        
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
        plt.legend([h0[0], h1[0], h2[0]], ["True", "Corr.", "Indep."], loc = "upper right", fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("Feature Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        
        ax_err = plt.subplot(gs[0,2:4])
        # Make a plot whose x-axis is σ_inf, and whose y-axis is the mean of the errors over the trials
        # Make it a line plot, connecting x0err, and a line plot connecting x1err, and use log scale on the y-axis
        # Add error bars to show the standard deviation of the errors over the trials
        df.groupby("sd_inf").mean().plot(y = ["x0err", "x1err"], logx = True, logy = True, yerr = df.groupby("sd_inf").std(), marker = "o", ax = ax_err)
        plt.gca().legend(["Indep.", "Corr."],  fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("sd$_{inf}$", fontsize=14)
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
        plt.ylim([-0.025,1]); plt.grid(True)
        label_axes.label_axes([ax_inf, ax_err, ax_tc], "ABC", fontsize=12,fontweight="bold")

        output_file = os.path.join(self.args.output_dir, "inf_dynamics.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")


class InferringThePrior(BaseFigure):
    def __init__(self, args):
        self.args = args
        self.prep()

    def prep(self):
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
    

    def plot(self):
        sisters = self.sisters
        vp = self.vp
        r  = self.r
        overlap = self.overlap
        pairs_ods = self.pairs_ods

        vp_scales = 2+0.5/(np.array(vp) + 1e-1)
        scales = [("Constant", 1),
                  ("f(Vapour Pressure)", vp_scales),
                  ("1/Eigenvector", 1/r),
        ]
        C_scales = [sisters.compute_covariance(scales=sc)[0] for name, sc in scales]
        
        fig = plt.figure(figsize=(16,6))
        ax_lab = []
        for i, ((name, scale),Ci) in enumerate(zip(scales, C_scales)):
            ax = plt.subplot(1, 3, i+1)
        
            if i == 0:
                Z = linkage(Ci, method="complete", metric="correlation")
                leaf_order = dendrogram(Z, no_plot=True)["leaves"]
                sdi = np.std(Ci)
            Ci = Ci/ np.std(Ci) * sdi
        
            if i == 0:
                vmin, vmax = np.percentile(abs(Ci), [1, 99])
            im =  plt.matshow(Ci[leaf_order][:, leaf_order], cmap="bwr", vmin = -vmax, vmax = vmax, fignum=False); #colorbar()
            
            # Mark the overlaps with a black edged rectangle wih transparent fill
            for sm1, sm2, ind1, ind2, oils in pairs_ods:
                if ind1 not in leaf_order or ind2 not in leaf_order:
                    continue
                i1 = leaf_order.index(ind1)
                i2 = leaf_order.index(ind2)
                ax.add_patch(Rectangle((i1-0.5, i2-0.5), 1, 1, edgecolor="black", facecolor="none", lw=0.5))
        
                #ax.add_patch(Rectangle((i1-0.5,i2-0.5), 1, 1, color="black", alpha=0.3))
                
            for ii, li in enumerate(leaf_order):
                if sisters.odour_smiles[li] in overlap:
                    ax.axvline(ii, color="black", ls=":", lw=0.5)
                    ax.axhline(ii, color="black", ls=":", lw=0.5)
                #ax.text(i1+0.5, i2+0.5, ", ".join(oils), ha="center", va="center", fontsize=8)
        
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.4)
            fig.colorbar(im, cax=cax, orientation="horizontal")    
            #fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
        
            od_labs = [sisters.odour_names[sisters.odours[i]] for i in leaf_order]
            ax.set_xticks(range(len(Ci)));
            ax.set_xticklabels(od_labs, rotation=90, fontsize=7)
            ax.set_yticks(range(len(Ci)))    
            ax.set_yticklabels(od_labs if i == 0 else [], rotation=0, fontsize=7)
            ax.set_title(f"Concentrations ~ {name}", fontsize=14)
            ax_lab.append(ax)
        
        label_axes.label_axes(ax_lab, "ABC", fontsize=16,fontweight="bold")

        output_file = os.path.join(self.args.output_dir, "inferred_priors.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        
    
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Make figures for the paper.")
    parser.add_argument("-f", "--fig", type=str, default="all", help="Figure to make")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    fig_name = args.fig.lower()
    # Fig name should be either all, or a comma-separted list of integers
    which_figs = list(map(int, fig_name.split(","))) if fig_name != "all" else range(1, 5)
    # Check that all requested figures are valid
    assert all([f in range(1, 5) for f in which_figs]), f"Invalid figure number(s): {fig_name}"
    # Check that the output directory exists
    assert os.path.exists(args.output_dir), f"Output directory does not exist: {args.output_dir}"
    # Check that the output directory is a directory
    assert os.path.isdir(args.output_dir), f"Output path is not a directory: {args.output_dir}"
    # Check that the output directory is writable
    assert os.access(args.output_dir, os.W_OK), f"Output directory is not writable: {args.output_dir}"

    print(f"Making figures {which_figs} in {args.output_dir}...")
    
    for fig_num in which_figs:
        print(f"Making Figure {fig_num}...")
        if fig_num == 1:            
            cs = ConnectivitySchematic(args)
            cs.plot()
        elif fig_num == 2:
            obj = InferenceDynamics(args)
            obj.plot()
        elif fig_num == 3:
            pass
        elif fig_num == 4:
            obj = InferringThePrior(args)
            obj.plot()
            pass
        else:
            raise ValueError(f"Invalid figure number: {fig_num}")

    print("ALLDONE")
        
            
            
            
    
        
    

