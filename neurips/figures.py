import os, sys, logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import compute

git_path = os.environ["GIT"]
sys.path.append(git_path)

try:
    from label_axes import label_axes
except ImportError:
    # Define a dummy function if label_axes is not available
    print("label_axes module not found. Using dummy function.")
    class label_axes:
        @staticmethod
        def label_axes(*args, **kwargs):
            pass
        
project_path = os.path.join(git_path, "crick-sisters-release")
sys.path.append(project_path)

art_path     = os.path.join(project_path, "art")
fig_path     = os.path.join(art_path, "figs")

plt.rcParams['figure.figsize']    = [8, 3]
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.style.use("default")

from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from mpl_toolkits.axes_grid1 import make_axes_locatable

def spines_off(ax = plt.gca(), which=["top", "right"]):
    for w in which:
        ax.spines[w].set_visible(False)
    return ax

class Figure:
    def __init__(self, plot_data):
        assert hasattr(plot_data, "computed"), "plot_data must have a 'computed' attribute"
        self.plot_data = plot_data

    def compute_and_plot(self, args):
        if not self.plot_data.computed:
            self.plot_data.compute()
        self.plot(args, self.plot_data)
        
    @classmethod
    def plot(cls, args, plot_data):
        raise NotImplementedError("plot() method not implemented")

class ConnectivitySchematic(Figure):
    label_fontsize = 40
    axis_label_fontsize = 30
    
    @classmethod
    def plot(cls, args, plot_data):        
        print("PLOTTING CONNECTIVITY SCHEMATIC")
        n_rows, n_cols = 24, 16
        gs = GridSpec(n_rows, n_cols)
        plt.figure(figsize=(30, 24))
        
        ax_mean = plt.subplot(gs[:12,:4])
        im_file = art_path + "/sister_conn_mean.jpg"
        ax_mean.imshow(plt.imread(im_file))
        ax_mean.axis("off")
            
        ax_cov  = plt.subplot(gs[:12,4:8])
        # Paste the svg image in ax_cov
        im_file = art_path + "/sister_conn_cov.jpg"
        ax_cov.imshow(plt.imread(im_file))
        ax_cov.axis("off")
        
        ax_angle = plt.subplot(gs[:12,8:12])
        im_file = art_path + "/angles.jpg"
        ax_angle.imshow(plt.imread(im_file))
        ax_angle.axis("off")
            
        ax_rot  = plt.subplot(gs[:12,12:16])
        im_file = art_path + "/rotated.jpg"
        ax_rot.imshow(plt.imread(im_file))
        ax_rot.axis("off")
        
        nr = [7,14]

        ax = {"conn": []}
        lab_ax = []
        row_offset = 10
        n_rows_per_conn = 3
        n_cols_per_conn = 11
        n_cols_per_cov = 2
        n_cols_per_aff = 2
        for i, (name, details, A) in enumerate(zip(
                ["Random", "Sparse", "Weighted"],
                [plot_data.details1_0, plot_data.details1_1, plot_data.details1_w],
                [plot_data.A1_0, plot_data.A1_1, plot_data.A1_w]
        )):
            aff, C = compute.ConnectivityDynamics.compute_affinity_and_correlation_from_weights(A)
            if i == 0:
                link = linkage(C, method='ward')
                order = leaves_list(link)
            
            cols_offset = 0
            rows = list(row_offset + np.arange(n_rows_per_conn))
            row_slice = slice(rows[0], rows[-1]+1)
            new_ax = plt.subplot(gs[row_slice, :n_cols_per_conn])
            ConnectivityDynamics.plot_W1(details, plot_data.Svals, plot_data.N, row_order = order, ax=new_ax, ls = "k:", lw=2)
            new_ax.axis("auto")
            new_ax.set_xlim(0, sum(plot_data.Svals))
            new_ax.set_ylim(0, plot_data.N-1)
            new_ax.set_xticks([]); new_ax.set_yticks([])
            row_offset = rows[-1]+1
            i == 2 and new_ax.set_xlabel("Mitral cell", fontsize=cls.axis_label_fontsize)
            new_ax.set_ylabel(f"Latent", fontsize=cls.axis_label_fontsize)
            # Left align the title
            #new_ax.set_title(name, fontsize=18, loc="right", verticalalignment="top")
            lab_ax.append(new_ax)
            ax["conn"].append(new_ax)
            
            cols_offset += n_cols_per_conn
            new_ax = plt.subplot(gs[row_slice, cols_offset:cols_offset + n_cols_per_cov])
            Wsol = details["Wsol"]
            new_ax.matshow(C[order[:10]][:, order[:10]], cmap=cm.bone_r)
            new_ax.set_xticks([]); new_ax.set_yticks([]);
            new_ax.set_ylabel("Latent", fontsize=cls.axis_label_fontsize)
            new_ax.set_xlabel("Latent", fontsize=cls.axis_label_fontsize)
            ax["conn"].append(new_ax)
            
            cols_offset += n_cols_per_cov+1
            lab_ax.append(new_ax)
            new_ax = plt.subplot(gs[row_slice, cols_offset:cols_offset + n_cols_per_aff])
            if i == 0:
                aff_order = np.argsort(aff[0])
            new_ax.plot(aff.T[aff_order,:2])
            new_ax.set_xlabel("Latent", fontsize=cls.axis_label_fontsize)
            # Set the tick sizes
            new_ax.tick_params(axis='both', labelsize=20)
            
            new_ax.set_ylabel("Affinity", fontsize=cls.axis_label_fontsize)
            spines_off(new_ax)
            ax["conn"].append(new_ax)
                
        plt.tight_layout(h_pad=0.02)
        
        align_y = [list(range(4))] + [[ii+4+offset for ii in range(3)] for offset in range(0,9,3)]
        align_x = [[0,4,7,10],[3,5,8,11],[6,9,12]]

        post_align_dx = [0]*13
        post_align_dx[3] += 0.05
        for iax in [5,6,8,9,11,12]:
            post_align_dx[iax] -= 0.02
        
        label_axes.label_axes([ax_mean, ax_cov, ax_angle, ax_rot] + ax["conn"],
                              ["A", "B", "C", "D","Ei" , "Eii", "Eiii", "Fi", "Fii", "Fiii", "Gi", "Gii", "Giii"],
                              align_y = align_y,
                              align_x = align_x,
                              dx = -0.02,
                              post_align_dx = post_align_dx,
                              fontsize=cls.label_fontsize, fontweight="bold")
        

        #plt.show()
        output_file = os.path.join(args.output_dir, "schematic.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        print(f"DONE PLOTTING CONNECTIVITY SCHEMATIC.")

class ConnectivityDynamics(Figure):
    panel_label_fontsize = 20
    axis_label_fontsize = 16
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
        plt.xlabel("input channel (sorted)")
        plt.ylabel("stimulus %")

    @staticmethod
    def plot_W1(d, Svals, N, row_order = None, vlim = 1/4, cmap="bwr", ls = "k", lw=1, ax = None):
        ax = gca() if ax is None else ax
        W1 = d["W1"]
        if row_order is None:
            row_order = np.arange(W1.shape[0])
        ax.matshow((W1.T)[row_order], vmin=-1/4,vmax=1/4, cmap="bwr")
        yl = plt.ylim()
        xl = plt.xlim()
        xx = np.cumsum(Svals)
        plt.plot([xx, xx], [-1+0*xx, (N)+0*xx], ls, lw=lw)
        plt.xlim(xl); plt.ylim(yl)

    @classmethod
    def plot(cls, args, plot_data):
        print("PLOTTING CONNECTIVITY DYNAMICS FIGURE...")
        out1_0 = plot_data.out1_0
        out1_1 = plot_data.out1_1
        out1_w = plot_data.out1_w
    
        ob_arr = plot_data.ob_arr
        which_odours = plot_data.which_odours
        corrs = plot_data.corrs

        details1_0 = plot_data.details1_0
        details1_1 = plot_data.details1_1
        details1_w = plot_data.details1_w
                
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

        row_offset = 0 * n_rows_per_resp
        names = ["Random", "Sparse", "Weighted"]
        ax["resp"] = {}
        resp_cols = ["forestgreen", "olive"]
        for i, (name, out, (glom, sis_inds), od_inds) in enumerate(zip(names, [out1_0, out1_1, out1_w],[(0,[0,1]), (0,[0,1]), (0,[0,1])], which_odours)):
            # sis_inds are in (glom, sis1, sis2) format
            # od_inds are in ind1,ind2 format
            rows = list(row_offset + np.arange(n_rows_per_resp))
            ax["resp"][f"sim{i}"] = []
            for j, (od_resp, od_ind) in enumerate(zip(out, od_inds)):
                new_ax = plt.subplot(gs[slice(rows[0], rows[1]+1), j])
                ax["resp"][f"sim{i}"].append(new_ax)
                t = od_resp["T"]
                la= od_resp["La"]
                new_ax.plot(t-0.5, la[glom][:, sis_inds[0]] * 1000, color = resp_cols[0], lw=2, label=f"ch{glom}s{sis_inds[0]}")
                new_ax.plot(t-0.5, la[glom][:, sis_inds[1]] * 1000, color = resp_cols[1], lw=2, label=f"ch{glom}s{sis_inds[1]}")
                i==0 and new_ax.legend(fontsize=8, frameon=False, labelspacing=0)
                i==0 and new_ax.set_title(f"Odour {od_ind}")
                #i != 2 and new_ax.set_xticklabels([])
                new_ax.set_xlim(-0.5,1.5)
                yl = plt.gca().get_ylim()
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
            cls.plot_diversity_bars(corrs[i], ax=new_ax)
            # The yticks are in fractions of 1, so we need to convert them to percentages
            new_ax.set_yticks([0, 25, 50, 75, 100])
            new_ax.set_ylabel("% stimuli", labelpad=-5)
            # Set the tick fontsize to 8
            new_ax.tick_params(axis='both', labelsize=10)
            (i < 2) and new_ax.set_xlabel("")
            spines_off(new_ax)
            lab_ax.append(new_ax)
                    
            new_ax = plt.subplot(gs[slice(rows[0], rows[1]+1), j+2])
            ax["resp"][f"sim{i}"].append(new_ax)
            cls.plot_diversity_cumfrac(corrs[i], ax=new_ax, lw=3)
            (i == 2) and new_ax.set_xlabel("temporal similarity index")
            new_ax.set_ylabel("cumu. frac. channel-stim.", fontsize=8, labelpad=-1)
            new_ax.tick_params(axis='both', labelsize=10)
            new_ax.set_xticks([0, 0.3, 0.7, 1])
            new_ax.set_yticks([0, 0.31, 0.69, 1])
            spines_off(new_ax)
            lab_ax.append(new_ax)
        
                    
            row_offset = rows[-1]+1

        row_slice = slice(row_offset, row_offset+2)
        ax["resp"] = {"real":[]}
        # First row is for the real data
        new_ax = plt.subplot(gs[row_slice,:3])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_resps.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)

        new_ax = plt.subplot(gs[row_slice,3])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_bars.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)
        
        new_ax = plt.subplot(gs[row_slice,4])
        ax["resp"]["real"].append(new_ax)
        im_file = art_path + "/yuxin_cumfrac.jpg"
        new_ax.imshow(plt.imread(im_file), extent=[0, 1, 0, 1], transform=new_ax.transAxes, aspect='auto'); new_ax.axis('off')
        lab_ax.append(new_ax)
        
        
        plt.tight_layout(h_pad=-0.25, w_pad = 0.5)    
        align_x = [[0,3,6,9], [1, 4,7,10], [2, 5,8,11]]
        align_y = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
        label_axes.label_axes(lab_ax, ["Ai", "Aii", "Aiii",
                                       "Bi", "Bii", "Biii",
                                       "Ci", "Cii", "Ciii",
                                       "Di", "Dii", "Diii"],
                              fontsize=cls.panel_label_fontsize,
                              fontweight="bold",
                              align_x = align_x,
                              align_y = align_y, dx=-0.001, dy=+0.001)

        output_file = os.path.join(args.output_dir, "conn_dynamics.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        print("DONE PLOTTING CONNECTIVITY DYNAMICS FIGURE.")
                                    
class InferenceDynamics(Figure):
    @staticmethod
    def mystem(x, y, *args, **kwargs):
        return plt.plot([x, x], [0*np.array(y), y], *args, **kwargs)

    @classmethod
    def plot(cls, args, plot_data):
        print("PLOTTING INFERENCE DYNAMICS FIGURE...")
        N = plot_data.N
        n = plot_data.n
        keep = plot_data.keep
        df   = plot_data.df
        out1 = plot_data.out1
        
        plt.figure(figsize=(8, 3))
        gs = GridSpec(1, 6)
        ax_inf = plt.subplot(gs[0,:2])
        x_true = np.arange(N) < n
        x0 = keep[0]["x"]
        x1 = keep[1]["x"]
        # Make a plot where we show the first 10 elements of the true x, the first 10 elements of x0, and the first 10 elements of x1
        # We do these as stem plots, and staggered to avoid overlap
        h0 = cls.mystem(np.arange(10), x_true[:10],   "o-", label = "x_true", color = "gray", markersize=2, lw=2)
        h1 = cls.mystem(np.arange(10) + 0.2, x1[:10], "o-", label = "x1", color = "C1", markersize=2, lw=2)
        h2 = cls.mystem(np.arange(10) + 0.4, x0[:10], "o-", label = "x0", color = "C0", markersize=2, lw=2)
        plt.plot(plt.xlim(),[0,0], "k--", lw=0.5)
        plt.legend([h0[0], h1[0], h2[0]], ["True", "Corr.", "Indep."], loc = "upper right", fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("Feature Index", fontsize=14)
        plt.ylabel("Concentration", fontsize=14)
        spines_off(ax_inf)

        ax_tc = plt.subplot(gs[0,2:4])
        n_true = 5 # First five features are actually there
        colors = [f"C1"] * (n_true) + [(0.5,0.5,0.5,0.25)] * (out1["X"].shape[1] - n_true)
        # [(1.0, 0.65, 0.0, 1.0)] * n_true
        lines = plt.plot(out1["T"], out1["X"])
        [l.set_color(c) for l,c in zip(lines, colors)]
        [l.set_lw(1) for l in lines[n_true:]]
        plt.xlim(out1["T"][0], out1["T"][-1])
        x_exact = keep[1]["x"][:n_true]
        spines_off(ax_tc)
        # Put red triangles < at the right most edge of the plot at the target values
        plt.plot(out1["T"][-1]*np.ones(len(x_exact)), x_exact, "r<", markersize=12)
        plt.xlabel("Time (s)", fontsize=14)
        plt.ylabel("Concentration", fontsize=14)
        plt.ylim(ax_inf.get_ylim()); plt.grid(True, linestyle=":", lw=0.5)
        
        ax_err = plt.subplot(gs[0,-2:])
        mean_df = df.groupby(["sd_n", "sd_inf"], as_index=False).agg({"x0err":["mean","std"], "x1err":["mean","std"]})
        mean_df.columns = ['_'.join(col).strip() for col in mean_df.columns.values]
        best_err0 = mean_df.loc[mean_df.groupby("sd_n_")["x0err_mean"].idxmin()]
        best_err1 = mean_df.loc[mean_df.groupby("sd_n_")["x1err_mean"].idxmin()]
        best_err0.plot(x = "sd_n_", y = "x0err_mean", yerr = "x0err_std", ax=ax_err, linewidth = 1.5, marker = "o", markersize=5, color = "C0",  logx = True, logy=True)
        best_err1.plot(x = "sd_n_", y = "x1err_mean", yerr = "x1err_std", ax=ax_err, linewidth = 1.5, marker = "o", markersize=5, color = "C1",  logx = True, logy=True)
        spines_off(ax_err)
        plt.gca().legend(["Indep.", "Corr."],  fontsize=10, frameon=False, labelspacing=0.2)
        plt.xlabel("$\sigma_n$", fontsize=14)
        plt.ylabel("Error", fontsize=14, labelpad=-10)

        plt.tight_layout()
        
        label_axes.label_axes([ax_inf, ax_tc, ax_err], "ABC", fontsize=12,fontweight="bold")

        output_file = os.path.join(args.output_dir, "inf_dynamics.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        print("DONE PLOTTING INFERENCE DYNAMICS FIGURE.")

class InferringThePrior(Figure):
    panel_label_fontsize = 24
    @classmethod
    def plot(cls, args, plot_data):
        print("PLOTTING PRIOR INFERENCE FIGURE...")

        rwg = LinearSegmentedColormap.from_list('rwg', ['red', 'white', 'green'])
        
        sisters = plot_data.sisters
        vp = plot_data.vp
        r  = plot_data.r
        overlap = plot_data.overlap
        pairs_ods = plot_data.pairs_ods

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
            im =  plt.matshow(-Ci[leaf_order][:, leaf_order], cmap=rwg, vmin = -vmax, vmax = vmax, fignum=False); #colorbar()
            
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
            #ax.set_title(f"Concentrations ~ {name}", fontsize=14)
            ax_lab.append(ax)
        
        label_axes.label_axes(ax_lab, "ABC", fontsize=cls.panel_label_fontsize,fontweight="bold")

        output_file = os.path.join(args.output_dir, "inferred_priors.pdf")
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Figure saved to {output_file}.")
        print("DONE PLOTTING PRIOR INFERENCE FIGURE.")
