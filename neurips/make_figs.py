import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

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
        if fig_num == 1:
            print("Making Figure 1...")
            cs = ConnectivitySchematic(args)
            cs.plot()
            
            
    
        
    

