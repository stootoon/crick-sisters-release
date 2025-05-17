import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
# Import a counter
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
INFO = logger.info
DEBUG = logger.debug

opj = os.path.join

from pathlib import Path
project_path = Path(__file__).resolve().parent.as_posix()
sys.path.append(project_path)
print(f"Project path: {project_path}")

class Experiment:
    def __init__(self,
                 data_dir = opj(project_path, "data"),
                 cell_file = "Y489_Zavg_cell_ss10.npy",
                 glom_file = "Y489_Zavg_glom_ss10.npy",
                 glom_assn_file = "glom_assignments.csv",
                 vapour_pressures_file = "Y489_odour_properties_withVapourPressure.csv",
                 odours_table = "Y489_odours_table.csv",
                 ):

        
        
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
        self.data_dir = data_dir

        self.cell_file = opj(data_dir, cell_file)
        assert os.path.exists(self.cell_file), f"Cell file {self.cell_file} does not exist."
        print(f"Loading cell data from {self.cell_file}")
        
        self.glom_file = opj(data_dir, glom_file)
        assert os.path.exists(self.glom_file), f"Glom file {self.glom_file} does not exist."
        print(f"Loading glom data from {self.glom_file}")
        
        self.glom_assn_file = opj(data_dir, glom_assn_file)
        assert os.path.exists(self.glom_assn_file), f"Glom assignment file {self.glom_assn_file} does not exist."
        print(f"Loading glom assignment data from {self.glom_assn_file}")
        
        self.vapour_pressures_file = opj(data_dir, vapour_pressures_file)
        assert os.path.exists(self.vapour_pressures_file), f"Vapour pressures file {self.vapour_pressures_file} does not exist."
        print(f"Loading vapour pressures data from {self.vapour_pressures_file}")
        
        self.odours_table_file = opj(data_dir, odours_table)
        assert os.path.exists(self.odours_table_file), f"Odours table file {self.odours_table_file} does not exist."
        print(f"Loading odours table data from {self.odours_table_file}")
        

        cell_data = np.load(self.cell_file, allow_pickle=True).item()
        self.cell_data = {k: v for k, v in cell_data.items() if "plane11" not in k}

        # Load the glomeruli data, where glom_name != "na"
        self.glom_assn  = pd.read_csv(self.glom_assn_file)
         
        
        self.all_gloms  = list(self.glom_assn["glom_name"].unique())
        self.good_gloms = [g for g in self.all_gloms if "glom" in g]

        self.glom_for_cell = {cell_name:self.glom_assn[self.glom_assn["roi_name"] == cell_name]["glom_name"].values[0] for cell_name in self.cell_data.keys()}
        self.cells_for_glom = {glom:[cell_name for cell_name, glom_name in self.glom_for_cell.items() if glom_name == glom] for glom in self.all_gloms}
        [DEBUG(f"{cell_name:<16} -> {glom_name}") for cell_name, glom_name in self.glom_for_cell.items()];

        self.gloms      = sorted(list({v for _, v in self.glom_for_cell.items() if v != "na"}))
        self.glom_sizes = {g: len(c) for g, c in self.cells_for_glom.items() if g in self.gloms}
        INFO(f"Found {len(self.gloms)} valid glomeruli.")
        for g, s in sorted(self.glom_sizes.items()):
            DEBUG(f"  {g} has {s:>2} cells: ", ",".join([c[10:] for c in self.cells_for_glom[g]]))
        # Use a Counter to count how many glomeruli of each size there are
        glom_sizes_counter = Counter(self.glom_sizes.values())
        # Print the sizes and counts
        INFO(f"Glomeruli sizes: {len(glom_sizes_counter)} unique sizes.")
        for size, count in sorted(glom_sizes_counter.items()):
            INFO(f"  {size:>2} cells: {count:>2} glomeruli.")
        
            
        # Get the odour names and shorten them
        self.odours = sorted(list(self.cell_data[list(self.cell_data.keys())[0]].keys()))
        self.short_odours = [od[4:6] +"." + od[12:] for od in self.odours]

        # Load the odours table
        self.odours_table = pd.read_csv(self.odours_table_file, delimiter=";")
        # Check that each of the items in the "mopairs" column is in the odours list
        mopairs = self.odours_table["mopairs"].values
        assert sorted(list(mopairs)) == sorted(self.odours), f"Odours in table do not match odours in data."
        # Get the name of the odour from the "odour" column
        self.odour_names = {od: self.odours_table[self.odours_table["mopairs"] == od]["odourant"].values[0] for od in self.odours}
        
        
        INFO(f"{len(self.odours)} odours.")
        [DEBUG(f"{od} ({sod})") for od, sod in zip(self.odours, self.short_odours)]

        self.glom_data = {glom:
                          {od: np.array([self.cell_data[cell][od] for cell in self.cells_for_glom[glom]]) for od in self.odours} 
                          for glom in self.gloms if self.glom_sizes[glom] > 0}

        self.vapour_pressures_df = pd.read_csv(self.vapour_pressures_file)
        # Form a dictionary with keys set by the "moPairs" column and values set by the "vapourPressure_mmHg" column
        self.vapour_pressures = self.vapour_pressures_df.set_index("moPairs")["vapourPressure_mmHg"].to_dict()        

        self.t = np.arange(400)/10-3
        INFO(f"{len(self.t)} time points from {self.t[0]} to {self.t[-1]} seconds.")

    def compute_responses(self, window_start=0, window_end=10):
        
        self.window_start = window_start
        self.window_end   = window_end
        self.ind_resp     = np.where((self.t >= window_start) & (self.t <= window_end))[0]

        assert len(self.ind_resp) > 0, f"Window ({window_start}, {window_end}) is empty."
        
        self.X = []
        self.S_vals = []
        self.used_gloms = []
        for glom in self.gloms:
            if glom not in self.glom_data:
                continue
            self.used_gloms.append(glom)
            gdata = self.glom_data[glom]
            Xod = [odata[:, self.ind_resp].mean(axis=1) for odour, odata in gdata.items()]
            self.X.append(np.array(Xod))
            self.S_vals.append(self.X[-1].shape[1])

        self.Xs     = [Xi/np.std(Xi, axis=0) for Xi in self.X]
        self.XXs    = np.hstack(self.Xs)
        self.XXs_ms = np.hstack([Xi - Xi.mean(axis=1)[:,np.newaxis] for Xi in self.Xs])

    def plot_responses(self, fig = None, figsize=(23,10)):
        # Check if self.X exists
        if not hasattr(self, 'X'):
            raise ValueError("Responses have not been computed. Please call compute_responses() first.")

        fig is None and plt.figure(figsize=figsize)
        plt.imshow(self.XXs, vmin=-7,vmax=7, cmap="bwr"); #axis("tight")
        xt = np.cumsum(self.S_vals)
        xt_lab = [xti - si/2 -1/2 for xti, si in zip(xt, self.S_vals)]
        plt.gca().set_xticks(xt_lab)
        plt.gca().set_xticklabels([f"{g}: {self.glom_sizes[g]}" for g in self.used_gloms], minor=False, fontsize=6, rotation=90)
        #gca().set_xticks(xt_lab, [f"{g}({glom_sizes[g]})" for g in glom_names], fontsize=6, rotation=90)
        [plt.axvline(xti-1/2, color="k", lw=1, linestyle=":") for xti in xt];
        plt.xlim(-1/2, self.XXs.shape[1]-1/2)
        plt.gca().set_yticks(range(self.XXs.shape[0]))
        plt.gca().set_yticklabels(self.odours, minor=False, fontsize=5);
        plt.title(f"Responses of {self.XXs.shape[1]} cells to {len(self.odours)} odours in {len(self.used_gloms)} glomeruli")

    def compute_covariance(self, scales = 1):
        # Check if self.X exists
        if not hasattr(self, 'X'):
            raise ValueError("Responses have not been computed. Please call compute_responses() first.")
        
        # If scales is a scalar, expand it to odour size
        if np.isscalar(scales):
            scales  = np.ones(len(self.odours)) * scales
            
        assert len(scales) == len(self.odours), \
            f"Scales must be a scalar or a vector of length {len(self.odours)}."
                                
        self.scales = scales

        Scales = np.diag(scales)

        self.Ys     = [Scales @ Xi for Xi in self.Xs]

        assert all([Si == Yi.shape[1] for Si, Yi in zip(self.S_vals, self.Ys)]), \
            f"Mismatch between expected and actual glomerular sizes."
        
        self.C_glom = [np.cov(Yi, bias=True) for Yi in self.Ys]
        self.C      = np.sum([Si * Ci for Si, Ci in zip(self.S_vals, self.C_glom)], axis=0)

        return self.C, self.C_glom
        



