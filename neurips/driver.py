import os
from argparse import ArgumentParser
import compute, figures
        
conn_schematic = compute.ConnectivitySchematic()
conn_dynamics  = compute.ConnectivityDynamics()
inf_dynamics   = compute.InferenceDynamics()
inf_priors     = compute.InferringThePrior()

figure_order = [
    figures.ConnectivitySchematic(conn_dynamics),
    figures.InferenceDynamics(inf_dynamics),
    figures.ConnectivityDynamics(conn_dynamics),
    figures.InferringThePrior(inf_priors),
]        

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
        figure_order[fig_num - 1].compute_and_plot(args)

    print("ALLDONE")
