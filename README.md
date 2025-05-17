# Inference with correlated priors using sister cells.
This repository contains code and data for manuscripts related "Inference with correlated priors using sister cells."
## Environment setup
1. Create a new Conda environment using python 3.10: `conda create -n myenv python=3.10`
2. Activate the environment: `conda activate myenv`
## Code and data installation
1. Unzip or clone the repository into a parent folder of your choice.
2. Set an environment variable `GIT` to this parent folder. 
   - For example ```export GIT=/Users/bob/git```
   - After this step, the folder structure should be `/Users/bob/git/crick-sisters-release`.
3. Navigate to the repo root at `crick-sisters-release` and install the required packages.
   - Use the requirements.txt file: `pip install -r requirements.txt`
3. Download `run_odours.zip` (~430 MB, unzips to ~1 GB) from [Figshare](https://figshare.com/s/f7b53f9bf48c01006571).
   - Unpack into `crick-sisters-release`.
   - After this step, under the `crick-sisters-release` folder there should be a `run_odours` subfolder.
     - This should in turn contain `bulb0`, `bulb1` and `bulb2` subfolders.
4. Download `data.zip` (~70 MB unzips to the same) from [Figshare](https://figshare.com/s/f7b53f9bf48c01006571).
   - Unpack into `crick-sisters-release`.
   - After this step, under `crick-sisters-release` folder there should be a `data` subfolder.
     - This contains experimental data from Zhang et al. 2025 used in this manuscript.
## Reproducing the figures in the NeurIPS manuscript.
1. Navigate to the `neurips` folder.
2. Run `python driver.py -f [N]` to create figure N from the paper.
   - N can be 1-4.
