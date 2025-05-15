import odours as od
from numpy import arange
import pickle

from argparse import ArgumentParser

def run_odours(bulb, which_od_ind, t_on = 0.5, t_off = 1.5, t_end = 3):
    odours = [od.Odour(arange(bulb.N)==i, bulb.A_mean, t_on = t_on, t_off = t_off) for i in which_od_ind]
    print(odours)
    out = [bulb.run_sister(od_i, t_end = t_end, dt=2e-4, keep_every = 10) for od_i in odours]
    return out

if __name__ == "__main__":
    parser = ArgumentParser(description="Run the odour model")
    parser.add_argument("--data_file", type=str, help="Path to the data file", default="data.p")
    parser.add_argument("which_bulb", type=int, help="Which bulb model to use.")
    parser.add_argument("which_odours", type=int, nargs='+', help="Indices of the odours to run")
    parser.add_argument("--t_on", type=float, default=0.5, help="Time on for the odour")
    parser.add_argument("--t_off", type=float, default=1.5, help="Time off for the odour")
    parser.add_argument("--t_end", type=float, default=3.0, help="End time for the simulation")

    args = parser.parse_args()
    print(args)

    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)

    bulb = data['ob_arr'][args.which_bulb]

    out = run_odours(bulb, args.which_odours, args.t_on, args.t_off, args.t_end)

    for i, (ind_od, result) in enumerate(zip(args.which_odours, out)):
        out_file = f"run_odours/bulb{args.which_bulb}/odour{ind_od}.p"
        with open(out_file, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved results of bulb {args.which_bulb} running odour {ind_od} to {out_file}.")
        
    

