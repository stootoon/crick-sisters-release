import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm as colormaps
from matplotlib import colors as mcolors
from scipy.signal import find_peaks
from tqdm import tqdm
import time
import logging

def create_logger(name, level = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s %(module)24s %(levelname)8s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger

def gen_cov2(N, n, ρ = 0.9):
    C = np.zeros((N,N))
    C[:n, :n] = ρ
    for i in range(n):
        C[i, i] = 1
    print(np.linalg.eigvalsh(C[:n, :n]))
    return C

class TimedBlock:
    def __init__(self, INFO, name = "BLOCK"):
        self.name = name
        self.start_time = -1
        self.end_time = -1
        self.elapsed = -1
        self.INFO = INFO

    def __enter__(self):
        self.start_time = time.time()
        self.INFO("Started  {}.".format(self.name))

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.INFO("Finished {} in {:.3f} seconds.".format(self.name, self.elapsed))

def rms_error(x,y, verbose = True):
    err = np.sqrt(np.mean((x-y)**2))
    if verbose:
        print("RMS error: {}".format(err))

    return err
    
def plot_odor_response(results, which_x, which_la, x_true = [], x_or_v = "x", plot_every = 1e-3, normalize=False, draw_mode = "tall"):
    t = results["T"]
    
    tplot = np.arange(t[0],t[-1], plot_every)
    iplot = np.array([np.argmin(np.abs(t-tp)) for tp in tplot])
    # PLOT EVERY
    
    X = results["X"][iplot, :]
    V = results["V"][iplot, :]
    La= [La_glom[iplot,:] for La_glom in results["La"]]
    t = tplot
    nt = len(t)
    
    Svals = [Lai.shape[0] for Lai in La]
    M = len(Svals)
    
    if type(which_x) is int:
        # Pick the GCs with the top variance
        vx = np.var(X,axis=0)
        which_x = np.argsort(vx)[::-1][:which_x]

    def proc_which_(which_la):
        if type(which_la) is int:
            vla = np.mean(np.var(La,axis=0),axis=1)
            return [(i, range(S)) for i in np.argsort(vla)[::-1][:which_la]]
        else:
            return which_la

    which_la = proc_which_(which_la)

    nx  = len(which_x)
    nla = len(which_la)

    ntop = max([nx, nla])

    if draw_mode == "tall":
        nrows = ntop + 2 + 2
        ncols = 3
    else:
        nrows = ntop
        ncols = 6
        

    gs = GridSpec(nrows, ncols)

    colors = {"x":"black",              
              "la":"red",
              }
              
    # PLOT THE GC ACTIVITY

    Var = X if x_or_v == "x" else V
    yr = np.array([np.min(Var), np.max(Var)])
    yrm = np.mean(yr)
    yre = (yr - yrm)*1.1 + yrm
    
    for i,ix in enumerate(which_x):
        plt.subplot(gs[i,0])
        plt.plot(t, Var[:,ix],color=colormaps.rainbow(ix/float(Var.shape[0]))),
        plt.xticks([])
        plt.ylabel("#{}".format(ix))        
        #plt.ylim(yre)
        if i == 0:
            plt.title(x_or_v.upper())

    if draw_mode == "tall":
        plt.subplot(gs[ntop:(ntop+4), 0])
    else:
        plt.subplot(gs[:ntop, 3])
        
    v = np.var(Var, axis=0)
    iv = np.argsort(v)
    nv = 10

    for i in range(Var.shape[1]-nv, Var.shape[1]):
        x = Var[:,iv[i]]
        a = np.var(x)/(max(v) + 1e-6)
        plt.plot(t,Var[:,iv[i]]*100 + iv[i], alpha = a,
                 color = colormaps.rainbow(iv[i]/float(Var.shape[1])),
        )
        plt.xticks(np.arange(t[0],t[-1],0.1))
        plt.ylim(0,Var.shape[1]+100*np.max(Var))
            

    def plot_sister_activity_(La, which_la, column, cm_line, ttl):

        La_flat = np.concatenate(La).flatten()
        la_max = np.max(np.abs(La_flat))
        
        Lan = [La_glom/(la_max + 1e-6) for La_glom in La] if normalize else La

        for i, (iglom, ind_sis) in enumerate(which_la):
            plt.subplot(gs[i, column])
            for which_sis in ind_sis:
                plt.plot(t, Lan[iglom][:,which_sis], color=cm_line(float(which_sis)/Svals[iglom]))
                normalize and plt.ylim([-1,1])
            plt.ylabel("#{}".format(iglom))
            #plt.xticks([])
            plt.xticks(np.arange(t[0],t[-1],0.2))
            (i == 0) and plt.title(ttl)

        if draw_mode == "tall":
            plt.subplot(gs[ntop:(ntop+4),column])
        else:
            plt.subplot(gs[:ntop, 3 + column])

        #La1 = np.reshape(La,(La.shape[0], La.shape[1]*La.shape[2])).T
        La1 = concatenate([La_ts_i.T for La_ts_i in La])

        plt.matshow(La1, fignum=False, aspect="auto", cmap=colormaps.seismic,
                    vmin=-la_max, vmax=la_max,
                    extent=[t[0],t[-1],0,La[0].shape[0]])

        plt.xticks(np.arange(t[0],t[-1],0.1))

        plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True,labeltop=False)

    plot_sister_activity_(La, which_la, 1, colormaps.rainbow, "La/max(|La|)" if normalize else "La")
    plt.tight_layout()
    

def multi_plot(y, x = None, wide = False):
    n = len(y)
    if x is None:
        x = np.arange(len(y[0]))
        
    if wide:
        plt.figure(figsize=(4*n,3))
        ax = [plt.subplot(1,n,i+1) for i in range(n)]
    else:
        plt.figure(figsize=(8,3*n))
        ax = [plt.subplot(n,1,i+1) for i in range(n)]

    for ax, yi in zip(ax,y):
        plt.sca(ax)        
        plt.plot(x, yi)
        plt.grid(True)
    plt.tight_layout()
    
def compare_abs_rel(true, pred, inds, wide = False):
    abs_errs = []
    norms    = []
    
    for i in range(len(inds)):
        norms.append(np.norm(true[i]))
        abs_errs.append(np.norm(true[i]-pred[i]))
    
    rel_errs  = abs_errs/np.array(norms)
    worst_abs = np.argmax(abs_errs)
    worst_rel = np.argmax(rel_errs)

    if wide:
        plt.figure(figsize=(16,3))
        ax = [plt.subplot(1,2,i) for i in range(1,3)]
    else:
        plt.figure(figsize=(8,6))
        ax = [plt.subplot(2,1,i) for i in range(1,3)]
        
    plt.sca(ax[0])
    plt.plot(t, true[worst_abs])
    plt.plot(t, pred[worst_abs])
    plt.title("Worst absolute at {} ({:1.3e})".format(inds[worst_abs], max(abs_errs)))
    plt.xlabel("t / sec")
    plt.ylabel("$\lambda_i(t)$")
    plt.grid(True)
    plt.sca(ax[1])
    plt.plot(t, true[worst_rel])
    plt.plot(t, pred[worst_rel])
    plt.title("Worst relative at {} ({:1.3e})".format(inds[worst_rel], max(rel_errs)))
    plt.xlabel("t / sec")
    plt.ylabel("$\lambda_i(t)$")
    plt.grid(True)
    plt.tight_layout()
    return abs_errs, rel_errs


def euler(A, x0, dt, nt):
    x = np.zeros((nt, len(x0)))
    x[0] = x0
    for i in tqdm(range(1,nt)):
        x[i] = x[i-1] + dt * np.dot(A,x[i-1])
    return x.T

def compare_gc(x, w=0.5, labs=None, cols = None, lw = 1):
    # x is a list containg the responses of some obs
    # We want to compare the responses to each odour
    # so plot them as vertical lines centered on the odour index
    # colured by ob
    dw = 0 if len(x) == 1 else w/(len(x)-1)
    hleg = []
    pos0 = np.arange(len(x[0]))
    plt.plot(pos0, np.zeros_like(pos0), color="lightgray", marker=".")
    cols = [f"C{i}" for i in range(len(x))] if cols is None else cols        
    for i, xi in enumerate(x):
        pos = pos0 - w/2 + dw*i
        h = plt.plot([pos,pos], [0*xi,xi], color=cols[i], lw=lw)
        hleg.append(h[0])
    plt.legend(hleg, [f"res{i}" for i in range(len(x))] if labs is None else labs)
