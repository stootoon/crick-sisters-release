import numpy as np  
import sisters_util as util

logger = util.create_logger("odour")
INFO   = logger.info

class Odour:
    def __init__(self, x, A, amp = 1, plume={"shape":"step"}, t_on = 0, t_off = np.inf):
        self.x    = x
        self.A = A
        self.val0 = np.dot(A, x) if A is not None else x
        self.amp = amp
        self.plume = plume
        self.t_on = t_on
        self.t_off = t_off

        if plume["shape"] == "step":
            self.amp_fun = lambda t: amp * (t>=t_on)*(t<t_off)
        elif plume["shape"] == "sin":
            freq, phase, bias = [plume[fld] for fld in ["freq", "phase", "bias"]]
            self.amp_fun = lambda t: (t>=t_on)*(t<t_off) * (amp * np.sin(2 * np.pi * freq * t + phase) + bias)
        else:
            raise ValueError("Don't know what to do for plume shape={}.".format(p["shape"]))
    
        INFO(f"Odour Parameters:")
        INFO(f"  Plume from {t_on=} to {t_off=}.")
        INFO(f"  Using {amp=} and {plume}.")                    

    def value_at(self, t, A = None, noise_sd = 0):
        amp  = self.amp_fun(t)

        if not hasattr(t, "len") or len(t) == 1:
            val = amp * self.val0
        elif len(t) > 1:
            val = np.outer(amp, self.val0)
        else:
            raise ValueError("Length of time vector must be greater than 0, was {}".format(len(t)))

        if noise_sd:
            val += np.random.randn(*val.shape)*noise_sd

        return val
