import numpy as np

def loglikelyhood(v, av, sl):
    av = av.reshape(-1, sl.ndim) #check this
    signals = sl.signals
    signal_errs = sl.signal_errs
    model_signals = sl.model_signals(v, dAVdd = av)
    val = - 0.5 * np.nansum((signals - model_signals)**2 / (signal_errs**2))
    return val 

def logprior_velocity(v):
    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        return -np.inf
    return 0.0

def logprior_davdd(av, av_spread):
    if np.any(av < 0) | np.any(av_spread < 0):
        return -np.inf
    return 0.0

def logprior_avscatter(av_offset):
    a_std = 8 * 0.010189143589820518
    val = -0.5 * np.nansum((av_offset)**2 / (a_std**2))
    return val

def logprobability(p, sl):
    ndim = sl.ndim
    nstar = sl.nsig
    nparams = 2 * sl.ndim + ndim * nstar
    v = p[:ndim]
    av = p[ndim:2*ndim]
    av_offset = p[2*ndim:]
    av_spread = np.repeat(av, nstar) + av_offset
    llikely = loglikelyhood(v, av_spread, sl)
    lprior_v = logprior_velocity(v)
    lprior_av = logprior_davdd(av, av_spread)
    lprior_avscatter = logprior_avscatter(av_offset)
    lprior = lprior_v + lprior_av + lprior_avscatter
    lprob = llikely + lprior
    if np.isnan(lprob):
        return -np.inf
    return lprob