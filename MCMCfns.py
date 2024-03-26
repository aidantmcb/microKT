import numpy as np

def loglikely_2(v, av, sl, **kwargs):

    # v = p[:int(len(p)/2)]
    # av = p[int(len(p)/2):]
    # av = np.tile(av, len(sl.stars)).reshape(len(sl.stars), -1)

    signal = sl.signals
    sigma = sl.signal_errs

    # print('loglikely av shape' ,av.shape)
    val = - 0.5 * np.nansum((signal - sl.model_signals(v, dAVdd = av))**2 / (sigma**2)) # IS THIS WRONG
    if np.isnan(val):
        # print('fail loglikely')
        return -np.inf
    else:
        return val
    # return - 0.5 * np.sum((signal - sl.model_signals(v, dAVdd = av))**2 / (sigma**2)) 

# def logprior_v(v, v_max = 5, prior_mult = 1, **kwargs):
#     if (np.any(np.abs(v) > prior_mult * v_max)):
#         # print('logprior v -inf')
#         return -np.inf
#     return 0.0


def logprior_v(v, v_max = 5, prior_mult = 1, **kwargs):
    if (np.any(v < -8.5)) or (np.any(v > 17.5)):
        # print('logprior v -inf')
        return -np.inf
    return 0.0

def logprior_davdd(av, AV_base = 5, AV_max = 10):   
    # if (np.any(np.abs(av - AV_base) > AV_max)):
    #     # print('av -inf')
    #     return -np.inf
    if ((np.any(av < 0))):
        # print('logprior av -inf')
        return -np.inf
    return 0.0

def logprior_davdd_reg(av,sl, mask = None, **kwargs):
    # print(av.shape)
    # av = np.tile(av, len(sl.stars)).reshape(len(sl.stars), -1) # FOR NOW 
    av = np.copy(av)


    mask = sl.dAVdd_mask
    # mask = av == 0
    av[mask] = np.nan

    # avmed = np.nanmedian(av, axis = 0)
    # avstd = np.nanstd(av, ddof = 1,  axis = 0)
    # avstd[np.isnan(avstd)] = 0.2

    avmed = sl.voxel_dAVdd
    # print(avmed.shape)
    avstd = sl.voxel_dAVdd_std * 10 # should be 10
    # print(avstd.shape)

    # print(av.shape)
    # return 0.0
    # lp_val = np.nansum(np.log(1/(np.sqrt(2 * np.pi) * avstd))) - 0.5 * np.nansum((av - avmed[:, np.newaxis])**2 / (2 * avstd[:, np.newaxis]**2))# first part might not be needed
    # lp_val = np.nansum(- 0.5 * np.nansum((av - avmed[np.newaxis, :])**2 / (2 * avstd[np.newaxis, :]**2)))# first part might not be needed
    lp_val = -np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * avstd**2)))# first part might not be needed

    
    return lp_val
    # return np.sum(np.log(1/(avstd[:, np.newaxis] * np.sqrt(2 * np.pi ))) - 0.5 * (av - avmed[:, np.newaxis])**2 / (2 * avstd[:, np.newaxis]**2)) # first part might not be needed

def logprior_davdd_reg_group(av,sl, mask = None,  width_factor = 3, **kwargs):
    av = np.copy(av)
    mask = sl.dAVdd_mask
    av[mask] = np.nan
    avmed = np.nanmedian(av, axis = 0,)
    avstd = sl.voxel_dAVdd_std


    lp_val = - np.nansum(np.log(np.sqrt(2 * np.pi))) + np.nansum(- 0.5 * np.nansum((av - avmed)**2 / (2 * (width_factor * avstd)**2)))# first part might not be needed
    return lp_val


def logprob_2(p, sl, logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[ :ndim]
    av = p[ndim:].reshape(-1, ndim)

    # print(av.shape)

    lp = logprior(v, **kwargs)
    lp_davdd = logprior_davdd(av, AV_base = sl.dAVdd)
    lp_davdd_reg = logprior_davdd_reg(av, sl, **kwargs)
    lp_davdd_reg_group = logprior_davdd_reg_group(av, sl)

    if (not np.isfinite(lp)) | (not np.isfinite(lp_davdd)) | (not np.isfinite(lp_davdd_reg)):
        # print('fail logprob')
        return -np.inf
    return lp + lp_davdd + lp_davdd_reg +  loglikely_2(v, av, sl = sl, **kwargs) + lp_davdd_reg_group # group term added 10.13

def logprob_avfix(p,sl, av = None,  logprior = logprior_v, loglikely = loglikely_2, **kwargs):
    ndim = len(sl.voxel_dAVdd)
    v = p[:ndim]

    # av = av.reshape(-1, ndim)

    lp = logprior(v, **kwargs)
    if (not np.isfinite(lp)):
        return -np.inf
    return lp + loglikely_2(v, av, sl = sl, **kwargs)

def logprior_foreground(l, b, distance, sl, foreground_distance = 400, **kwargs):    
    def polynomial2d(x1, x2, theta = None):  
        if theta is None:
            theta = np.array([5.76803551, -0.93688804, -0.83121174, -0.18054651, -0.02163556, -0.2999652])
            uncert = array(
                [[ 3.73734579e+00, -2.15312709e+00,  3.77688040e-01, -1.09780155e-01,  2.73540008e-01, -9.40627449e-02],
                [-2.15312709e+00,  2.07232236e+00, -1.47408631e-01, 1.52590867e-02, -3.18111493e-01, -6.05259982e-02],
                [ 3.77688040e-01, -1.47408631e-01,  1.14152006e+00, -2.94225460e-01, -8.72619885e-04,  4.27465860e-02],
                [-1.09780155e-01,  1.52590867e-02, -2.94225460e-01, 1.03988171e-01,  8.01196096e-03,  2.89406984e-03],  
                [ 2.73540008e-01, -3.18111493e-01, -8.72619885e-04, 8.01196096e-03,  5.37180547e-02,  1.06799789e-02],
                [-9.40627449e-02, -6.05259982e-02,  4.27465860e-02, 2.89406984e-03,  1.06799789e-02,  7.81419107e-02]])
        x1 = x1 - 160 # FOR CA CLOUD SPECIFICIALLY
        x2 = x2 + 8 # DITTO
        X = np.array([[np.ones(len(x1)), x1, x2, x1 * x2, x1**2, x2**2]]).T
        matrix = X * theta[:, np.newaxis]
        return np.nansum(matrix, axis = 1)
    
    foreground = distance <= foreground_distance
    # maybe I should calculate the expected velocity BEFORE instead of INSIDE the run?

class Logprior_Foreground:
    def __init__(self, l, b):
        self.pointfit = self.polynomial2d(l, b)
        self.pointfit_width = 2.404363059339516

    def polynomial2d(self, x1, x2, theta = None, uncert = None):  
        if theta is None:
            theta = np.array([5.03964666, -1.04129592, -0.72842925, -0.20292219,  0.0206567,  -0.14442016])
        if uncert is None:
            uncert = 2.404363059339516
        x1 = x1 - 160 # FOR CA CLOUD SPECIFICIALLY
        x2 = x2 + 8 # DITTO
        X = np.array([[np.ones(np.array(x1).shape), x1, x2, x1 * x2, x1**2, x2**2]]).T
        matrix = X * theta[:, np.newaxis]
        return np.nansum(matrix, axis = 1)
    
    def logprior_foreground_v(v, distance, foreground_distance = 400, **kwargs):    
        foreground = distance <= foreground_distance
        prior_val = np.zeros(distance.shape)
        prior_val[foreground] = np.nansum(- 0.5 * np.nansum((v - self.pointfit)**2 / (self.pointfit_width**2)))[foreground]
        return prior_val
        
    def logprior_foreground_av(av, distance, foreground_distance = 400):
        foreground = distance <= foreground_distance
        prior_val = np.zeros(distance.shape)
        ampfit = (0.01928233, 0.01431857)
        avf = lambda x, mu, sigma :  -(x - mu)**2 / (2 * sigma**2)
        prior_val[foreground] = - 0.5 * np.nansum((av - ampfit[0])**2 / (ampfit[1]**2))[foreground]
        return prior_val


class BayesFramework:
    def __init__(self, **kwargs):
        return

    def add_logprior(self):
        return
    
    def add_logprob(self):
        return
    
