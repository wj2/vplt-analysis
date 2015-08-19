
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ssts

from utility import *

def get_img_response(data, use_imgnames, pretime=-100, posttime=500, 
                     winsize=50, step=10, spksfield='spike_times', 
                     onfield='centimgon', offfield='centimgoff', 
                     drunfield='datanum', errfield='TrialError', deserr=0, 
                     binsize=5, ttfield='trial_type', imgnumfield='image_nos',
                     tbound=6):
    noerr_data = data[data[errfield] == deserr]
    ne_corrtr_data = noerr_data[noerr_data[ttfield] <= tbound]
    druns = get_data_run_nums(ne_corrtr_data, drunfield)
    m_sunits = {}
    s_sunits = {}
    filtwin = np.ones(winsize/step) / winsize
    xs = np.arange(pretime + (winsize/2.), posttime - (winsize/2.) + step, 
                   step)
    for i, dr in enumerate(druns):
        srun = ne_corrtr_data[ne_corrtr_data[drunfield] == dr]
        runims = {}
        for i, t in enumerate(srun):
            imnames = get_img_names(t[imgnumfield])
            useims = np.array([im in use_imgnames for im in imnames])
            tims = imnames[useims]
            tons = t[onfield][useims]
            for i, im in enumerate(tims[:, 0]):
                on_t = tons[i]
                rb, re = on_t + pretime, on_t + posttime
                n_bs = np.ceil((re - rb) / float(step))[0]
                tneur = np.zeros((len(t[spksfield][0]), len(xs), 1))
                for j, neur in enumerate(t[spksfield][0]):
                    if len(neur) == 0:
                        bigbin_spks = np.ones(len(xs))
                        bigbin_spks[:] = np.nan
                    else:
                        spks, _ = np.histogram(neur, range=(rb, re), bins=n_bs)
                        bigbin_spks = np.convolve(spks, filtwin, mode='valid')
                    tneur[j, :, 0] = bigbin_spks
                prevsee = runims.get(im, [])
                if len(prevsee) == 0:
                    runims[im] = tneur
                else:
                    runims[im] = np.concatenate((prevsee, tneur), axis=2)
        for im in runims.keys():
            m_prevunits = m_sunits.get(im, [])
            s_prevunits = s_sunits.get(im, [])
            currus = runims[im]
            m_cu = np.nanmean(currus, axis=2)
            s_cu = np.nanstd(currus, axis=2)
            if len(m_prevunits) == 0:
                m_sunits[im] = m_cu
                s_sunits[im] = s_cu
            else:
                m_sunits[im] = np.concatenate((m_prevunits, m_cu), axis=0)
                s_sunits[im] = np.concatenate((s_prevunits, s_cu), axis=0)
    return m_sunits, s_sunits, xs

def show_single_unit_resp(m_units, xs, imgs, mean=False):
    f = plt.figure()
    n = np.ceil(np.sqrt(len(imgs)))
    for i, im in enumerate(imgs):
        ax = f.add_subplot(n, n, i+1)
        if mean:
            ax.plot(xs, m_units[im].mean(0))
        else:
            ax.plot(xs, m_units[im].T)
        ax.set_title(im)
    plt.show(block=False)

def show_likelihoods(imlikelies, xs, target=None, legend=False):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    if target is not None:
        ax.plot(xs, imlikelies[target], label='target')
        others = np.zeros((len(imlikelies.keys()), len(imlikelies[target])))
        others[:, :] = np.nan
    for i, cond in enumerate(imlikelies.keys()):
        if target is not None:
            if target != cond:
                others[i, :] = imlikelies[cond]
        else:
            ax.plot(xs, imlikelies[cond], label=cond)
    if target is not None:
        ocourse = np.nanmean(others, axis=0)
        ax.plot(xs, ocourse, label='other')
        ax.errorbar(xs, ocourse, yerr=[ocourse - np.nanmin(others, axis=0), 
                                       np.nanmax(others, axis=0) - ocourse])
    if legend:
        ax.legend()
    plt.show(block=False)

class LikelihoodModel(object):
    
    def __init__(self, neur_means, neur_stds=None, model='poisson', 
                 eps=.000001):
        self.distribs = {}
        self.model = model
        self.eps = eps
        for cond in neur_means.keys():
            self.distribs[cond] = np.zeros(neur_means[cond].shape, 
                                           dtype='object')
            for i in xrange(neur_means[cond].shape[0]):
                for j in xrange(neur_means[cond].shape[1]):
                    if model.lower() == 'poisson':
                        d = ssts.poisson(neur_means[cond][i, j])
                    elif model.lower() == 'gaussian':
                        d = ssts.norm(loc=neur_means[cond][i, j],
                                      scale=neur_stds[cond][i, j] + eps)
                    else:
                        raise RuntimeError('distribution not recognized')
                    self.distribs[cond][i, j] = d

    def get_likelihood(self, popact):
        ll = {}
        for cond in self.distribs.keys():
            likely = np.zeros(self.distribs[cond].shape)
            for i in xrange(likely.shape[1]):
                for j in xrange(likely.shape[0]):
                    if self.model == 'poisson':
                        likely[j, i] = self.distribs[cond][j, i].pmf(popact[j])
                    elif self.model == 'gaussian':
                        likely[j, i] = self.distribs[cond][j, i].pdf(popact[j])
            ll[cond] = np.log(likely + self.eps)
        return ll

    def decode(self, popact):
        ll = self.get_likelihood(popact)
        dprobs = np.zeros(len(ll.keys()))
        sumlike = {}
        for i, cond in enumerate(ll.keys()):
            sumlike[cond] = np.nansum(ll[cond], 0)
            dprobs[i] = np.max(sumlike[cond])
        return ll.keys()[np.argmax(dprobs)], sumlike
