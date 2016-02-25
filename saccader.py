
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def plot_salience(ts, sals, sallabs=('familiar img', 'novel img', 'off img'), 
                  ax=None, sal_ts=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    revsals = sals[::-1, :]
    for i, row in enumerate(revsals):
        print row
        ax.plot(ts, row, label=sallabs[i])
    ax.set_ylabel('salience')
    ax.set_xlabel('time (ms)')
    ax.legend(frameon=False, loc=2)
    ax.set_xticks(sal_ts)
    return ax

def plot_presentation(ts, sals, looks, saccs, lookprob, axbuff=.5,
                      sallabs=('off img', 'left', 'right'), sal_ax=None,
                      sal_ts=None, xtickstep=500, show=True):
    f = plt.figure(figsize=(8, 10))
    salax = f.add_subplot(3, 1, 1)
    probax = f.add_subplot(3, 1, 2, sharex=salax)
    lookax = f.add_subplot(3, 1, 3, sharex=salax)
    xticks = np.arange(ts[0], ts[-1], xtickstep)
    for i, row in enumerate(sals):
        if sal_ax is not None:
            print 'hello'
            sal_ax.plot(ts, row, label=sallabs[i])
        salax.plot(ts, row, label=sallabs[i])
    if sal_ax is not None:
        print 'hey'
        sal_ax.set_ylabel('salience')
        sal_ax.legend(frameon=False, loc=2)
        sal_ax.set_xticks(sal_ts)
    salax.set_ylabel('target salience')
    salax.legend()
    salax.set_xticklabels(xticks, visible=False)

    probax.plot(ts, lookprob)
    probax.set_xticklabels(xticks, visible=False)
    probax.set_ylabel('saccade prob')

    lookax.plot(ts, looks[0, :])
    lookax.plot(saccs, np.ones(len(saccs)), 'o')
    lookax.set_ylabel('fixation location')
    lookax.set_ylim([np.min(looks) - axbuff, np.max(looks) + axbuff])
    lookax.set_xticks(xticks)
    if show:
        plt.show(block=False)

def get_avg_preference(looks, winsize=500., winstep=50., novind=1, famind=2):
    lookarr = np.concatenate(looks, axis=0)
    winends = np.arange(winsize, lookarr.shape[1] + winstep, winstep)
    ts = winends - winsize/2.
    prefs = np.zeros((3, winends.size))
    for i, end in enumerate(winends):
        rellook = lookarr[:, end - winsize:end]
        novlook = np.sum(rellook == novind, axis=1)
        famlook = np.sum(rellook == famind, axis=1)
        bias = (novlook - famlook) / winsize
        prefs[0, i] = np.mean(bias)
        prefs[1, i] = np.std(bias) / np.sqrt(bias.size - 1)
        t, p = sts.ttest_1samp(bias, 0.)
        prefs[2, i] = p
    return ts, prefs

def plot_avg_preference(looks, winsize=500., winstep=50., novind=1, famind=2, 
                        ax=None, show=True):
    ts, prefs = get_avg_preference(looks, winsize, winstep, novind, famind)
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    ax.errorbar(ts, prefs[0, :], yerr=prefs[1, :])
    ax.set_ylabel('magnitude of novelty bias')
    ax.set_xlabel('time (ms)')
    xticks = np.arange(0, 4000+1, 1000)
    ax.set_xticks(xticks)
    if show:
        plt.show(block=False)
    return ts, prefs, xticks

class NaiveSaccader(object):

    def __init__(self, lt_novfunc, lt_novpars, st_novfunc, st_novpars, 
                 sal_timeconst, out_targnov, prob_timeconst, prob_diffpar, 
                 prob_growconst, samebias):
        """         
        numtargs - number of saccade targets excluding off of all targets, 
          this gives numtargs + 1 states
        lt_novfunc - function to use to track longterm novelty, takes number
          of image views as first argument and remainer are parameters
        lt_novpars - parameters for lt_novfunc
        st_novfunc - function to use to track short-term novelty (ie, recency),
          takes time since last view
        st_novpars - parameters for st_novfunc
        """
        self.lt_novfunc = lt_novfunc
        self.lt_novpars = lt_novpars
        self.st_novfunc = st_novfunc
        self.st_novpars = st_novpars
        self.sal_tau = sal_timeconst
        self.out_targnov = out_targnov
        self.prob_tau = prob_timeconst
        self.probgrow = prob_growconst
        self.prob_diffpar = prob_diffpar
        self.samebias = samebias
        
    def present_many(self, targnov, n=50, prestime=4000, tstep=.1):
        sal_samps, look_samps, sacc_samps, lookprob_samps = [], [], [], []
        for i in xrange(n):
            ts, sals, looks, saccs, lookprobs = self.present(targnov, 
                                                             prestime=prestime,
                                                             tstep=tstep)
            sal_samps.append(sals)
            look_samps.append(looks)
            sacc_samps.append(saccs)
            lookprob_samps.append(lookprobs)
        return ts, sal_samps, look_samps, sacc_samps, lookprob_samps

    def present(self, targnov, prestime=4000, tstep=.1):
        """
        targnov - list of number of presentations of each of the targets
        prestime - how long the targets are presented for, default is 4s
        """
        alltargnov = (self.out_targnov,) + tuple(targnov)
        look_inds = np.arange(len(alltargnov))
        ts = np.arange(0, prestime, tstep)
        sals = np.zeros((len(alltargnov), ts.size))
        looks = np.zeros((1, ts.size))  
        lookprob = np.zeros(ts.size)
        sacc_ts = []
        for tind in xrange(1, ts.size):
            for i, nov in enumerate(alltargnov):
                lt = looks[0, tind - 1] == i
                sals[i, tind] = np.max((sals[i, tind - 1] 
                                 + tstep*self._dsdt(sals[i, tind - 1], nov, 
                                                    lt), 0))
            # find look prob based on salience -- model as dfq?
            looking = looks[0, tind - 1]
            not_looking = np.logical_not(look_inds == looking)
            lookprob[tind] = np.max((lookprob[tind - 1] 
                              + tstep*self._dpldt(lookprob[tind - 1],
                                                  sals[looking, tind - 1], 
                                                  sals[not_looking, tind - 1]),
                                     0))
            if lookprob[tind]*tstep > np.random.rand():
                looks[0, tind] = self._look_change(looks[0, tind - 1], 
                                                   sals[:, tind])
                lookprob[tind] = 0.
                sacc_ts.append(ts[tind])
            else:
                looks[0, tind] = looks[0, tind - 1]
        return ts, sals, looks, sacc_ts, lookprob
            
    def _look_change(self, lookind, sals):
        sump = np.sum(sals) + self.samebias
        lps = np.zeros(sals.size)
        for i, ls in enumerate(sals):
            if i == lookind:
                ls = ls + self.samebias
            lps[i] = ls/sump
        return np.random.choice(np.arange(sals.size), p=lps)
        
    def _dsdt(self, s, nov, look):
        flt = self.lt_novfunc(look, nov, self.lt_novpars)
        fst = self.st_novfunc(look, nov, self.st_novpars)
        dsdt = (-s + flt + fst) / self.sal_tau
        return dsdt

    def _dpldt(self, lprob, lsal, nlsals):
        x = nlsals - lsal
        salgrow = self.prob_diffpar*np.max([np.max(x), 0])
        dpldt = (-lprob + self.probgrow + salgrow) / self.prob_tau
        return dpldt

