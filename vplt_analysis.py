
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

from eyes import *
from utility import *

def binarize_spiketimes(spts, binsize, bounds):
    nbins = np.ceil((bounds[1] - bounds[0]) / binsize)
    bin_range = (bounds[0], bounds[0] + binsize*nbins)
    bspks, _ = np.histogram(spts, nbins, bin_range)
    return bspks    

def neur_arr_omega_squared(spktrains, levelinds, width=50, step=5):
    out = np.zeros((spktrains.shape[1], spktrains.shape[2]))
    for i in xrange(spktrains.shape[1]):
        out[i, :] = omega_squared_tc(spktrains[:, i, :], levelinds)
    return out

def omega_squared_tc(spkcount_tc, levelinds):
    out = np.zeros(spkcount_tc.shape[1])
    for i in xrange(spkcount_tc.shape[1]):
        out[i] = omega_squared(spkcount_tc[:, i], levelinds)
    return out

def omega_squared(spkcounts, levelinds):
    # get SS_betweengroups
    total_mean = np.mean(spkcounts)
    total_sse = np.sum((spkcounts - total_mean)**2)
    group_means = np.zeros(len(levelinds))
    group_sses = np.zeros(len(levelinds))
    group_df = len(levelinds) - 1
    sse_b = 0
    for i, li in enumerate(levelinds):
        gspks = spkcounts[li]
        group_means[i] = np.mean(gspks)
        sse_b = sse_b + len(gspks)*((group_means[i] - total_mean)**2)
        group_sses[i] = np.sum((gspks - group_means[i])**2)
    mse_w = np.sum(group_sses)/(len(spkcounts) - len(levelinds))
    omeg_2 = (sse_b - group_df*mse_w)/(total_sse + mse_w)
    return omeg_2  

def get_spikecourse(data, trial_type=None, pretime=-100, posttime=5000, 
                    binsize=2, timefield='fixation_off', ttfield='trial_type', 
                    mantimes=None, spksfield='spike_times'):
    if trial_type is not None:
        trls = data[data[ttfield] == trial_type]
        if mantimes is not None:
            mantimes = mantimes[data[ttfield] == trial_type]
    else:
        trls = data
    nbins = np.ceil((posttime - pretime) / binsize)
    numtrains = len(trls[spksfield])
    if len(trls[spksfield]) > 0:
        spkscoll = np.concatenate(trls[spksfield], axis=0)
    else:
        spkscoll = np.array([[]])
    numtrains = np.ones(spkscoll.shape[1])*len(trls[spksfield])
    binspks = np.zeros((spkscoll.shape[1], nbins))
    if mantimes is None:
        centspks = spkscoll - np.reshape(trls[timefield], (-1, 1))
    else:
        centspks = spkscoll - mantimes
    trlbounds = (pretime, posttime)
    for i in xrange(centspks.shape[1]):
        collectall = np.concatenate(centspks[:, i], axis=0)
        binspks[i, :] = (binarize_spiketimes(collectall, binsize, trlbounds) 
                         / float(centspks.shape[0]))
    return binspks, numtrains

def get_varexpcourse(data, pretime=-200, posttime=5000, width=50, step=5.,
                     timefield='fixation_off', mantimes=None, 
                     spksfield='spike_times', drunfield='datanum',
                     fieldvalpairs=[]):
    xs = np.arange(pretime + width/2., posttime - width/2. + step, step)
    train_filt = np.ones(width/step)
    druns = get_data_run_nums(data, drunfield)
    for i, dr in enumerate(druns):
        srun = data[data[drunfield] == dr]
        if mantimes is not None:
            currtimes = mantimes[data[drunfield] == dr]
        useinds = []
        alluse = np.zeros(len(srun))
        for fpv in fieldvalpairs:
            use = fpv[1](srun[fpv[0]])
            alluse = alluse + use
            useinds.append(use)
        alluse = alluse > 0
        for o, use in enumerate(useinds):
            useinds[o] = use[alluse]
        if mantimes is not None:
            currtimes = currtimes[alluse]
        l_trains = np.zeros((np.sum(alluse), srun[spksfield][0].shape[1],
                             len(xs)))     
        use_srun = srun[alluse]
        for j, trl in enumerate(use_srun):
            for k in xrange(trl[spksfield].shape[1]):
                if mantimes is not None:
                    offt = currtimes[j]
                else:
                    offt = trl[timefield][0, 0]
                bounds = (offt + pretime, offt + posttime)
                spks = trl[spksfield][0, k]
                strain = binarize_spiketimes(spks, step, bounds)
                l_trains[j, k, :] = np.convolve(strain, train_filt, 'valid')
        expvars = neur_arr_omega_squared(l_trains, useinds)
        if i == 0:
            cond_spks = {}
            all_expvars = expvars
            for o, use in enumerate(useinds):
                cond_spks[o] = l_trains[use, :, :].mean(0)
            all_spktr = l_trains.mean(0)
        else:
            all_expvars = np.concatenate((all_expvars, expvars), axis=0)
            for o, use in enumerate(useinds):
                u_spks = l_trains[use, :, :].mean(0)
                cond_spks[o] = np.concatenate((cond_spks[o], u_spks), axis=0)
            all_spktr = np.concatenate((all_spktr, l_trains.mean(0)), axis=0)
    return xs, all_expvars, all_spktr, cond_spks

def get_nlook_timedir(data, n, eyefield='eyepos', postfix='fixation_off', 
                      prefix=None, stimoff=3, tlim=200):
    lts = np.ones(len(data))*-1
    lds = np.zeros(len(data), dtype='S1')
    eyes = data[eyefield]
    for i, t in enumerate(eyes):
        aft = data[i][postfix]
        ls = get_look_thresh(t, stimoff, thresh=tlim, offset=aft[0, 0])
        if len(ls[0]) > n:
            lds[i] = ls[0][n]
            lts[i] = ls[1][n]
    return lds, lts

def get_nsacc_activity(data, n, tts, labels, pretime=-100,
                       binsize=2, timefield='fixation_off', imgon=0,
                       imgoff=5000, drunfield='datanum', tlim=300, 
                       eyefield='eyepos', lloc=-3, rloc=3, stimoff=3,
                       ttfield='trial_type'):
    druns = get_data_run_nums(data, drunfield)
    sunits = {}
    avgunits = {}
    for i, t in enumerate(tts):
        ttrials = data[data[ttfield] == t]
        l_wgts = []
        r_wgts = []
        for j, dr in enumerate(druns):
            srun = ttrials[ttrials[drunfield] == dr]
            eps = srun[eyefield]
            afters = np.concatenate(srun[timefield])
            t_lens, t_looks, t_s_bs, t_s_es = get_fixtimes(srun, [t], 
                                                           postthr=timefield)
            
            luse = np.array(map(lambda x: len(x) > n and x[n] == 'l', 
                                t_looks[t]))
            lruns = srun[luse]
            ruse = np.array(map(lambda x: len(x) > n and x[n] == 'r', 
                                t_looks[t]))
            rruns = srun[ruse]
            lstarts = np.reshape(map(lambda x: x[n], t_s_bs[t][luse]), (-1, 1))
            rstarts = np.reshape(map(lambda x: x[n], t_s_bs[t][ruse]), (-1, 1))
            lspks, lwg = get_spikecourse(lruns, pretime=pretime, posttime=tlim, 
                                         binsize=binsize, mantimes=lstarts)
            rspks, rwg = get_spikecourse(rruns, pretime=pretime, posttime=tlim, 
                                         binsize=binsize, mantimes=rstarts)
            l_wgts = np.concatenate((l_wgts, lwg))
            r_wgts = np.concatenate((r_wgts, rwg))
            if j == 0:
                alllspks = lspks
                allrspks = rspks
            else:
                alllspks = np.concatenate((alllspks, lspks), axis=0)
                allrspks = np.concatenate((allrspks, rspks), axis=0)
        l = labels[i]
        sunits[l] = {}
        sunits[l]['l'] = alllspks*(1000. / binsize)
        sunits[l]['r'] = allrspks*(1000. / binsize)
        avgunits[l] = {}
        avgunits[l]['l'] = np.average(sunits[l]['l'], axis=0, 
                                      weights=l_wgts > 0)
        avgunits[l]['r'] = np.average(sunits[l]['r'], axis=0, 
                                      weights=r_wgts > 0)
    return avgunits, sunits

def show_nsacc_activity(data, tts, n, labels, pretime=-100, binsize=2, 
                        timefield='fixation_off', imgon=0, imgoff=5000, 
                        drunfield='datanum', tlim=300, eyefield='eyepos', 
                        lloc=-3, rloc=3, stimoff=3):
    avgus, sus = get_nsacc_activity(data, tts, n, labels, pretime=pretime, 
                                    binsize=binsize, timefield=timefield, 
                                    imgon=imgon, imgoff=imgoff, 
                                    drunfield=drunfield, tlim=tlim, 
                                    eyefield=eyefield, lloc=lloc, rloc=rloc, 
                                    stimoff=stimoff)
    xs = np.arange(pretime, tlim, binsize)
    fig_avg = plt.figure()
    fig_su = plt.figure()
    avgdim = np.sqrt(len(avgus.keys()))
    su_dep = len(avgus.keys())
    su_wid = 2
    for i, k in enumerate(avgus.keys()):
        avgax = fig_avg.add_subplot(avgdim, avgdim, i+1)
        avgax.plot(xs, avgus[k]['l'], label='l')
        avgax.plot(xs, avgus[k]['r'], label='r')
        avgax.set_title(k)
        avgax.legend()

        suax_l = fig_su.add_subplot(su_dep, su_wid, 2*i+1)
        suax_l = show_single_units(sus[k]['l'], pretime=pretime, posttime=tlim, 
                                   binsize=binsize, ax=suax_l)
        suax_l.set_title(k)
        suax_l.set_ylabel('units (l)')

        suax_r = fig_su.add_subplot(su_dep, su_wid, 2*i+2)
        suax_r = show_single_units(sus[k]['r'], pretime=pretime, posttime=tlim,
                                   binsize=binsize, ax=suax_r)
        suax_r.set_ylabel('units (r)')
    plt.show(block=False)
    return avgus, sus

def get_nlook_activity(data, n, tts, labels, pretime=-100,
                       binsize=2, timefield='fixation_off', imgon=0,
                       imgoff=5000, drunfield='datanum', tlim=300, 
                       eyefield='eyepos', lloc=-3, rloc=3, stimoff=3,
                       ttfield='trial_type'):
    druns = get_data_run_nums(data, drunfield)
    sunits = {}
    avgunits = {}
    for i, t in enumerate(tts):
        ttrials = data[data[ttfield] == t]
        l_wgts = []
        r_wgts = []
        for j, dr in enumerate(druns):
            srun = ttrials[ttrials[drunfield] == dr]
            eps = srun[eyefield]
            afters = np.concatenate(srun[timefield])
            looks = np.array([get_look_thresh(ep, stimoff, thresh=tlim, 
                                              offset=afters[k, 0]) 
                              for k, ep in enumerate(eps)])
            luse = np.array(map(lambda x: len(x[0]) > n and x[0][n] == 'l', 
                                looks))
            lruns = srun[luse]
            ruse = np.array(map(lambda x: len(x[0]) > n and x[0][n] == 'r', 
                                looks))
            rruns = srun[ruse]
            lstarts = get_starts(luse, looks, n)
            rstarts = get_starts(ruse, looks, n)
            lspks, lwg = get_spikecourse(lruns, pretime=pretime, posttime=tlim, 
                                         binsize=binsize, mantimes=lstarts)
            rspks, rwg = get_spikecourse(rruns, pretime=pretime, posttime=tlim, 
                                         binsize=binsize, mantimes=rstarts)
            l_wgts = np.concatenate((l_wgts, lwg))
            r_wgts = np.concatenate((r_wgts, rwg))
            if j == 0:
                alllspks = lspks
                allrspks = rspks
            else:
                alllspks = np.concatenate((alllspks, lspks), axis=0)
                allrspks = np.concatenate((allrspks, rspks), axis=0)
        l = labels[i]
        sunits[l] = {}
        sunits[l]['l'] = alllspks*(1000. / binsize)
        sunits[l]['r'] = allrspks*(1000. / binsize)
        avgunits[l] = {}
        avgunits[l]['l'] = np.average(sunits[l]['l'], axis=0, 
                                      weights=l_wgts > 0)
        avgunits[l]['r'] = np.average(sunits[l]['r'], axis=0, 
                                      weights=r_wgts > 0)
    return avgunits, sunits

def show_nlook_activity(data, tts, n, labels, pretime=-100, binsize=2, 
                        timefield='fixation_off', imgon=0, imgoff=5000, 
                        drunfield='datanum', tlim=300, eyefield='eyepos', 
                        lloc=-3, rloc=3, stimoff=3):
    avgus, sus = get_nlook_activity(data, tts, n, labels, pretime=pretime, 
                                    binsize=binsize, timefield=timefield, 
                                    imgon=imgon, imgoff=imgoff, 
                                    drunfield=drunfield, tlim=tlim, 
                                    eyefield=eyefield, lloc=lloc, rloc=rloc, 
                                    stimoff=stimoff)
    xs = np.arange(pretime, tlim, binsize)
    fig_avg = plt.figure()
    fig_su = plt.figure()
    avgdim = np.sqrt(len(avgus.keys()))
    su_dep = len(avgus.keys())
    su_wid = 2
    for i, k in enumerate(avgus.keys()):
        avgax = fig_avg.add_subplot(avgdim, avgdim, i+1)
        avgax.plot(xs, avgus[k]['l'], label='l')
        avgax.plot(xs, avgus[k]['r'], label='r')
        avgax.set_title(k)
        avgax.legend()

        suax_l = fig_su.add_subplot(su_dep, su_wid, 2*i+1)
        suax_l = show_single_units(sus[k]['l'], pretime=pretime, posttime=tlim, 
                                   binsize=binsize, ax=suax_l)
        suax_l.set_title(k)
        suax_l.set_ylabel('units (l)')

        suax_r = fig_su.add_subplot(su_dep, su_wid, 2*i+2)
        suax_r = show_single_units(sus[k]['r'], pretime=pretime, posttime=tlim,
                                   binsize=binsize, ax=suax_r)
        suax_r.set_ylabel('units (r)')
    plt.show(block=False)
    return avgus, sus

def show_nov_fam_suavg(sus, pretime, tlim, binsz):
    xs = np.arange(pretime, tlim, binsz) + binsz/2.
    novs = np.concatenate((sus['7']['r'], sus['9']['l'], sus['9']['r'], 
                           sus['10']['l']), axis=0)
    fams = np.concatenate((sus['7']['l'], sus['8']['l'], sus['8']['r'], 
                           sus['10']['r']), axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(xs, novs.mean(0), label='nov')
    ax.plot(xs, fams.mean(0), label='fam')
    ax.legend()
    plt.show(block=False)
    return novs, fams
    
def show_trialtype_spikecourse(data, tts, labels, pretime=-100, posttime=5000,
                               binsize=2, timefield='fixation_off', imgon=0, 
                               imgoff=5000, drunfield='datanum'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = np.arange(pretime, posttime, binsize)
    druns = get_data_run_nums(data, drunfield)
    alltcs = {}
    meantcs = {}
    for i, t in enumerate(tts):
        wgts = []
        for j, dr in enumerate(druns):
            srun = data[data[drunfield] == dr]
            spktc, nt = get_spikecourse(srun, t, pretime=pretime, 
                                        posttime=posttime, binsize=binsize, 
                                        timefield=timefield)
            wgts = np.concatenate((wgts, nt))
            if j == 0:
                allspktc = spktc
            else:
                allspktc = np.concatenate((allspktc, spktc), axis=0)
        allspktc = (1000. / binsize)*allspktc
        alltcs[labels[i]] = allspktc
        meantcs[labels[i]] = np.average(allspktc, axis=0, weights=wgts > 10)
        ax.plot(xs, meantcs[labels[i]], label=labels[i])
    ymin, ymax = ax.get_ylim()
    ax.vlines([imgon, imgoff], ymin, ymax)
    ax.set_xlabel('ms')
    ax.set_ylabel('spks/s')
    ax.legend()
    plt.show(block=False)
    return meantcs, alltcs

def show_all_single_units(sudict, pretime=-100, posttime=5000, binsize=2, 
                          norm=True):
    fig = plt.figure()
    figdim = np.sqrt(len(sudict.keys()))
    for i, k in enumerate(sudict.keys()):
        ax = fig.add_subplot(figdim, figdim, i+1)
        ax = show_single_units(sudict[k], pretime=pretime, posttime=posttime,
                               binsize=binsize, ax=ax, norm=norm)
        ax.set_title(k)
    plt.show()
    return fig

def show_single_units(sus, pretime=-100, posttime=5000, binsize=2, ax=None, 
                      norm=True):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    xs = np.arange(pretime, posttime, binsize)
    if norm:
        sus = (sus - np.reshape(sus.min(1), (-1, 1))) / np.reshape(sus.max(1), 
                                                                  (-1, 1))
    extent = (xs[0], xs[-1], 1, sus.shape[0])
    ax.imshow(sus, interpolation='none', extent=extent)
    ax.set_xlabel('time')
    ax.set_ylabel('unit')
    return ax

def show_separate_units(sus, pretime=-100, posttime=5000, binsize=2, xs=None, 
                        add=1, sameplot=False):
    for i, k in enumerate(sus.keys()):
        if not (sameplot and i > 0):
            f = plt.figure()
        spk_cs = sus[k]
        aim = np.ceil(np.sqrt(spk_cs.shape[0]))
        if xs is None:
            xs = np.arange(pretime, posttime, binsize)
        for i, spks in enumerate(spk_cs):
            ax = f.add_subplot(aim, aim, i+1)
            ax.plot(xs, spks)
            maxspk = np.max(spks) + add
            minspk = np.min((0, np.min(spks)))
            ax.vlines(0, minspk, maxspk, linestyles='dashed')
            ax.set_ylim([minspk, maxspk])
            ax.set_yticks([minspk, maxspk])
            ax.set_xticks([])
        f.suptitle(k)
    plt.show(block=False)
    return f
            
