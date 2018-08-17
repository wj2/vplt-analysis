
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import scipy.stats as sts
from sklearn import svm

from eyes import *
from utility import *

def binarize_spiketimes(spts, binsize, bounds, accumulate=False):
    nbins = np.ceil((bounds[1] - bounds[0]) / binsize)
    bin_range = (bounds[0], bounds[0] + binsize*nbins)
    bspks, _ = np.histogram(spts, nbins, bin_range)
    if accumulate:
        aspks = np.zeros_like(bspks)
        for i in xrange(len(bspks)):
            aspks[i] = np.sum(bspks[:i+1])
        bspks = aspks
    return bspks    

def accumulate_spiketimes(spts, binsize, bounds):
    nbins = np.ceil((bounds[1] - bounds[0]) / binsize)
    bin_range = (bounds[0], bounds[0] + binsize*nbins)
    bspks, _ = np.histogram(spts, nbins, bin_range)
    aspks = np.zeros_like(bspks)
    for i in xrange(len(bspks)):
        aspks[i] = np.sum(bspks[:i+1])
    return aspks

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

def sample_trials_svm(dims, n, with_replace=False):
    trls = np.zeros((len(dims), n, dims[0].shape[1]))
    for i, d in enumerate(dims):
        trl_inds = np.random.choice(d.shape[0], n, replace=with_replace)
        trls[i, :, :] =  d[trl_inds, :]
    return trls

def fold_svm(cat1, cat2, leave_out=1, norm=True, eps=.00001,
             shuff_labels=False):
    alltr = np.concatenate((cat1, cat2), axis=1)
    alllabels = np.concatenate((np.zeros(cat1.shape[1]), 
                                np.ones(cat2.shape[1])))
    inds = np.arange(alltr.shape[1])
    np.random.shuffle(inds)
    alltr = alltr[:, inds, :]
    if shuff_labels:
        inds = np.arange(alltr.shape[1])
        np.random.shuffle(inds)
    alllabels = alllabels[inds]
    if norm:
        mu = alltr.mean(1).reshape((alltr.shape[0], 1, alltr.shape[2]))
        sig = alltr.std(1).reshape((alltr.shape[0], 1, alltr.shape[2]))
        sig[sig < eps] = 1.
        alltr = (alltr - mu)/sig
    folds_n = int(np.floor(alltr.shape[1] / leave_out))
    results = np.zeros((folds_n, cat1.shape[2]))
    for i in xrange(folds_n):
        train_tr = np.concatenate((alltr[:, (i+1)*leave_out:], 
                                   alltr[:, :i*leave_out]),
                                  axis=1)
        train_l = np.concatenate((alllabels[(i+1)*leave_out:], 
                                  alllabels[:i*leave_out]))

        test_tr = alltr[:, i*leave_out:(i+1)*leave_out]
        test_l = alllabels[i*leave_out:(i+1)*leave_out]
        results[i, :] = svm_decode_tc(train_tr, train_l, test_tr, test_l)
    mr = np.mean(results, axis=0)
    return mr, results, alltr

def svm_decode_tc(train, trainlabels, test, testlabels):
    percent_corr = np.zeros(train.shape[2])
    for i in xrange(train.shape[2]):
        s = svm.SVC()
        s.fit(train[:, :, i].T, trainlabels)
        preds = s.predict(test[:, :, i].T)
        percent_corr[i] = np.sum(preds == testlabels) / float(len(testlabels))
    return percent_corr

def svm_decoding(cat1, cat2, leave_out=1, require_trials=15, resample=100,
                 with_replace=False, shuff_labels=False):
    bool1 = [x.shape[0] < require_trials for x in cat1]
    bool2 = [x.shape[0] < require_trials for x in cat2]
    combool = np.logical_not(np.logical_or(bool1, bool2))
    cat1_f = cat1[combool]
    cat2_f = cat2[combool]
    tcs = np.zeros((resample, cat1_f[0].shape[1]))
    for i in xrange(resample):
        cat1_samp = sample_trials_svm(cat1_f, require_trials, with_replace)
        cat2_samp = sample_trials_svm(cat2_f, require_trials, with_replace)
        tcs[i, :], _, _ = fold_svm(cat1_samp, cat2_samp, leave_out, 
                                shuff_labels=shuff_labels)
    return tcs, cat1_f, cat2_f

def get_spikecourse(data, trial_type=None, pretime=-100, posttime=5000, 
                    binsize=2, timefield='fixation_off', ttfield='trial_type', 
                    mantimes=None, spksfield='spike_times', excl_empty=False,
                    smooth=False, step=5, boots=0, cumulative=False):
    if trial_type is not None:
        trls = data[data[ttfield] == trial_type]
        if mantimes is not None:
            mantimes = mantimes[data[ttfield] == trial_type]
    else:
        trls = data
    out = get_spikecourse_raster(trls, pretime, posttime,
                                 binsize, timefield, mantimes, 
                                 spksfield, excl_empty, smooth, step, boots,
                                 accumulate=cumulative)
    return out

def get_spikecourse_raster(trls, pretime=-100, posttime=5000, 
                           binsize=2, timefield='fixation_off', 
                           mantimes=None, spksfield='spike_times', 
                           excl_empty=False, smooth=False, step=5,
                           boots=0, accumulate=False):
    numtrains = len(trls[spksfield])
    if len(trls[spksfield]) > 0:
        spkscoll = np.concatenate(trls[spksfield], axis=0)
    else:
        spkscoll = np.array([[]])
    numtrains = np.ones(spkscoll.shape[1])*len(trls[spksfield])
    if smooth:
        nbins = np.ceil((posttime - pretime + binsize) / float(step))
        posttime = posttime + binsize
        filtwin = np.ones(binsize/step) / float(binsize)
        binspks = np.zeros((spkscoll.shape[1], 
                            nbins - (binsize/float(step)) + 1))
    else:
        nbins = np.ceil((posttime - pretime) / binsize)
        binspks = np.zeros((spkscoll.shape[1], nbins))
    if mantimes is None:
        centspks = spkscoll - np.reshape(trls[timefield], (-1, 1))
    else:
        centspks = spkscoll - mantimes
    trlbounds = (pretime, posttime)
    if boots > 0:
        allboots = []
    else:
        allboots = None
    for i in xrange(centspks.shape[1]):
        if excl_empty:
            emfilt = filter(lambda x: len(x) > 0, centspks[:, i])
            normfact = float(len(emfilt))
            if len(emfilt) == 0:
                collectall = np.array([])
            else:
                collectall = np.concatenate(emfilt, axis=0)
        else:
            normfact = float(centspks.shape[0])
            collectall = np.concatenate(centspks[:, i], axis=0)
        if smooth:
            prefilt = binarize_spiketimes(collectall, step, trlbounds, 
                                          accumulate=accumulate)
            binspks[i, :] = np.convolve(prefilt, filtwin, mode='valid')
        else:
            binspks[i, :] = binarize_spiketimes(collectall, binsize, trlbounds,
                                                accumulate=accumulate)
        if normfact > 0:
            binspks[i, :] = binspks[i, :] / normfact
        if boots > 0:
            if smooth:
                trls_bin = np.array([
                    np.convolve(binarize_spiketimes(x, step, 
                                                    trlbounds,
                                                    accumulate=accumulate),
                                filtwin, mode='valid')
                                     for x in centspks[:, i]])
            else:
                trls_bin = np.array([binarize_spiketimes(x, binsize, trlbounds,
                                                         accumulate=accumulate) 
                                     for x in centspks[:, i]])
            # boot_trls = bootstrap_sus(trls_bin, boots=boots)
            allboots.append(trls_bin)            
    if boots > 0:
        ab_new = np.ones(len(allboots), dtype=object)
        ab_new[:] = allboots
    return binspks, numtrains, ab_new

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

def merge_conditions(neurs):
    n_neurs = len(neurs[0])
    nts = neurs[0][0].shape[1]
    allneurs = np.zeros(n_neurs, dtype=object)
    for i in xrange(n_neurs):
        trs = filter(lambda y: not np.isnan(y).all(), 
                     map(lambda x: x[i], neurs))
        if len(trs) > 0:
            allneurs[i] = np.concatenate(trs, axis=0)
        else:
            allneurs[i] = np.zeros((0, nts))
    return allneurs

def get_nsacc_activity(data, n, tts, labels, pretime=-100,
                       binsize=50, step=5, timefield='fixation_off', imgon=0,
                       imgoff=5000, drunfield='datanum', tlim=300, 
                       eyefield='eyepos', lloc=-3, rloc=3, stimoff=3,
                       ttfield='trial_type', slock=True, excl_empty=True,
                       spksfield='spike_times', boots=0, cumulative=False,
                       smooth=True, thr=.1, lowfilt=50):
    druns = get_data_run_nums(data, drunfield)
    sunits = {}
    sboots = {}
    avgunits = {}
    if smooth:
        xs = np.arange(pretime + binsize/2., tlim + binsize/2. + step, step)
    else:
        xs = np.arange(pretime + binsize/2., tlim + binsize/2., 
                       binsize)
    druns = get_data_run_nums(data, drunfield)
    for i, t in enumerate(tts):
        ttrials = data[data[ttfield] == t]
        # druns = get_data_run_nums(ttrials, drunfield)
        for j, dr in enumerate(druns):
            srun = ttrials[ttrials[drunfield] == dr]
            eps = srun[eyefield]
            # afters = np.concatenate(srun[timefield])
            t_lens, t_looks, t_s_bs, t_s_es = get_fixtimes(srun, [t], 
                                                           thr=thr,
                                                           postthr=timefield)
            
            luse = np.array(map(lambda x: len(x) > n and x[n] == 'l', 
                                t_looks[t]))
            if len(luse) == 0:
                lruns = srun
                lstarts = np.reshape(map(lambda x: x[n], t_s_bs[t]), 
                                     (-1, 1))
            else:
                lruns = srun[luse]
                lstarts = np.reshape(map(lambda x: x[n], t_s_bs[t][luse]), 
                                     (-1, 1))
            ruse = np.array(map(lambda x: len(x) > n and x[n] == 'r', 
                                t_looks[t]))
            if len(ruse) == 0:
                rruns = srun
                rstarts = np.reshape(map(lambda x: x[n], t_s_bs[t]), (-1, 1))
            else: 
                rruns = srun[ruse]
                rstarts = np.reshape(map(lambda x: x[n], t_s_bs[t][ruse]), 
                                     (-1, 1))
            if not (lstarts.shape[0] == 0 or rstarts.shape[0] == 0):
                if slock:
                    mantimes_l = lstarts
                    mantimes_r = rstarts
                else:
                    mantimes_l = None
                    mantimes_r = None
                lspks, lwg, lbs = get_spikecourse(lruns, pretime=pretime, 
                                                  posttime=tlim, 
                                                  binsize=binsize, 
                                                  mantimes=mantimes_l, 
                                                  excl_empty=excl_empty, 
                                                  smooth=smooth, step=step,
                                                  boots=boots, 
                                                  cumulative=cumulative)
                rspks, rwg, rbs = get_spikecourse(rruns, pretime=pretime, 
                                                  posttime=tlim, 
                                                  binsize=binsize,
                                                  mantimes=mantimes_r, 
                                                  excl_empty=excl_empty, 
                                                  smooth=smooth, step=step,
                                                  boots=boots,
                                                  cumulative=cumulative)
                n_neurs = rspks.shape[0]
            else:
                n_neurs = data[data[drunfield] == dr][0][spksfield].shape[1]
                if smooth:
                    t_pts = np.ceil((tlim - pretime) / float(step) + 1)
                else:
                    t_pts = np.ceil((tlim - pretime) / float(binsize))
                lspks = np.zeros((n_neurs, t_pts))
                lspks[:, :] = np.nan
                rspks = np.zeros((n_neurs, t_pts))
                rspks[:, :] = np.nan
                lwg = np.zeros(n_neurs)
                rwg = np.zeros(n_neurs)
                lbs = np.zeros(n_neurs, dtype=object)
                lbs_ins = np.zeros(0, t_pts)
                lbs[:] = np.nan
                rbs = np.zeros(n_neurs, dtype=object)
                rbs[:] = np.nan
            neur_record = np.zeros((n_neurs, 2))
            neur_record[:, 0] = dr
            neur_record[:, 1] = np.arange(n_neurs) + 1
            if j == 0:
                alllspks = lspks
                allrspks = rspks
                l_wgts = lwg
                r_wgts = rwg
                all_neur_record = neur_record
                l_boots = lbs
                r_boots = rbs
            else:
                alllspks = np.concatenate((alllspks, lspks), axis=0)
                allrspks = np.concatenate((allrspks, rspks), axis=0)
                l_wgts = np.concatenate((l_wgts, lwg))
                r_wgts = np.concatenate((r_wgts, rwg))
                all_neur_record = np.concatenate((all_neur_record, 
                                                  neur_record), axis=0)
                r_boots = np.concatenate((r_boots, rbs), axis=0)
                l_boots = np.concatenate((l_boots, lbs), axis=0)
        l = labels[i]
        sunits[l] = {}
        sboots[l] = {}
        if cumulative:
            conv_mult = 1.
        else:
            conv_mult = 1000.
        sunits[l]['l'] = alllspks*conv_mult
        sunits[l]['r'] = allrspks*conv_mult
        sboots[l]['l'] = l_boots*conv_mult
        sboots[l]['r'] = r_boots*conv_mult
        avgunits[l] = {}
        avgunits[l]['l'] = np.average(sunits[l]['l'], axis=0, 
                                      weights=l_wgts > 0)
        avgunits[l]['r'] = np.average(sunits[l]['r'], axis=0, 
                                      weights=r_wgts > 0)
    return avgunits, sunits, xs, all_neur_record, sboots

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
    plt.show()
    return avgus, sus

def get_nlook_activity(data, n, tts, labels, pretime=-100,
                       binsize=2, timefield='fixation_off', imgon=0,
                       imgoff=5000, drunfield='datanum', tlim=300, 
                       eyefield='eyepos', lloc=-3, rloc=3, stimoff=3,
                       ttfield='trial_type', boots=0):
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
            lspks, lwg, lbs = get_spikecourse(lruns, pretime=pretime, 
                                              posttime=tlim, binsize=binsize,
                                              mantimes=lstarts, boots=boots)
            rspks, rwg, rbs = get_spikecourse(rruns, pretime=pretime, 
                                              posttime=tlim, binsize=binsize, 
                                              mantimes=rstarts, boots=boots)
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
    plt.show()
    return avgus, sus

def comb_ci_saccimg(img_sus, xs_i, sacc_sus, xs_s, ipsicontra='contra', 
                    base=None, title='', figsize=(14, 5), suptitle='',
                    save=False, savename='saccimg_fig.pdf', save_half=False, 
                    perms=0, comp_sacc=False, novcol='g', famcol='b'):
    f = plt.figure(figsize=figsize)
    ax_img = f.add_subplot(1, 2, 1)
    show_contra_ipsi_suavg(img_sus, xs_i, ipsicontra=ipsicontra, base=base, 
                           title=title, ax=ax_img,
                           perms=perms, comp_sacc=comp_sacc, novcol=novcol,
                           famcol=famcol, xlabel='time from image onset (ms)')
    ax_sacc = f.add_subplot(1, 2, 2)
    show_contra_ipsi_suavg(sacc_sus, xs_s, ipsicontra=ipsicontra, base=base,
                           title=title, ax=ax_sacc,
                           perms=perms, comp_sacc=comp_sacc, novcol=novcol,
                           famcol=famcol, 
                           xlabel='time from first saccade (ms)')
    f.suptitle(suptitle)
    f.tight_layout()
    if save_half:
        ax_sacc.set_frame_on(False)
        ax_sacc.get_xaxis().set_visible(False)
        ax_sacc.get_yaxis().set_visible(False)
        ax_sacc.get_legend().set_visible(False)
        ax_sacc.set_title('')
        [l.set_visible(False) for l in ax_sacc.get_lines()]
        name, ext = os.path.splitext(savename)
        halfname = name + '_half' + ext
        f.savefig(halfname, bbox_inches='tight')
        ax_sacc.set_frame_on(True)
        ax_sacc.get_xaxis().set_visible(True)
        ax_sacc.get_yaxis().set_visible(True)
        ax_sacc.get_legend().set_visible(True)
        ax_sacc.set_title(title)
        [l.set_visible(True) for l in ax_sacc.get_lines()]
    if save:
        f.savefig(savename, bbox_inches='tight')
    plt.show()

def fam_latency_popperm(novs_bs, fams_bs, xs, perms=500, trlen=10, 
                        balance_trials=False):    
    perms_a = np.zeros((perms, len(xs)))
    p_pop = np.zeros((len(novs_bs), len(xs)))
    obs_diff = np.zeros((len(novs_bs), len(xs)))
    novs = np.zeros((len(novs_bs), len(xs)))
    fams = np.zeros((len(novs_bs), len(xs)))
    for i in xrange(perms):
        c = range(len(novs_bs)) # np.random.choice(range(len(novs_bs)), len(novs_bs))
        nneurs = novs_bs[c]
        fneurs = fams_bs[c]
        for j, neur in enumerate(nneurs):
            if (np.isnan(neur).all() or np.isnan(fneurs[j]).all()
                or neur.shape[0] < trlen or fneurs[j].shape[0] < trlen):
                p_pop[j, :] = np.nan
                if i == 0:
                    novs[j, :] = np.nan
                    fams[j, :] = np.nan
                    obs_diff[j, :] = np.nan
            else:
                nn, fn = neur, fneurs[j]
                len_n, len_f = len(nn), len(fn)
                if i == 0:
                    if balance_trials:
                        n_inds = range(len_n)
                        f_inds = range(len_f)
                        np.random.shuffle(n_inds)
                        np.random.shuffle(f_inds)
                        len_n = min(len_n, len_f)
                        len_f = len_n
                        nn = nn[n_inds[:len_n]]
                        fn = fn[f_inds[:len_f]]
                    novs[j, :] = np.nanmean(nn, 0)
                    fams[j, :] = np.nanmean(fn, 0)
                    obs_diff[j, :] = (novs[j, :] - fams[j, :])
                pool = np.concatenate((nn, fn), axis=0)
                pool_inds = range(len(pool))
                np.random.shuffle(pool_inds)
                shuff_pool = pool[pool_inds]
                nn_c = shuff_pool[:len_n]
                fn_c = shuff_pool[len_n:]
                assert len(fn_c) == len_f
                assert len(nn_c) == len_n
                p_pop[j, :] = np.nanmean(nn_c, 0) - np.nanmean(fn_c, 0)
        perms_a[i, :] = np.nanmean(p_pop, 0)
    obs_diff_a = np.nanmean(obs_diff, axis=0)
    novs_a = np.nanmean(novs, axis=0)
    fams_a = np.nanmean(fams, axis=0)
    ps_high = np.sum(obs_diff_a < perms_a, 0)/float(perms)
    ps_low = np.sum(obs_diff_a > perms_a, 0)/float(perms)
    return perms_a, novs_a, fams_a, obs_diff_a, ps_high, ps_low

def compute_perm_conds(xs_io, sb_io, xs_so, sb_so, 
                       nov_conds, nov_looks, fam_conds, 
                       fam_looks, nov_condname, fam_condname, savename, 
                       nov_color=(.5, 0, 0), 
                       nov_line='-', fam_color=(0, 0, .5), fam_line='-', 
                       diff_color=(0, .5, 0), diff_line='-', siglev=19,
                       diff_siglev=10, pval=.01, figsize=(10.87, 3.8),
                       n_perms=2500, balance_trials=False):
    novs_bss = merge_conditions([sb_so[cond][nov_looks[i]] 
                                 for i, cond in enumerate(nov_conds)])
    fams_bss = merge_conditions([sb_so[cond][fam_looks[i]] 
                                 for i, cond in enumerate(fam_conds)])

    out1 = fam_latency_popperm(novs_bss, fams_bss, xs_so, perms=n_perms, 
                                    trlen=0, balance_trials=balance_trials)
    perms_s, novs_as, fams_as, obsdiff_s, ps_sl, ps_sh = out1

    novs_bsi = merge_conditions([sb_io[cond][nov_looks[i]] 
                                 for i, cond in enumerate(nov_conds)])
    fams_bsi = merge_conditions([sb_io[cond][fam_looks[i]] 
                                 for i, cond in enumerate(fam_conds)])

    out2 = fam_latency_popperm(novs_bsi, fams_bsi, xs_io, perms=n_perms, 
                                    trlen=0, balance_trials=balance_trials)
    perms_i, novs_ai, fams_ai, obsdiff_i, ps_il, ps_ih = out2
    out = plot_traces_diff(xs_io, novs_ai, fams_ai, ps_il, ps_ih, xs_so, novs_as, 
                           fams_as, ps_sl, ps_sh, nov_condname, fam_condname,
                           savename, nov_color=nov_color, nov_line=nov_line, 
                           fam_color=fam_color, fam_line=fam_line, siglev=siglev,
                           diff_color=diff_color, diff_line=diff_line, 
                           diff_siglev=diff_siglev, figsize=figsize, pval=pval)
    return out

def plot_traces_diff(xs_io, novs_ai, fams_ai, ps_il, ps_ih, xs_so, novs_as, 
                     fams_as, ps_sl, ps_sh, nov_condname, fam_condname,
                     savename, nov_color=(.5, 0, 0), nov_line='-', 
                     fam_color=(0, 0, .5), fam_line='-', siglev=19,
                     diff_color=(0, .5, 0), diff_line='-', 
                     diff_siglev=10, figsize=(10.87, 3.8), pval=.005):
    f = plt.figure(figsize=figsize)
    ax_i = f.add_subplot(1, 2, 1)
    _ = ax_i.plot(xs_io, novs_ai, color=nov_color, linestyle=nov_line)
    _ = ax_i.plot(xs_io, fams_ai, color=fam_color, linestyle=fam_line)
    pxs_il = xs_io[ps_il < pval]
    pxs_ih = xs_io[ps_ih < pval]
    pxs_i = np.concatenate((pxs_il[:1], pxs_ih[:1]))
    if len(pxs_i) > 0:
        print 'image', np.min(pxs_i)
    else:
        print 'image no sig'
    _ = ax_i.plot(pxs_il, np.ones_like(pxs_il)*siglev, '*', color=nov_color)
    _ = ax_i.plot(pxs_ih, np.ones_like(pxs_ih)*siglev, '*', color=fam_color)
    _ = ax_i.set_ylim(min(np.min(novs_ai), np.min(fams_ai)) - 1, siglev +1)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['top'].set_visible(False)
    ax_i.yaxis.set_ticks_position('left')
    ax_i.xaxis.set_ticks_position('bottom')
    ax_i.set_xlabel('time from image onset (ms)')
    ax_i.set_ylabel('spikes/s')
    
    ax_s = f.add_subplot(1, 2, 2)
    _ = ax_s.plot(xs_so, novs_as, color=nov_color, linestyle=nov_line, 
                  label='to {}'.format(nov_condname))
    _ = ax_s.plot(xs_so, fams_as, color=fam_color, linestyle=fam_line, 
                  label='to {}'.format(fam_condname))
    pxs_sl = xs_so[ps_sl < pval]
    pxs_sh = xs_so[ps_sh < pval]
    _ = ax_s.plot(pxs_sl, np.ones_like(pxs_sl)*siglev, '*', color=nov_color)
    _ = ax_s.plot(pxs_sh, np.ones_like(pxs_sh)*siglev, '*', color=fam_color)
    pxs_s = np.concatenate((pxs_sl[:1], pxs_sh[:1]))
    if len(pxs_s) > 0:
        print 'sacc', np.min(pxs_s)
    else:
        print 'sacc no sig'
    _ = ax_s.set_ylim(min(np.min(novs_as), np.min(fams_as)) - 1, siglev +1)
    ax_s.spines['right'].set_visible(False)
    ax_s.spines['top'].set_visible(False)
    ax_s.yaxis.set_ticks_position('left')
    ax_s.xaxis.set_ticks_position('bottom')
    ax_s.set_xlabel('time from first saccade (ms)')
    ax_s.set_ylabel('spikes/s')
    ax_s.legend(frameon=False)
    f.tight_layout()
    f.savefig(savename, bbox_inches='tight')
    
    sn, ext = savename.split('.')
    savename_diff = sn + '_diff.' + ext
    f2 = plt.figure(figsize=figsize)
    ax_i = f2.add_subplot(1, 2, 1)
    ax_i = plot_traces(ax_i, xs_io, [novs_ai - fams_ai], [diff_color], [diff_line],
                       ['{} - {}'.format(nov_condname, fam_condname)], 
                       [np.concatenate((pxs_il, pxs_ih))], showlegend=False)
    ax_i.set_xlabel('time from image onset (ms)')
    # _ = ax_i.plot(xs_io, novs_ai - fams_ai, color=diff_color, linestyle=diff_line)
    # _ = ax_i.plot(pxs_il, np.ones_like(pxs_il)*diff_siglev, '*', color=nov_color)
    # _ = ax_i.plot(pxs_ih, np.ones_like(pxs_ih)*diff_siglev, '*', color=fam_color)
    # _ = ax_i.set_ylim(np.min(novs_ai - fams_ai) - 1, diff_siglev +1)
    # ax_i.spines['right'].set_visible(False)
    # ax_i.spines['top'].set_visible(False)
    # ax_i.yaxis.set_ticks_position('left')
    # ax_i.xaxis.set_ticks_position('bottom')
    # ax_i.set_xlabel('time from image onset (ms)')
    # ax_i.set_ylabel('spikes/s')
    
    ax_s = f2.add_subplot(1, 2, 2)
    ax_s = plot_traces(ax_s, xs_so, [novs_as - fams_as], [diff_color], [diff_line],
                       ['{} - {}'.format(nov_condname, fam_condname)], 
                       [np.concatenate((pxs_sl, pxs_sh))])
    # _ = ax_s.plot(xs_so, novs_as - fams_as, color=diff_color, linestyle=diff_line, 
    #               label='{} - {}'.format(nov_condname, fam_condname))
    # _ = ax_s.plot(pxs_sl, np.ones_like(pxs_sl)*diff_siglev, '*', color=nov_color)
    # _ = ax_s.plot(pxs_sh, np.ones_like(pxs_sh)*diff_siglev, '*', color=fam_color)
    # _ = ax_s.set_ylim(np.min(novs_as - fams_as) - 1, diff_siglev +1)
    # ax_s.spines['right'].set_visible(False)
    # ax_s.spines['top'].set_visible(False)
    # ax_s.yaxis.set_ticks_position('left')
    # ax_s.xaxis.set_ticks_position('bottom')
    # ax_s.set_xlabel('time from first saccade (ms)')
    # ax_s.set_ylabel('spikes/s')
    # ax_s.legend(frameon=False)
    f2.tight_layout()
    f2.savefig(savename_diff, bbox_inches='tight')

    return novs_ai, fams_ai, pxs_il, pxs_ih, novs_as, fams_as, pxs_sl, pxs_sh

def plot_traces(ax, xs, traces, colors, linestyles, labels, sigps, 
                showlegend=True):
    alltr = np.concatenate(traces)
    maxd = np.max(alltr) + 1
    _ = ax.set_ylim(np.min(alltr) - 1, maxd + len(traces) + 1)
    for i, tr in enumerate(traces):
        _ = ax.plot(xs, tr, color=colors[i], linestyle=linestyles[i], 
                    label=labels[i])
        _ = ax.plot(sigps[i], np.ones_like(sigps[i])*(maxd + i), '*', 
                    color=colors[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('time from first saccade (ms)')
    ax.set_ylabel('spikes/s')
    if showlegend:
        ax.legend(frameon=False)
    return ax

def fam_latency_pop(bs, xs, boots=100):
    novs_bs = np.concatenate((bs['10']['l'], bs['9']['l']), axis=0)
    fams_bs = np.concatenate((bs['8']['l'], bs['7']['l']), axis=0)
    perm_bs = np.concatenate((novs_bs, fams_bs), axis=0)
    f_ps = np.zeros((boots, len(xs)))
    n_ps = np.zeros((boots, len(xs)))
    nf_ps = np.zeros((boots, len(xs)))
    perm_ps = np.zeros((boots, len(xs)))
    ln = len(novs_bs)
    ln_r = range(ln)
    n_pop = np.zeros((ln, len(xs)))
    lf = len(fams_bs)
    lf_r = range(lf)
    f_pop = np.zeros((lf, len(xs)))
    p_pop = np.zeros((lf+ln, len(xs)))
    for i in xrange(boots):
        f_neursamp = np.random.choice(lf_r, lf)
        f_neursamp_neurs = fams_bs[f_neursamp]
        n_neursamp = np.random.choice(ln_r, ln)
        n_neursamp_neurs = novs_bs[f_neursamp]
        p_neursamp = np.random.choice(lf_r+ln_r, ln+lf)
        p_neursamp_neurs = perm_bs[p_neursamp]
        for j, neur in enumerate(f_neursamp_neurs):
            if np.isnan(neur).all():
                f_pop[j, :] = np.nan
            else:
                fneur_samp = np.random.choice(range(len(neur)), len(neur))
                f_pop[j, :] = np.nanmean(neur[fneur_samp, :], 0)
        f_ps[i, :] = np.nanmean(f_pop, 0)
        for j, neur in enumerate(n_neursamp_neurs):
            if np.isnan(neur).all():
                n_pop[j, :] = np.nan
            else:
                nneur_samp = np.random.choice(range(len(neur)), len(neur))
                n_pop[j, :] = np.nanmean(neur[nneur_samp, :], 0)
        n_ps[i, :] = np.nanmean(n_pop, 0)
        for j, neur in enumerate(p_neursamp_neurs):
            if np.isnan(neur).all():
                p_pop[j, :] = np.nan
            else:
                pneur_samp = np.random.choice(range(len(neur)), len(neur))
                p_pop[j, :] = np.nanmean(neur[pneur_samp, :], 0)
        nf_ps[i, :] = np.nanmean(n_pop - f_pop, 0)
        perm_ps[i, :] = np.nanmean(p_pop[:lf] - p_pop[lf:], 0)
    ts, ps = sts.ttest_ind(n_ps, f_ps, axis=0, equal_var=False, 
                           nan_policy='omit')
    ts2, ps2 = sts.ttest_ind(nf_ps, perm_ps, axis=0, equal_var=False, 
                             nan_policy='omit')
    return n_ps, f_ps, ps, ps2

def fam_latency(sus, bs, xs):
    novs = np.concatenate((sus['10']['l'], sus['9']['l']), axis=0)
    novs_bs = np.concatenate((bs['10']['l'], bs['9']['l']), axis=0)
    fams = np.concatenate((sus['8']['l'], sus['7']['l']), axis=0)
    fams_bs = np.concatenate((bs['8']['l'], bs['7']['l']), axis=0)
    ps, ks = np.zeros_like(novs), np.zeros_like(novs)
    print fams_bs.shape, novs_bs.shape
    for i, neur in enumerate(fams_bs):
        if np.isnan(novs_bs[i]).all() or np.isnan(fams_bs[i]).all():
            ks[i, :] = np.nan
            ps[i, :] = np.nan
        else:
            for j, t in enumerate(xs):
                try:
                    ks[i, j], ps[i, j] = sts.ks_2samp(novs_bs[i][:, j],
                                                      fams_bs[i][:, j])
                except ValueError:
                    ks[i, j], ps[i, j] = np.nan, np.nan
    return ps, ks, novs, novs_bs, fams, fams_bs

def show_contra_ipsi_suavg(sus, xs, ipsicontra='contra', base=None, 
                           title='', ax=None, perms=0, comp_sacc=False,
                           novcol='g', famcol='b', xlabel='time (ms)'):
    if base == 'fam':
        if ipsicontra == 'contra':
            novs = sus['10']['l']
            fams = sus['8']['l']
        elif ipsicontra == 'ipsi':
            novs = sus['7']['r']
            fams = sus['8']['r']
    elif base == 'all':
        if ipsicontra == 'contra':
            novs = np.concatenate((sus['10']['l'], sus['9']['l']), axis=0)
            fams = np.concatenate((sus['8']['l'], sus['7']['l']), axis=0)
            famline = 'solid'
            novline = 'solid'
            if comp_sacc:
                novs_comp = np.concatenate((sus['10']['r'], sus['9']['r']),
                                           axis=0)
                fams_comp = np.concatenate((sus['8']['r'], sus['7']['r']), 
                                           axis=0)
        elif ipsicontra == 'ipsi':
            famline = 'dashed'
            novline = 'dashed'
            novs = np.concatenate((sus['7']['r'], sus['9']['r']), axis=0)
            fams = np.concatenate((sus['8']['r'], sus['10']['r']), axis=0)
            if comp_sacc:
                novs_comp = np.concatenate((sus['7']['l'], sus['9']['l']), 
                                           axis=0)
                fams_comp = np.concatenate((sus['8']['l'], sus['10']['l']), 
                                           axis=0)
    elif base == 'same':
        if ipsicontra == 'contra':
            novs = sus['9']['l']
            fams = sus['8']['l']
        elif ipsicontra == 'ipsi':
            novs = sus['9']['r']
            fams = sus['8']['r']
    elif base == 'nov':
        if ipsicontra == 'contra':
            novs = sus['9']['l']
            fams = sus['7']['l']
            if comp_sacc:
                novs_comp = sus['9']['r']
                fams_comp = sus['7']['r']
        elif ipsicontra == 'ipsi':
            novs = sus['9']['r']
            fams = sus['10']['r']
            if comp_sacc:
                novs_comp = sus['9']['l']
                fams_comp = sus['10']['l']
    elif base == 'off_nov':
        if ipsicontra == 'contra':
            novs = sus['9']['l']
            fams = sus['10']['l']
        elif ipsicontra == 'ipsi':
            novs = sus['9']['r']
            fams = sus['7']['r']
    elif base == 'off_fam':
        if ipsicontra == 'contra':
            novs = sus['7']['l']
            fams = sus['8']['l']
        elif ipsicontra == 'ipsi':
            novs = sus['10']['r']
            fams = sus['8']['r']
    else:
        if ipsicontra == 'contra':
            novs = sus['10']['l']
            fams = sus['7']['l']
        elif ipsicontra == 'ipsi':
            novs = sus['7']['r']
            fams = sus['10']['r']
    if not comp_sacc:
        novs_comp = None
        fams_comp = None
    return plot_t1_t2_avg(novs, fams, xs, title=title, ax=ax, perms=perms,
                          novs_comp=novs_comp, fams_comp=fams_comp,
                          novcol=novcol, famcol=famcol, novline=novline,
                          famline=famline, xlabel=xlabel)

def show_nov_fam_suavg(sus, xs, title='', usekeys=['7', '10'], ax=None):
    if len(usekeys) == 4:
        novs = np.concatenate((sus['7']['r'], sus['9']['l'], sus['9']['r'],
                               sus['10']['l']), axis=0)
        fams = np.concatenate((sus['7']['l'], sus['8']['l'], sus['8']['r'], 
                               sus['10']['r']), axis=0)
    elif len(usekeys) == 2:
        novs = np.concatenate((sus['7']['r'], sus['10']['l']), axis=0)
        fams = np.concatenate((sus['7']['l'], sus['10']['r']), axis=0)
    elif len(usekeys) == 1:
        if usekeys[0] == '7':
            novs = sus['7']['r']
            fams = sus['7']['l']
        elif usekeys[0] == '10':
            novs = sus['10']['l']
            fams = sus['10']['r']
    return plot_t1_t2_avg(novs, fams, xs, title=title, ax=ax)

def bootstrap_sus(sus, boots=0):
    boot_sus = np.zeros((boots, sus.shape[1]))
    for i in xrange(boots):
        samp_inds = np.random.choice(np.arange(sus.shape[0]), sus.shape[0])
        samp = sus[samp_inds, :]
        boot_sus[i, :] = np.nanmean(samp, axis=0)
    return boot_sus

def plot_t1_t2_avg(novs, fams, xs, title='', ax=None, perms=0, starh=18,
                   trlen=0, p_level=.005, novs_comp=None, fams_comp=None,
                   xlabel='time (ms)', novcol='g', famcol='b',
                   novline='solid', famline='solid'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    if perms > 0:
        _, novs_a, fams_a, _, ps, _ = fam_latency_popperm(novs, fams, xs, 
                                                       perms=perms, 
                                                       trlen=trlen)
        ax.plot(xs, fams_a, color=famcol, linestyle=famline, 
                label='to familiar')
        ax.plot(xs, novs_a, color=novcol, linestyle=novline,
                label='to novel')
        sig_xs = xs[ps < p_level]
        ax.plot(sig_xs, np.ones_like(sig_xs)*starh, 'r*')
        if novs_comp is not None:
            _, _, novs_comp_a, _, ps_nc, _ = fam_latency_popperm(novs, novs_comp,
                                                              xs, perms=perms,
                                                              trlen=trlen)
            _, _, fams_comp_a, _, ps_fc, _  = fam_latency_popperm(fams, fams_comp,
                                                              xs, perms=perms,
                                                              trlen=trlen)
            ax.plot(xs, fams_comp_a, 'b--')
            ax.plot(xs, novs_comp_a, 'g--')
            sig_fam_comp = xs[ps_fc < p_level]
            ax.plot(sig_fam_comp, np.ones_like(sig_fam_comp)*(starh - .5), 'b*')
            sig_nov_comp = xs[ps_nc < p_level]
            ax.plot(sig_nov_comp, np.ones_like(sig_nov_comp)*(starh - 1), 'g*')
        if len(sig_xs) > 0:
            print 'first p < {}'.format(p_level), sig_xs[0]
        else:
            print 'no sig difference'
    else:
        ax.plot(xs, np.nanmean(fams, axis=0), 'b', label='to familiar')
        ax.plot(xs, np.nanmean(novs, axis=0), 'g', label='to novel')
        if novs_comp is not None:
            ax.plot(xs, np.nanmean(fams_comp, axis=0), color=famcol, 
                    linestype=famline)
            ax.plot(xs, np.nanmean(novs_comp, axis=0), color=novcol,
                    linestyle=novline)                    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('spikes/s')
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.legend(frameon=False)
    if ax is None:
        plt.show()
    return novs, fams

def extract_rowi(arrdict, i):
    newd = {}
    for k in arrdict.keys():
        newd[k] = {}
        for k2 in arrdict[k].keys():
            newd[k][k2] = arrdict[k][k2][i, :]
    return newd

def comb_ci_sus_saccimg(img_sus, xs_i, sacc_sus, xs_s, inds=None, base=None,
                        titlecontra='', titleipsi='', figsize=(14,5)):
    if inds is None:
        inds = np.arange(img_sus[img_sus.keys()[0]]['l'].shape[0])
    titlecontra = titlecontra + ', contra'
    titleipsi = titleipsi + ', ipsi'
    for i in inds:
        suptit = 'neuron {}'.format(i)
        comb_ci_saccimg(extract_rowi(img_sus, [i]), xs_i, 
                        extract_rowi(sacc_sus, [i]), xs_s, ipsicontra='contra',
                        base=base, title=titlecontra, figsize=figsize, 
                        suptitle=suptit)
        comb_ci_saccimg(extract_rowi(img_sus, [i]), xs_i, 
                        extract_rowi(sacc_sus, [i]), xs_s, ipsicontra='ipsi',
                        base=base, title=titleipsi, figsize=figsize,
                        suptitle=suptit)
        
def show_nov_fam_singles(sus, xs, inds=None, comp=None, title='', rec=None,
                         contraipsi=None):
    if len(sus.keys()) == 4:
        if contraipsi is not None:
            if contraipsi == 'contra':
                novs = np.dstack((sus['9']['l'], sus['10']['l']))
                fams = np.dstack((sus['8']['l'], sus['7']['l']))
            else:
                novs = np.dstack((sus['9']['r'], sus['7']['r']))
                fams = np.dstack((sus['8']['r'], sus['10']['r']))
        else:
            novs = np.dstack((sus['7']['r'], sus['9']['l'], sus['9']['r'], 
                              sus['10']['l']))
            fams = np.dstack((sus['7']['l'], sus['8']['l'], sus['8']['r'], 
                              sus['10']['r']))
        if comp is not None:
            comp_novs = np.dstack((comp['7']['r'], comp['9']['l'], 
                                   comp['9']['r'], comp['10']['l']))
            comp_fams = np.dstack((comp['7']['l'], comp['8']['l'], 
                                   comp['8']['r'], comp['10']['r']))
    elif len(sus.keys()) == 2:
        novs = np.dstack((sus['7']['r'], sus['10']['l']))
        fams = np.dstack((sus['7']['l'], sus['10']['r']))
        if comp is not None:
            comp_novs = np.dstack((comp['7']['r'], comp['10']['l']))
            comp_fams = np.dstack((comp['7']['l'], comp['10']['r']))
    novs = np.mean(novs, axis=2)
    fams = np.mean(fams, axis=2)
    if comp is not None:
        comp_novs = np.mean(comp_novs, axis=2)
        comp_fams = np.mean(comp_fams, axis=2)
    if inds is None:
        a = sus.keys()[0]
        b = sus[a].keys()[0]
        inds = np.arange(sus[a][b].shape[0])
    for i in inds:
        f = plt.figure(figsize=(14, 5))
        if comp is None:
            ax = f.add_subplot(1, 1, 1)
        else:
            ax = f.add_subplot(1, 2, 1)
        ax.plot(xs, fams[i, :], label='fam')
        ax.plot(xs, novs[i, :], label='nov')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('firing rate (spks/s)')
        if rec is not None:
            neur_ident = 'run {}, sig {}'.format(rec[i, 0], rec[i, 1])
        else:
            neur_ident = 'neuron {}'.format(i)
        ax.set_title('{} {}'.format(title, neur_ident))
        ax.legend()
        if comp is not None:
            ax2 = f.add_subplot(1, 2, 2)
            ax2.plot(xs, comp_fams[i, :], label='fam')
            ax2.plot(xs, comp_novs[i, :], label='nov')
            ax2.set_xlabel('time (ms)')
    plt.show()

def show_trialtype_spikecourse(data, tts, labels, pretime=-100, posttime=5000,
                               binsize=2, timefield='fixation_off', imgon=0, 
                               imgoff=5000, drunfield='datanum', boots=0):
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
            spktc, nt, bs = get_spikecourse(srun, t, pretime=pretime, 
                                            posttime=posttime, binsize=binsize,
                                            timefield=timefield, boots=boots)
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
    plt.show()
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
    plt.show()
    return f
            
