
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as p
import matplotlib.gridspec as gs
from scipy.ndimage.filters import median_filter
import scipy.stats as sts

from general.utility import *

def plot_eyetrace_xy(bhv, useconds, skip=100, coff=(0,0), 
                     ylim=(-12,12), xlim=(-12, 12), 
                     tnumfield='trial_type', eyefield='eyepos',
                     img1_xy='img1_xy', img2_xy='img2_xy', 
                     img_wid='img_wid', img_hei='img_hei',
                     trunc_trial=0, truncfield=None, gradient=False,
                     gradstep=.5, figsize=None, wid=None, hei=None,
                     savename=None, dpi=300, cmap='Blues'):
    coff = np.array(coff)
    loc_s = set()
    coff = np.array(coff)
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(1, 1, 1, aspect='equal')
    if gradient:
        xs = []
        ys = []
    for t in bhv:
        if t[tnumfield] in useconds:
            if truncfield is not None:
                tt1 = t[truncfield]
            else:
                tt1 = 0
            tt2 = tt1 + trunc_trial
            ef = t[eyefield][tt2:, :]
            if gradient:
                xs = xs + list(ef[::skip, 0])
                ys = ys + list(ef[::skip, 1])
            else:
                _ = ax.plot(ef[trunc_trial::skip, 0], 
                            ef[trunc_trial::skip, 1],
                            'o')
            # loc_s.add(tuple(t[img1_xy][:, 0] + coff))
            # loc_s.add(tuple(t[img2_xy][:, 0] + coff))
            if wid is None:
                wid = t[img_wid]
            if hei is None:
                hei = t[img_hei]
    if gradient:
        k = sts.gaussian_kde(np.vstack([xs, ys]))
        xi, yi = np.mgrid[xlim[0]:xlim[1] + gradstep/2.:gradstep, 
                          ylim[0]:ylim[1] + gradstep/2.:gradstep]
        inds = np.vstack([xi.flatten(), yi.flatten()])
        vals = k(inds).reshape(xi.shape)
        pc = ax.pcolormesh(xi, yi, vals, cmap=plt.get_cmap(cmap))
        col_ax = f.add_axes((.91, .12, .02, .78), label='cbarax')
        cb = f.colorbar(pc, cax=col_ax)
        cb.set_ticks([0, .01, .02])
    _ = ax.set_xlim(xlim)
    _ = ax.set_ylim(ylim)
    _ = ax.set_xlabel('visual degrees')
    _ = ax.set_ylabel('visual degrees')
    for c in loc_s:
        ax.add_artist(plt.Rectangle((c[0] - wid/2., c[1] - hei/2.), 
                                    wid, hei, fill=False))
    if savename is not None:
        f.savefig(savename, bbox_inches='tight', dpi=dpi, transparent=True)
    return ax

def get_tm_cond(tm, total, targs, let_to_ind):
    for i, t in enumerate(targs):
        for j, s in enumerate(t[:-1]):
            beg = let_to_ind[s]
            end = let_to_ind[t[j+1]]
            tm[beg, end] += 1
            total[beg] += 1
    return tm, total
    
def get_transition_timecourse(targs, lens, times, bins, leftconds=(10,), 
                              rightconds=(7,)):
    tmtc = np.zeros((len(bins) - 1, 3, 3))
    for i, b in enumerate(bins[:-1]):
        tmtc[i] = get_transition_prob(targs, lens, leftconds=leftconds,
                                      rightconds=rightconds, times=times,
                                      time_lb=b, time_hb=bins[i+1])
    return tmtc
                            

def get_transition_prob(targs, lens, leftconds=(10,), rightconds=(7,),
                        time_lb=0, time_hb=10000, times=None):
    trans_mat = np.zeros((3, 3))    
    totals = {0:0, 1:0, 2:0}
    for lc in leftconds:
        let_to_ind = {'l':0, 'r':1, 'o':2}
        tgs = targs[lc]
        if times is not None:
            tms = times[lc]
            tgs = [tg[np.logical_and(tms[j] >= time_lb, tms[j] < time_hb)]
                   for j, tg in enumerate(tgs)]
        trans_mat, totals = get_tm_cond(trans_mat, totals, tgs, let_to_ind)
    for rc in rightconds:
        let_to_ind = {'l':1, 'r':0, 'o':2}
        tgs = targs[rc]
        if times is not None:
            tms = times[rc]
            tgs = [tg[np.logical_and(tms[j] >= time_lb, tms[j] < time_hb)]
                   for j, tg in enumerate(tgs)]
        trans_mat, totals = get_tm_cond(trans_mat, totals, tgs, let_to_ind)
    trans_mat[0, :] = trans_mat[0, :]/float(totals[0])
    trans_mat[1, :] = trans_mat[1, :]/float(totals[1])
    trans_mat[2, :] = trans_mat[2, :]/float(totals[2])
    return trans_mat
            
def get_fixtimes(data, ttypes, skips=1, stdthr=None, filtwin=40, 
                 ttfield='trial_type', eyefield='eyepos', postthr=None,
                 readdpost=True, vthr=None, thr=.1, lc=(-3., 0), 
                 rc=(3., 0), wid=5.5, hei=5.5, use_bhv_img_params=False,
                 centoffset=(0, 0)):
    lens, looks, s_bs, s_es = {}, {}, {}, {}
    for tt in ttypes:
        use = data[data[ttfield] == tt]
        lens[tt] = []
        looks[tt] = []
        s_bs[tt] = []
        s_es[tt] = []
        for t in use:
            if use_bhv_img_params:
                lc = (t['img1_xy'][0] + centoffset[0], 
                      t['img1_xy'][1] + centoffset[1])
                rc = (t['img2_xy'][0] + centoffset[0],
                      t['img2_xy'][1] + centoffset[1])
                wid = t['img_wid']
                hei = t['img_hei']
            eyep = t[eyefield]
            if postthr is not None:
                p_thr = t[postthr]
            out = analyze_eyemove(eyep, lc, rc, 
                                  skips=skips, stdthr=stdthr, filtwin=filtwin,
                                  vthr=vthr, thr=thr, wid=wid, hei=hei, 
                                  postthr=p_thr, readdpost=readdpost)
            sac_bs, sac_es, t_lens, ls = out
            s_bs[tt].append(sac_bs)
            s_es[tt].append(sac_es)
            lens[tt].append(t_lens)
            looks[tt].append(ls)
        s_bs[tt] = np.array(s_bs[tt])
        s_es[tt] = np.array(s_es[tt])
    return lens, looks, s_bs, s_es

def show_sacc_latency_by_tnum(data, tnumrange, tbins):
    f_on = sacc_latency_by_tnum(data, tnumrange, tbins, onim=True)
    f_no = sacc_latency_by_tnum(data, tnumrange, tbins, onim=False)
    f_fi = sacc_latency_by_tnum(data, tnumrange, tbins, onim=False, 
                                firstim=True)

    m = lambda x: np.mean(x)
    sem = lambda x: np.std(x) / np.sqrt(len(x) - 1)

    f_on_m = map(m, f_on)
    f_on_s = map(sem, f_on)

    f_no_m = map(m, f_no)
    f_no_s = map(sem, f_no)

    f_fi_m = map(m, f_fi)
    f_fi_s = map(sem, f_fi)

    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    xs = np.linspace(tnumrange[0], tnumrange[1], tbins + 1)
    xs = xs[:-1] + (np.diff(xs)/2.)
    
    ax.errorbar(xs, f_no_m, yerr=f_no_s, label='first saccade')
    ax.errorbar(xs, f_fi_m, yerr=f_fi_s, label='first saccade if img')
    ax.errorbar(xs, f_on_m, yerr=f_on_s, label='first saccade onto img')
    ax.set_ylabel('ms from fixation off')
    ax.set_xlabel('exp avg trial number')
    ax.set_title('first saccade latencies')
    ax.legend()

    f_fi_l = np.array(map(len, f_fi), dtype=np.float)
    f_no_l = np.array(map(len, f_no), dtype=np.float)
    problook = f_fi_l / f_no_l
    f2 = plt.figure()
    ax2 = f2.add_subplot(1, 1 ,1)
    ax2.plot(xs, problook)
    ax2.set_ylabel('probability looked at img first')
    ax2.set_xlabel('exp avg trial number')

    plt.show()
    return f_on, f_no, f_fi

def split_early_late(lens, dests, fbegs, early_thr):
    early_lens, early_dests = {}, {}
    late_lens, late_dests = {}, {}
    for k in fbegs.keys():
        elk = []
        edk = []
        llk = []
        ldk = []
        for i, tr in enumerate(fbegs[k]):
            earlbegs = tr[1:] < early_thr
            elk.append(lens[k][i][earlbegs])
            edk.append(dests[k][i][:-1][earlbegs])
            latebegs = np.logical_not(earlbegs)
            llk.append(lens[k][i][latebegs])
            ldk.append(dests[k][i][:-1][latebegs])
        early_lens[k] = elk
        early_dests[k] = edk
        late_lens[k] = llk
        late_dests[k] = ldk
    return early_lens, early_dests, late_lens, late_dests

def show_first_sacc_lat(s_bs_nore, looks_nore, onim=True, first_n=1, 
                        save=False, sidesplit=True, low_filt=50, ax_use=None,
                        bins=50, nov_color=(.5, 0, 0), fam_color=(0, 0, .5),
                        linewidth=3):
    fls = get_first_sacc_latency_nocompute(s_bs_nore, looks_nore, onim=True, 
                                           first_n=1, sidesplit=True)
    first_nov = np.array(fls[7]['r'] + fls[10]['l'])
    first_fam = np.array(fls[7]['l'] + fls[10]['r'])
    first_nov = first_nov[first_nov > low_filt]
    first_fam = first_fam[first_fam > low_filt]
    if ax_use is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    else:
        ax = ax_use
    _ = ax.hist(first_fam, bins=bins, histtype='step', color=fam_color,
                linewidth=linewidth,
                label='familiar (median={})'.format(np.median(first_fam)),
                normed=True)
    _ = ax.hist(first_nov, bins=bins, histtype='step', color=nov_color,
                linewidth=linewidth,
                label='novel (median={})'.format(np.median(first_nov)),
                normed=True)
    k, p = sts.mannwhitneyu(first_fam, first_nov)
    ax.set_xlabel('latency of first saccade (ms)')
    ax.set_ylabel('density')
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save and ax_use is None:
        f.savefig(savename, bbox_inches='tight')
    if ax_use is None:
        plt.show()
    return first_nov, first_fam

def show_sacc_latency_dist(data, ttypes=None, postthr='fixation_off', bins=150, 
                           lenlim=None, onim=False):
    if ttypes is None:
        ttypes = [7, 8, 9, 10]
    fls = get_first_sacc_latency(data, ttypes, postthr=postthr, onim=onim)
    all_fls = collapse_list_dict(fls)
    
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    if lenlim is not None:
        r = (0, lenlim)
    else:
        r = None
    ax.hist(all_fls, bins=bins, range=r)
    ax.set_xlabel('ms from fixation to first saccade')
    ax.set_ylabel('count')
    ax.set_title('first saccade distribution')
    plt.show()
    return all_fls

def sacc_latency_by_tnum(data, tnumrange, tbins, ttypes=None,
                         postthr='fixation_off', onim=True, firstim=False):
    if ttypes is None:
        ttypes = [7, 8, 9, 10]
    binning = np.linspace(tnumrange[0], tnumrange[1], tbins + 1)
    first_ls = []
    for i, be in enumerate(binning[:-1]):
        first_l = get_first_sacc_latency(data, ttypes, 
                                         tnumrange=(be, binning[i+1]), 
                                         postthr=postthr, onim=onim, 
                                         firstim=firstim)
        all_fls = collapse_list_dict(first_l)
        first_ls.append(all_fls)
    return first_ls

def get_first_sacc_latency(data, ttypes, tnumrange=None, tnumfield='trialnum', 
                           postthr='fixation_off', onim=False, firstim=False,
                           sidesplit=False, first_n=-1):
    if tnumrange is not None:
        data = data[np.logical_and(tnumrange[0] <= data[tnumfield],
                                   tnumrange[1] > data[tnumfield])]
    lens, looks, s_bs, s_es = get_fixtimes(data, ttypes, postthr=postthr, 
                                           readdpost=False)
    fls = get_first_sacc_latency_nocompute(s_bs, looks, onim, firstim,
                                           sidesplit, first_n)
    return fls

def get_conf_interval_saccs(fls, leftconds, rightconds, boots=200):
    # construct list
    def get_frac(selection):
        hs, _ = np.histogram(selection, bins=(0, 1, 2, 3))
        frac = hs[0] / float(hs[0] + hs[1])
        return frac

    ps = np.concatenate(([fls[l_cond]['l'] for l_cond in leftconds] +
                         [fls[r_cond]['r'] for r_cond in rightconds]))
    dps = np.concatenate(([fls[l_cond]['r'] for l_cond in leftconds] +
                          [fls[r_cond]['l'] for r_cond in rightconds]))
    ops = np.concatenate(([fls[o_cond]['o'] 
                           for o_cond in rightconds + leftconds]))
    pop = np.array([0]*len(ps) + [1]*len(dps) + [2]*len(ops))
    popfrac = get_frac(pop)
    dist = np.zeros(boots)
    for i in range(boots):
        samp = np.random.choice(pop, len(pop))
        dist[i] = get_frac(samp)
    lowb = np.percentile(dist, 2.5)
    highb = np.percentile(dist, 97.5)
    return popfrac, dist, lowb, highb
                        
def get_first_sacc_latency_nocompute(s_bs, looks, onim=False, firstim=False,
                                     sidesplit=False, first_n=1):
    flooktime = {}
    for tt in s_bs.keys():
        sb = s_bs[tt]
        flooktime[tt] = []
        if sidesplit:
            flooktime[tt] = {}
            flooktime[tt]['l'] = []
            flooktime[tt]['r'] = []
            if not (firstim or onim):
                flooktime[tt]['o'] = []
        for i, t in enumerate(sb):
            if len(t) > 0:
                if onim:
                    if sidesplit:
                        ft_l = np.where(looks[tt][i][:first_n] == b'l')[0]
                        if len(ft_l) > 0:
                            flooktime[tt]['l'].append(t[ft_l][0])
                        ft_r = np.where(looks[tt][i][:first_n] == b'r')[0]
                        if len(ft_r) > 0:
                            flooktime[tt]['r'].append(t[ft_r][0])
                    else:
                        ft = np.where(np.logical_or(looks[tt][i] == b'l', 
                                                    looks[tt][i] == b'r'))[0]
                        if len(ft) > 0:
                            flooktime[tt].append(t[ft[0]])
                elif firstim:
                    if sidesplit:
                        if looks[tt][i][0] == b'l':
                            flooktime[tt]['l'].append(t[0])
                        if looks[tt][i][0] == b'r':
                            flooktime[tt]['r'].append(t[0])
                    else:
                        if looks[tt][i][0] in [b'l', b'r']:
                            flooktime[tt].append(t[0])
                else:
                    if len(looks[tt][i]) > 0:
                        if sidesplit:
                            reg_str = looks[tt][i][0].decode('UTF-8')
                            flooktime[tt][reg_str].append(t[0])
                        else: 
                            flooktime[tt].append(t[0])
    return flooktime

def detect_edge(tcourse, c=20, ep=.2):
    tc_p = np.diff(tcourse)
    tc_pp = np.diff(tc_p)
    tc_p = tc_p[:-1]
    
    c = tc_pp - c*(tc_p**2)
    return c
    
def show_fixtime_dist(lens, cutoffmax=3000, bins=500):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for tt in lens.keys():
        c_lens = np.concatenate(lens[tt]).T
        ax.hist(c_lens, bins=bins, range=(0, cutoffmax), normed=True, 
                histtype='step')
    ax.set_xlabel('fixation time (ms)')
    ax.set_ylabel('count')
    plt.show()

def get_fixtime_by_looknum(lens, looks, uselooks, maxlooks=15):
    nlens = filter(lambda x: len(x) >= maxlooks, lens)
    nlooks = filter(lambda x: len(x) >= maxlooks, looks)
    all_lens = np.zeros((len(nlens), maxlooks))
    all_looks = np.zeros((len(nlens), maxlooks), dtype='str')
    for i, l in enumerate(nlens):
        all_lens[i, :] = l[:maxlooks]
        all_looks[i, :] = nlooks[i][:maxlooks]
    m_lens, s_lens = {}, {}
    for loo in uselooks:
        m_lens[loo] = np.zeros(maxlooks)
        s_lens[loo] = np.zeros(maxlooks)
        for j in xrange(maxlooks):
            corrlooks = all_looks[:, j] == loo
            rellooks = all_lens[corrlooks, j]
            m_lens[loo][j] = np.mean(rellooks)
            s_lens[loo][j] = np.std(rellooks) / np.sqrt(rellooks.shape[0] - 1)
    return m_lens, s_lens, all_lens, all_looks

def show_fixtime_by_looknum(lens, looks, uselooks=['l', 'r'], 
                            usekeys=[7, 10], maxlooks=15):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    xs = np.arange(1, maxlooks + 1)
    for tt in usekeys:
        out = get_fixtime_by_looknum(lens[tt], looks[tt], uselooks, 
                                     maxlooks=maxlooks)
        m_lens, s_lens, all_lens, all_looks = out
        for loo in uselooks:
            ax.errorbar(xs, m_lens[loo], yerr=s_lens[loo], 
                        label='{} {}'.format(tt, loo))
    ax.legend()
    plt.show()
    return m_lens, s_lens, all_lens

def show_numlooks_dist_opposed(looks, key_nov_pair, key_fam_pair, labels=None,
                               ax_use=None, save=False, bins=100, 
                               nov_color=(.5, 0, 0), fam_color=(0, 0, .5),
                               linewidth=3):
    if ax_use is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = ax_use
    nov_looks = []
    for i, k in enumerate(key_nov_pair):
        tt, side = k
        look_trs = looks[tt]
        numlooks = map(lambda x: np.sum(side == x)/float(np.sum(x != 'o')),
                       look_trs)
        nov_looks = nov_looks + numlooks
    fam_looks = []
    for i, k in enumerate(key_fam_pair):
        tt, side = k
        look_trs = looks[tt]
        numlooks = map(lambda x: np.sum(side == x)/float(np.sum(x != 'o')), 
                       look_trs)
        fam_looks = fam_looks + numlooks
    fam_median = np.round(np.median(fam_looks), 2)
    ax.hist(fam_looks, bins=bins, 
            label='familiar (median={})'.format(fam_median), 
            normed=True, histtype='step', color=fam_color,
            linewidth=linewidth)
    nov_median = np.round(np.median(nov_looks), 2)
    ax.hist(nov_looks, bins=bins, 
            label='novel (median={})'.format(nov_median), 
            normed=True, histtype='step', color=nov_color,
            linewidth=linewidth)
    k, p = sts.mannwhitneyu(fam_looks, nov_looks)
    ax.set_xlabel('percent of looks')
    ax.set_ylabel('density')
    ax.legend(frameon=False, loc=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save and ax_use is None:
        f.savefig(savename, bbox_inches='tight')
    if ax_use is None:
        plt.show()
    return nov_looks, fam_looks

def show_numlooks_dist(lens, keyps=None, labels=None, ax_use=None, save=False,
                       color_list=((0, 0, .5), (.5, 0, 0)), linewidth=3):
    if ax_use is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = ax_use
    if keyps is None:
        keyps = lens.keys()
    if labels is None:
        labels = [str(k) for k in keyps]
    nls = []
    for i, k in enumerate(keyps):
        klens = lens[k]
        num_looks = map(lambda x: max(x.shape), klens)
        ax.hist(num_looks, bins=np.arange(np.max(num_looks)+1), 
                label=labels[i]+' (median={})'.format(np.median(num_looks)), 
                normed=True, histtype='step', color=color_list[i],
                linewidth=linewidth)
        nls.append(num_looks)
    k, p = sts.mannwhitneyu(nls[0], nls[1])
    ax.set_xlabel('number of looks')
    ax.set_ylabel('density')
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save and ax_use is None:
        f.savefig(savename, bbox_inches='tight')
    if ax_use is None:
        plt.show()

def get_look_probs(looks, n, countchar):
    sumlooks = np.zeros(n)
    f_looks = filter(lambda x: len(x) >= n, looks)
    for l in f_looks:
        a = (l[:n] == countchar).astype(np.int)
        sumlooks = a + sumlooks
    nf = float(len(f_looks))
    return sumlooks / nf, nf

def find_saccades(eyepos, skips=1, stdthr=None, filtwin=40, thr=.1, 
                  fixthr=10, vthr=None):
    if filtwin is not None:
        eyep = np.zeros(eyepos.shape)
        eyep[:, 0] = median_filter(eyepos[:, 0], filtwin)
        eyep[:, 1] = median_filter(eyepos[:, 1], filtwin)
        eyepos = eyep
    eyederiv = np.diff(eyepos[::skips, :], axis=0)
    abseyed = np.abs(eyederiv)
    if thr is None:
        thr = stdthr*np.std(abseyed[abseyed < vthr])
    saccs = np.logical_or(abseyed[:, 0] > thr, abseyed[:, 1] > thr)
    saccs = saccs.astype(np.int)
    saccs_d = np.diff(saccs)
    sac_bs = np.where(saccs_d == 1)[0]*skips
    sac_es = np.where(saccs_d == -1)[0]*skips
    if len(sac_es) > 0 and len(sac_bs) > 0:
        if sac_es[0] < sac_bs[0]:
            sac_es = sac_es[1:]
        if sac_es[-1] < sac_bs[-1]:
            sac_bs = sac_bs[:-1]
        sac_bs, sac_es = filter_short_fixes(sac_bs, sac_es, thr=fixthr)
    else:
        sac_es = np.array([])
        sac_bs = np.array([])
    assert len(sac_bs) == len(sac_es)
    return sac_bs, sac_es

def analyze_eyemove(eyep, lc, rc, skips=1, stdthr=None, filtwin=40,
                   vthr=None, thr=.1, wid=5.5, hei=5.5, postthr=None,
                    readdpost=True, fixthr=10):
    if (postthr is not None) and (not np.isnan(postthr)):
        eyep = eyep[postthr:, :]
    sac_bs, sac_es = find_saccades(eyep, skips=skips, stdthr=stdthr, 
                                   filtwin=filtwin, vthr=vthr, thr=thr)
    t_lens = get_fixation_lengths(sac_bs, sac_es)
    ls = get_fixation_dests(sac_es, eyep, lc, rc, heideg=hei, 
                            widdeg=wid)
    if postthr is not None and readdpost:
        sac_bs = sac_bs + postthr[0,0]
        sac_es = sac_es + postthr[0,0]
    return sac_bs, sac_es, t_lens, ls

def get_fixation_lengths(sac_bs, sac_es):
    tbs = np.reshape(sac_bs[1:], (1, -1))
    tes = np.reshape(sac_es[:-1], (1, -1))
    st = np.concatenate((tes, tbs), axis=0)
    f_lens = np.diff(st, axis=0)
    return f_lens[0, :]

def filter_short_fixes(sac_bs, sac_es, thr=10):
    lens = get_fixation_lengths(sac_bs, sac_es)
    lens = np.concatenate((lens, (thr+1,)), axis=0)
    f_bs = sac_bs[lens > thr]
    f_es = sac_es[lens > thr]
    return f_bs, f_es

def full_box_locs(locs, x, y, xw, yw):
    return box_locs(locs[:, 0], x, xw) & box_locs(locs[:, 1], y, yw)

def compare_fix_lens(lens, looks, n=None, keyp=None, famornov='nov'):
    if keyp is None:
        if 'nov' == famornov:
            keyp = {7:('r',), 10:('l',)}
        elif 'fam' == famornov:
            keyp = {7:('l',), 10:('r',)}
    all_lens = []
    num_trials = 0
    for k in keyp.keys():
        le = lens[k]
        lo = looks[k]
        num_trials = num_trials + len(le)*len(keyp[k])
        for i, t in enumerate(le):
            look = lo[i][:-1]
            if n is not None:
                t = t[n[0]:n[1]]
                look = look[n[0]:n[1]]
            for x in keyp[k]:
                rellooks = t[look == x]
                all_lens = all_lens + list(rellooks)
    return all_lens, num_trials

def show_compare_fix_lens(lens, looks, n=None, keyp=None, bins=200, save=False,
                          savename='fixation_time_distrib.pdf', ax_use=None,
                          nov_color=(.5, 0, 0), fam_color=(0, 0, .5), 
                          linewidth=3):
    n_looks, n_trls = compare_fix_lens(lens, looks, n=n, keyp=keyp, 
                                       famornov='nov')
    f_looks, f_trls = compare_fix_lens(lens, looks, n=n, keyp=keyp,
                                       famornov='fam')
    if ax_use is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    else:
        ax = ax_use
    n1, _, _ = ax.hist(f_looks, bins=bins, normed=True, histtype='step', 
                       color=fam_color, linewidth=linewidth,
                       label='familiar (median={})'.format(np.median(f_looks)))
    n2, _, _ = ax.hist(n_looks, bins=bins, normed=True, histtype='step', 
                       color=nov_color, linewidth=linewidth,
                       label='novel (median={})'.format(np.median(n_looks)))
    ax.set_ylim([0, np.max([np.max(n1), np.max(n2)]) + .0005])
    ax.legend(frameon=False)
    ax.set_xlabel('fixation time (ms)')
    ax.set_ylabel('density')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save and ax_use is None:
        f.savefig(savename, bbox_inches='tight')
    if ax_use is None:
        plt.show()
    return n_looks, n_trls, f_looks, f_trls

def get_fixation_dests(sac_es, eyes, pleft, pright, 
                       heideg=5.5, widdeg=5.5):
    xleft, yleft = pleft
    xright, yright = pright
    sac_es_int = sac_es.astype(np.int)
    endlocs = eyes[sac_es_int, :]
    looks = np.ones(sac_es.shape, dtype='S1')
    looks[full_box_locs(endlocs, xleft, yleft, widdeg, heideg)] = 'l'
    looks[full_box_locs(endlocs, xright, yright, widdeg, heideg)] = 'r'
    looks[looks == b'1'] = 'o'
    return looks

grey_color = (.4, .4, .4)
x_trace_color = (.3, .3, .6)
y_trace_color = (.2, .3, .3)
    
def plot_eyeloc(eyepos, skips=1, stdthr=1, filtwin=40, thr=None, save=False,
                rects=(), stimwid=0, stimhei=0, cols=(), labels=(), nov_ind=1,
                fam_ind=0, comb_color=(.5, 0, 0), barwid=.7, figsize=(10,10),
                show_saccades=False, savename='single_trial_example.pdf'):
    f1 = plt.figure(figsize=figsize)
    spec = gs.GridSpec(3, 3)
    ax1 = f1.add_subplot(spec[1:, :2], aspect='equal')
    ax3 = f1.add_subplot(spec[1:, 2:])
    t_recs = np.zeros(len(rects))
    barlocs = np.arange(1, len(rects) + 2)
    for i, r in enumerate(rects):
        x, y = r
        ax1.add_patch(p.Rectangle((x - stimwid/2.,
                                   y - stimhei/2.),
                                  stimwid,
                                  stimhei, fill=False,
                                  edgecolor=cols[i]))
        t_recs[i] = np.sum(full_box_locs(eyepos, r[0], r[1], stimwid, 
                                         stimhei))
        ax3.barh(barlocs[-i-1], t_recs[i]/float(len(eyepos)), height=barwid, 
                 color=cols[i], hatch='//')
    t_recs_norm = t_recs/float(len(eyepos))
    ax3.barh(barlocs[0], t_recs_norm[nov_ind] - t_recs_norm[fam_ind], height=barwid,
             color=cols[0])
    ax3.set_xticks([.1, .3, .5])
    ax3.xaxis.set_ticks_position('top')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_yticks(barlocs + barwid/2.)
    all_labels = list(labels + ('combined (n - f)',))[::-1]
    ax3.set_yticklabels(all_labels, rotation='vertical')
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xlabel('normalized looking time')


    ax1.plot(eyepos[::skips, 0], eyepos[::skips, 1], color=grey_color)
    ax1.yaxis.set_ticks_position('right')
    ax1.xaxis.set_ticks_position('top')
    ax1.set_xlabel('visual degrees')
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel('visual degrees')
    ax1.yaxis.set_label_position('right')
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylim([-10, rects[0][1] + stimhei/2. + .5])
    
    ax2 = f1.add_subplot(spec[0, :])
    ax2.plot(eyepos[::skips, 0], label='x', color=x_trace_color, markersize=2)
    ax2.plot(eyepos[::skips, 1], label='y', color=y_trace_color, markersize=2)    
    s_b, s_e = find_saccades(eyepos, skips=skips, stdthr=stdthr, 
                             filtwin=filtwin, thr=thr)
    if show_saccades:
        ax2.plot(s_b / skips, np.ones(s_b.shape)*5, 'o', label='saccade')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('visual degrees')
    ax2.legend(frameon=False, loc=3)
    f1.tight_layout()
    if save:
        f1.savefig(savename, bbox_inches='tight')
    plt.show()

def get_starts(using, alllooks, n):
    sts = np.array(map(lambda x: x[1][n], alllooks[using]))
    return np.reshape(sts, (-1, 1))

def box_locs(locs, cent, wid):
    above_l = locs >= cent - wid/2.
    below_r = locs <= cent + wid/2.
    return above_l & below_r

def comb_looks(sides, ons, offs, combthr=50):
    num_sides = np.where(sides == 'l', 1, 0)
    trans = np.diff(num_sides)
    cons = np.reshape(ons[1:], (1, -1))
    coffs = np.reshape(offs[:-1], (1, -1))
    offons = np.concatenate((coffs, cons), axis=0)
    gaplen = np.diff(offons, axis=0)[0, :]
    new_sides, new_ons, new_offs = [], [], []
    c_on = ons[0]
    for i, on in enumerate(ons[1:]):
        if trans[i] == 0 and gaplen[i] < combthr:
            if c_on is None:
                c_on = ons[i]
        else:
            if c_on is not None:
                new_sides.append(sides[i])
                new_ons.append(c_on)
                new_offs.append(offs[i])
                c_on = None
            else:
                new_sides.append(sides[i])
                new_ons.append(ons[i])
                new_offs.append(offs[i])
    if c_on is not None:
        new_sides.append(sides[-1])
        new_ons.append(c_on)
        new_offs.append(offs[-1])
    else:
        new_sides.append(sides[-1])
        new_ons.append(ons[-1])
        new_offs.append(offs[-1])
    return np.array(new_sides), np.array(new_ons), np.array(new_offs)

def thresh_and_combine_looks(sides, ons, offs, thresh=250, combthr=50):
    if len(sides) > 0:
        s, on, off = comb_looks(sides, ons, offs, combthr=combthr)
        s, on, off = thresh_looks(s, on, off)
    else:
        s = sides
        on = ons
        off = offs
    return s, on, off

def thresh_looks(sides, ons, offs, thresh=250):
    cons = np.reshape(ons, (1, -1))
    coffs = np.reshape(offs, (1, -1))
    onoffs = np.concatenate((cons, coffs), axis=0)
    doos = np.diff(onoffs, axis=0)[0, :]
    tsides = sides[doos > thresh]
    tons = ons[doos > thresh]
    toffs = offs[doos > thresh]
    return tsides, tons, toffs

def get_look_thresh(eyeloc, stimoff, thresh=250, combthr=50, heideg=5.5, 
                    widdeg=5.5, offset=None):
    wim, ons, offs = look_sequence(eyeloc, -stimoff, stimoff, heideg=heideg,
                                   widdeg=widdeg)
    if offset is not None:
        use = ons >= offset
        wim = wim[use]
        ons = ons[use]
        offs = offs[use]
    return thresh_and_combine_looks(wim, ons, offs, thresh=thresh, 
                                    combthr=combthr)

def get_lookimg(eyeloc, xstimoff, ystimoff, heideg=5.5, widdeg=5.5):
    xloc = eyeloc[:, 0]
    yloc = eyeloc[:, 1]
    xin = box_locs(xloc, xstimoff, widdeg)
    yin = box_locs(yloc, ystimoff, heideg)
    onim = xin & yin
    onim = np.where(onim, 1, 0)
    onim = np.concatenate(([0], onim, [0]), axis=0)
    changeim = np.diff(onim)
    ons = np.where(changeim == 1)[0]
    offs = np.where(changeim == -1)[0]
    return onim, ons, offs

def look_sequence(eyeloc, xleftloc, xrightloc, yleftloc=0, yrightloc=0, 
                  heideg=5.5, widdeg=5.5):
    look_l, on_l, off_l = get_lookimg(eyeloc, xleftloc, yleftloc, heideg, 
                                      widdeg)
    look_r, on_r, off_r = get_lookimg(eyeloc, xrightloc, yrightloc, heideg,
                                      widdeg)
    ls = np.ones(len(on_l), dtype='S1')
    ls[:] = 'l'
    rs = np.ones(len(on_r), dtype='S1')
    rs[:] = 'r'
    all_strs = np.concatenate((ls, rs))
    all_ons = np.concatenate((on_l, on_r))
    all_offs = np.concatenate((off_l, off_r))
    sort_inds = np.argsort(all_ons)
    sort_strs = all_strs[sort_inds]
    sort_ons = all_ons[sort_inds]
    sort_offs = all_offs[sort_inds]
    return sort_strs, sort_ons, sort_offs
