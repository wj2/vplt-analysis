
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter

from utility import *

def get_fixtimes(data, ttypes, skips=1, stdthr=1, filtwin=40, 
                 ttfield='trial_type', eyefield='eyepos', postthr=None,
                 readdpost=True):
    lens, looks, s_bs, s_es = {}, {}, {}, {}
    for tt in ttypes:
        use = data[data[ttfield] == tt]
        lens[tt] = []
        looks[tt] = []
        s_bs[tt] = []
        s_es[tt] = []
        for t in use:
            eyep = t[eyefield]
            if postthr is not None:
                p_thr = t[postthr]
                eyep = eyep[p_thr:, :]
            sac_bs, sac_es = find_saccades(eyep, skips=skips, stdthr=stdthr, 
                                           filtwin=filtwin)
            t_lens = get_fixation_lengths(sac_bs, sac_es)
            ls = get_fixation_dests(sac_es, eyep, -3., 3.)
            if postthr is not None and readdpost:
                sac_bs = sac_bs + p_thr[0,0]
                sac_es = sac_es + p_thr[0,0]
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

    plt.show(block=False)
    return f_on, f_no, f_fi

def show_sacc_latency_dist(data, ttypes=None, postthr='fixation_off', bins=150, 
                           lenlim=None):
    if ttypes is None:
        ttypes = [7, 8, 9, 10]
    fls = get_first_sacc_latency(data, ttypes, postthr=postthr)
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
    plt.show(block=False)
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
                           postthr='fixation_off', onim=False, firstim=False):
    if tnumrange is not None:
        data = data[np.logical_and(tnumrange[0] <= data[tnumfield],
                                   tnumrange[1] > data[tnumfield])]
    lens, looks, s_bs, s_es = get_fixtimes(data, ttypes, postthr=postthr, 
                                           readdpost=False)
    flooktime = {}
    for tt in s_bs.keys():
        sb = s_bs[tt]
        flooktime[tt] = []
        for i, t in enumerate(sb):
            if len(t) > 0:
                if onim:
                    ft = np.where(np.logical_or(looks[tt][i] == 'l', 
                                                looks[tt][i] == 'r'))[0]
                    if len(ft) > 0:
                        flooktime[tt].append(t[ft[0]])
                elif firstim:
                    if looks[tt][i][0] in ['l', 'r']:
                        flooktime[tt].append(t[0])
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
        c_lens = np.concatenate(lens[tt], axis=1).T
        ax.hist(c_lens, bins=bins, range=(0, cutoffmax), normed=True, 
                histtype='step')
    ax.set_xlabel('fixation time (ms)')
    ax.set_ylabel('count')
    plt.show(block=False)

def get_fixtime_by_looknum(lens, maxlooks=15):
    nlens = filter(lambda x: len(x) >= maxlooks, lens)
    all_lens = np.zeros((len(nlens), maxlooks))
    for i, l in enumerate(nlens):
        all_lens[i, :] = l[:maxlooks]
    m_lens = np.mean(all_lens, axis=0)
    s_lens = np.std(all_lens, axis=0) / np.sqrt(all_lens.shape[0] - 1)
    return m_lens, s_lens, all_lens

def show_fixtime_by_looknum(lens, maxlooks=15):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    xs = np.arange(1, maxlooks + 1)
    for tt in lens.keys():
        m_lens, s_lens, all_lens = get_fixtime_by_looknum(lens[tt], 
                                                          maxlooks=maxlooks)
        ax.errorbar(xs, m_lens, yerr=s_lens, label=tt)
        ax.legend()
    plt.show(block=False)
    return m_lens, s_lens, all_lens

def show_numlooks_dist(lens):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    num_looks = map(lambda x: max(x.shape), lens)
    ax.hist(num_looks, bins=np.arange(np.max(num_looks)+1))
    ax.set_xlabel('number of looks')
    ax.set_ylabel('count')
    plt.show(block=False)

def get_look_probs(looks, n, countchar):
    sumlooks = np.zeros(n)
    f_looks = filter(lambda x: len(x) >= n, looks)
    for l in f_looks:
        a = (l[:n] == countchar).astype(np.int)
        sumlooks = a + sumlooks
    nf = float(len(f_looks))
    return sumlooks / nf, nf

def find_saccades(eyepos, skips=1, stdthr=1, filtwin=None, thr=None, 
                  fixthr=10, vthr=1.5):
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
    if sac_es[0] < sac_bs[0]:
        sac_es = sac_es[1:]
    if sac_es[-1] < sac_bs[-1]:
        sac_bs = sac_bs[:-1]
    sac_bs, sac_es = filter_short_fixes(sac_bs, sac_es, thr=fixthr)
    assert len(sac_bs) == len(sac_es)
    return sac_bs, sac_es

def get_fixation_lengths(sac_bs, sac_es):
    tbs = np.reshape(sac_bs[1:], (1, -1))
    tes = np.reshape(sac_es[:-1], (1, -1))
    st = np.concatenate((tes, tbs), axis=0)
    f_lens = np.diff(st, axis=0)
    return f_lens[0, :]

def filter_short_fixes(sac_bs, sac_es, thr=10):
    lens = get_fixation_lengths(sac_bs, sac_es)
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

def show_compare_fix_lens(lens, looks, n=None, keyp=None, bins=200):
    n_looks, n_trls = compare_fix_lens(lens, looks, n=n, keyp=keyp, 
                                       famornov='nov')
    f_looks, f_trls = compare_fix_lens(lens, looks, n=n, keyp=keyp,
                                       famornov='fam')
    
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.hist(n_looks, bins=bins, normed=True, histtype='step', label='nov')
    ax.hist(f_looks, bins=bins, normed=True, histtype='step', label='fam')
    ax.legend()
    plt.show(block=False)
    return n_looks, n_trls, f_looks, f_trls

def get_fixation_dests(sac_es, eyes, xleft, xright, yleft=0, yright=0, 
                       heideg=5.5, widdeg=5.5):
    sac_es_int = sac_es.astype(np.int)
    endlocs = eyes[sac_es_int, :]
    looks = np.ones(sac_es.shape, dtype='S1')
    looks[full_box_locs(endlocs, xleft, yleft, widdeg, heideg)] = 'l'
    looks[full_box_locs(endlocs, xright, yright, widdeg, heideg)] = 'r'
    looks[looks == '1'] ='o'
    return looks
    
def plot_eyeloc(eyepos, skips=1, stdthr=1, filtwin=40, thr=None, save=False):
    f1 = plt.figure()
    ax1 = f1.add_subplot(1, 1, 1)
    ax1.plot(eyepos[::skips, 0], eyepos[::skips, 1])
    
    f2 = plt.figure()
    ax2 = f2.add_subplot(1, 1, 1)
    ax2.plot(eyepos[::skips, 0], label='x', markersize=2)
    ax2.plot(eyepos[::skips, 1], label='y', markersize=2)    
    s_b, s_e = find_saccades(eyepos, skips=skips, stdthr=stdthr, 
                             filtwin=filtwin, thr=thr)
    ax2.plot(s_b / skips, np.ones(s_b.shape)*5, 'o', label='saccade')
    ax2.plot(np.diff(eyepos[::skips, 0]), label='dx')
    ax2.plot(np.diff(eyepos[::skips, 1]), label='dy')
    ax2.legend(frameon=False)
    if save:
        f2.savefig('x-y-course.pdf')
    plt.show(block=False)

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
