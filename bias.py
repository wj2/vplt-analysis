
import numpy as np
import matplotlib.pyplot as plt
from general.utility import *
import general.plotting as gpl
import pref_looking.eyes as es

nn_cond = 9
ff_cond = 8
fn_cond = 7
nf_cond = 10

condnum_field = 'trial_type'
left_views_field = 'leftviews'
right_views_field = 'rightviews'
eyemove_flag = 'fixation_acquired'
eye_field = 'eyepos'
error_field = 'TrialError'

def quantify_bias_by_separation(bhv, wid=3, hei=3, centoffset=(0, 0), 
                                cond_fn=fn_cond, cond_nf=nf_cond, 
                                cond_ff=ff_cond, cond_nn=nn_cond, sacc_vthr=.1, 
                                postthr='fixation_off', bias_win=50, 
                                bias_step=20, tlen=3000, fixtime=500, 
                                readdpost=False, first_sacc_thr=0, 
                                subplot_size=(4, 4), filt_err=True,
                                corr_trial=0, err_field='TrialError'):
    if filt_err:
        bhv = bhv[bhv[err_field] == corr_trial]
    bhv_plt = get_only_conds(bhv, [cond_nf, cond_nn, cond_ff, cond_fn])
    locs = [(x[0][0], x[1][0], bhv_plt['img2_xy'][i][0][0], 
             bhv_plt['img2_xy'][i][1][0])
            for i, x in enumerate(bhv_plt['img1_xy'])]
    seps = np.array([compute_angular_separation(x[:2], x[2:]) for x in locs])
    seps = np.round(seps, 1)
    unique_seps = set(seps)
    u_seps = np.array(list(unique_seps))
    s_seps = np.sort(u_seps)
    f = plt.figure(figsize=(subplot_size[0], 
                            (subplot_size[1]*len(s_seps) 
                             + 2*subplot_size[1])))
    n_locs = len(s_seps)
    ax_bias = f.add_subplot(n_locs + 2, 1, n_locs + 2)
    ax_prop = f.add_subplot(n_locs + 2, 1, n_locs + 1)
    props = np.zeros_like(s_seps)
    props_e = np.zeros((2, props.shape[0]))
    for i, sep in enumerate(s_seps):
        l1 = None
        l2 = None
        mask = seps == sep
        l_bhv = bhv_plt[mask]
        if i == 0:
            ax_dist_l = f.add_subplot(n_locs + 2, 1, i+1)
        else:
            ax_dist_l = f.add_subplot(n_locs + 2, 1, i+1, sharex=ax_dist_l)
        out = quantify_bias(l_bhv, l1, l2, wid=wid, hei=hei, 
                            centoffset=centoffset, 
                            use_bhv_img_params=True, 
                            sacc_vthr=sacc_vthr, postthr=postthr, 
                            bias_win=bias_win, bias_step=bias_step, tlen=tlen,
                            fixtime=fixtime, readdpost=readdpost, 
                            first_sacc_thr=first_sacc_thr, ax_bias=ax_bias, 
                            cond_fn=cond_fn, cond_ff=cond_ff, 
                            cond_nf=cond_nf, cond_nn=cond_nn,
                            ax_dist=ax_dist_l, sep=sep)
        props[i] = out[0][0]
        props_e[0, i] = out[0][2]
        props_e[1, i] = out[0][3]
    ax_bias.legend(frameon=False)
    ax_prop.errorbar(s_seps, props, np.abs(props_e - props))

def quantify_bias(bhv, lc, rc, wid=3, hei=3, centoffset=(0, 0), 
                  use_bhv_img_params=True, cond_fn=fn_cond, cond_nf=nf_cond, 
                  cond_ff=ff_cond, cond_nn=nn_cond, sacc_vthr=.1, 
                  postthr='fixation_off', bias_win=50, bias_step=20, tlen=3000,
                  fixtime=500, readdpost=False, first_sacc_thr=0, ax_bias=None,
                  ax_dist=None, sacc_boots=500, sep=None):
    ls, ts, begs, ends = es.get_fixtimes(bhv, [cond_fn, cond_ff, cond_nn, 
                                               cond_nf],
                                         postthr=postthr, thr=sacc_vthr, 
                                         readdpost=readdpost, lc=lc, rc=rc, 
                                         wid=wid, hei=hei, 
                                         use_bhv_img_params=use_bhv_img_params,
                                         centoffset=centoffset)
    fls = es.get_first_sacc_latency_nocompute(begs, ts, onim=False, first_n=1,
                                              sidesplit=True)
    first_nov = np.array(fls[cond_fn]['r'] + fls[cond_nf]['l'])
    first_fam = np.array(fls[cond_fn]['l'] + fls[cond_nf]['r'])
    frac, dist, lb, hb = es.get_conf_interval_saccs(fls, [cond_nf], [cond_fn],
                                                    sacc_boots)
    if ax_bias is None or ax_dist is None:
        f = plt.figure()
        ax_dist = f.add_subplot(2, 1, 1)
        ax_bias = f.add_subplot(2, 1, 2)
    ax_dist.hist(first_nov, histtype='step', label='novel', normed=True)
    ax_dist.hist(first_fam, histtype='step', label='familiar', normed=True)
    ax_dist.legend()
    first_sacc_nov_prop = frac
    fsnp = round(first_sacc_nov_prop*100, 1)
    title = '{}% first saccades to novel'.format(fsnp)
    if sep is not None:
        prefix_title = 'sep: {}; '.format(sep)
        title = prefix_title + title
    ax_dist.set_title(title)
    
    p, e, d, pxs = get_bias_tc(bhv, winsize=bias_win, winstep=bias_step, 
                               tlen=tlen, fix_time=fixtime, lc=lc, rc=rc,
                               wid=wid, hei=hei, rightconds=(cond_fn,),
                               leftconds=(cond_nf,), 
                               use_bhv_img_params=use_bhv_img_params,
                               centoffset=centoffset)
    gpl.plot_trace_werr(pxs, d, ax=ax_bias, label=prefix_title[:-2])
    print(max(np.mean(d, axis=0))*(1000/bias_win))
    return (frac, dist, lb, hb), (p, e, pxs)

def produce_eyes_plot_nums(resorted=True, save=False, fname=None):
    if resorted:
        pattern = 'bootsy-bhvit-rs-run[0-9]*.mat'
    else:
        pattern = 'bootsy-bhvit-run[0-9]*.mat'
    data = load_separate(['./data/'], pattern)
    data = data[data[error_field] == 0]
    datavplt = get_only_vplt(data)
    lc, xs = get_bias_tc(datavplt, winsize=1, winstep=1)
    lens, looks, s_bs, s_es = es.get_fixtimes(datavplt, [7, 8, 9, 10])
    for i, tt in enumerate(lens.keys()):
        cond_lens = np.concatenate(lens[tt], axis=1)
        if i == 0:
            all_lens = cond_lens
        else:
            all_lens = np.concatenate((all_lens, cond_lens), axis=0)
    plot_looks(lc, xs)
    if save and fname is not None:
        np.savez(open(fname, 'wb'), xs=xs, look_course=lc, fix_lens=all_lens)
    return lc, xs, all_lens

def plot_looks(look_courses, xs):
    mean_lc = look_courses.mean(0)
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    for i in xrange(mean_lc.shape[1]):
        ax.plot(xs, mean_lc[:, i], label='{}'.format(i))
    ax.legend()
    plt.show()
    return f

def plot_pref(bias, xs, err=None, ax_g=None, label=None):
    if ax_g is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    else:
        ax = ax_g
    if err is None:
        ax.plot(xs, bias, label=label)
    else:
        ax.errorbar(xs, bias, err, label=label)
    if ax_g is None:
        plt.show()

def get_bias_tc(data, leftconds=(10,), rightconds=(7,), pref_range=None,
                dispref_range=None, winsize=50, winstep=20, fix_time=500,
                tlen=5000, tbt=True, wid=5.5, hei=5.5, lc=(-3, 0), rc=(3, 0),
                use_bhv_img_params=False, centoffset=(0,0), get_sacc=False,
                boot_sacc=200):
    out = get_lookcourse(data, leftconds, rightconds, pref_range,
                         dispref_range, fix_time, tlen, hei, wid, lc, rc,
                         use_bhv_img_params=use_bhv_img_params,
                         centoffset=centoffset, get_sacc=get_sacc,
                         boot_sacc=boot_sacc)
    if get_sacc:
        lc, xs, sb, lb, hb, dist = out
    else:
        lc, xs = out
    if tbt:
        p, e, d, p_xs = pref_dispref_bias_tbt(lc, xs, winsize, winstep)
    else:
        p, p_xs = pref_dispref_bias(lc, xs, winsize, winstep)
        e = None
        d = None
    if get_sacc:
        ret = p, e, d, p_xs, sb, lb, hb, dist
    else:
        ret = p, e, d, p_xs
    return ret

def get_lookcourse(data, leftconds=(10,), rightconds=(7,), pref_range=None,
                   dispref_range=None, fix_time=500, tlen=5000, hei=5.5,
                   wid=5.5, lc=(-3, 0), rc=(3, 0), use_bhv_img_params=False,
                   centoffset=(0,0), get_sacc=False, boot_sacc=200):
    left_ts = ()
    for i, cond in enumerate(leftconds):
        condmask = data[condnum_field] == cond
        ts = data[condmask]
        keep_ts = get_fit_trials(ts, left_views_field, right_views_field,
                                 pref_range, dispref_range)
        if i == 0:
            left_ts = keep_ts
        else:
            left_ts = np.concatenate((left_ts, keep_ts))
    right_ts = ()
    for i, cond in enumerate(rightconds):
        condmask = data[condnum_field] == cond
        ts = data[condmask]
        keep_ts = get_fit_trials(ts, right_views_field, left_views_field,
                                 pref_range, dispref_range)
        if i == 0:
            right_ts = keep_ts
        else:
            right_ts = np.concatenate((right_ts, keep_ts))
    if get_sacc:
        _, tsl, begsl, _ = es.get_fixtimes(left_ts, leftconds, 
                                           postthr='fixation_off', thr=.1,
                                           readdpost=False, lc=lc, rc=rc, 
                                           wid=wid, hei=hei,
                                           use_bhv_img_params=True)
        fls = es.get_first_sacc_latency_nocompute(begsl, tsl, sidesplit=True)
        _, tsr, begsr, _ = es.get_fixtimes(right_ts, rightconds, 
                                           postthr='fixation_off', thr=.1,
                                           readdpost=False, lc=lc, rc=rc, 
                                           wid=wid, hei=hei, 
                                           use_bhv_img_params=True)
        frs= es.get_first_sacc_latency_nocompute(begsr, tsr, sidesplit=True)
        frs.update(fls)
        sacc_prop, dist, lb, hb = es.get_conf_interval_saccs(frs, leftconds,
                                                             rightconds,
                                                             boots=boot_sacc)
    look_course_l = get_look_img(left_ts, 1, 2, 0, fix_time=fix_time, 
                                 tlen=tlen, left_cent=lc, right_cent=rc, 
                                 img_wid=wid, img_hei=hei, 
                                 use_bhv_img_params=use_bhv_img_params,
                                 centoffset=centoffset)
    look_course_r = get_look_img(right_ts, 2, 1, 0, fix_time=fix_time,
                                 tlen=tlen, left_cent=lc, right_cent=rc, 
                                 img_wid=wid, img_hei=hei,
                                 use_bhv_img_params=use_bhv_img_params,
                                 centoffset=centoffset)
    look_courses = np.concatenate((look_course_l, look_course_r), axis=0)
    xs = np.arange(-fix_time, tlen)
    if get_sacc:
        ret = look_courses, xs, sacc_prop, lb, hb, dist
    else:
        ret = look_courses, xs
    return ret

def pref_dispref_bias_tbt(look_courses, xs, winsize=50., winstep=1.,
                          pref_ind=1, dispref_ind=2):
    win = np.ones(winsize)
    con_bias = np.zeros((look_courses.shape[0], 
                         look_courses.shape[1] - winsize + 1))
    if winsize > 1:
        for i, t in enumerate(look_courses):
            ptc = np.convolve(t[:, pref_ind], win, 'valid')
            dtc = np.convolve(t[:, dispref_ind], win, 'valid')
            con_bias[i] = (ptc - dtc)/winsize
        xs = np.convolve(xs, win, 'valid') / winsize
    else:
        con_bias = (look_courses[:, :, pref_ind] 
                    - look_courses[:, :, dispref_ind])
    bias_m = np.mean(con_bias, 0)
    bias_sem = np.std(con_bias, 0)/np.sqrt(look_courses.shape[0] - 1)
    bias_m_sk = bias_m[::winstep]
    bias_sem_sk = bias_sem[::winstep]
    xs_sk = xs[::winstep]
    thinned_bias = con_bias[:, ::winstep]
    return bias_m_sk, bias_sem_sk, thinned_bias, xs_sk

def pref_dispref_bias(look_courses, xs, winsize=50., winstep=1.,
                      pref_ind=1, dispref_ind=2):
    win = np.ones(winsize)
    mean_lc = np.mean(look_courses, axis=0)
    bias_tc = mean_lc[:, pref_ind] - mean_lc[:, dispref_ind]
    if winsize > 1:
        smooth_tc = np.convolve(bias_tc, win, 'valid') / winsize
        smooth_xs = np.convolve(xs, win, 'valid') / winsize
    else:
        smooth_tc = bias_tc
        smooth_xs = xs
    skipped = smooth_tc[::int(winstep)]
    skipped_xs = smooth_xs[::int(winstep)]
    return skipped, skipped_xs
    
def pref_dispref_distrib(look_courses, xs, winsize=50., winstep=1.):
    win = np.ones(winsize)
    bias_tc = look_courses[:, :, 0] - look_courses[:, :, 1]
    n = bias_tc.shape[1]
    bias_tc_dist = np.zeros((bias_tc.shape[0], n - winsize + 1))
    if winsize > 1:
        for i, trial in enumerate(bias_tc):
            bias_tc_dist[i, :] = np.convolve(trial, win, 'valid') / winsize
        smooth_xs = np.convolve(xs, win, 'valid') / winsize
    else:
        bias_tc_dist = bias_tc
        smooth_xs = xs
    skipped = bias_tc_dist[:, ::winstep]
    skipped_xs = smooth_xs[::winstep]
    return skipped, skipped_xs

def get_look_img(trials, left_ind, right_ind, off_ind, left_cent=(-3, 0), 
                 right_cent=(3, 0), img_wid=4, img_hei=4, fix_time=500, 
                 tlen=5000, use_bhv_img_params=False, centoffset=(0,0)):
    all_eyes = np.zeros((len(trials), fix_time+tlen, 3))
    for i, t in enumerate(trials):
        if use_bhv_img_params:
            left_cent = (t['img1_xy'][0] + centoffset[0], 
                         t['img1_xy'][1] + centoffset[1])
            right_cent = (t['img2_xy'][0] + centoffset[0],
                          t['img2_xy'][1] + centoffset[1])
            img_wid = t['img_wid']
            img_hei = t['img_hei']
        start_ind = t[eyemove_flag]
        end_ind = start_ind + fix_time + tlen
        eyeper = t[eye_field][start_ind:end_ind, :]
        if end_ind > t[eye_field].shape[0]:
            all_eyes[i, :, :] = np.nan
        else:
            all_eyes[i, :, :] = categ_eyepos(eyeper, left_ind, right_ind,
                                             off_ind, left_cent, right_cent,
                                             img_wid, img_hei)
    return all_eyes

def categ_eyepos(eyes, li, ri, oi, left_cent, right_cent, wid, hei):
    cats = np.zeros((eyes.shape[0], 3))
    
    in_left = in_box(eyes, left_cent, wid, hei)
    cats[:, li] = in_left
    in_right = in_box(eyes, right_cent, wid, hei)
    cats[:, ri] = in_right
    cats[:, oi] = np.logical_not(np.logical_or(in_right, in_left))
    return cats

def in_box(eyes, cent, wid, hei):
    left_edge = cent[0] - wid/2.
    right_edge = cent[0] + wid/2.
    top_edge = cent[1] + wid/2.
    bottom_edge = cent[1] - wid/2.
    in_y = np.logical_and(eyes[:, 1] < top_edge, eyes[:, 1] > bottom_edge)
    in_x = np.logical_and(eyes[:, 0] < right_edge, eyes[:, 0] > left_edge)
    in_both = np.logical_and(in_y, in_x)
    return in_both

def get_fit_trials(ts, prefv_field, disprefv_field, pref_range, dispref_range):
    if pref_range is not None:
        prefcond = np.logical_and(ts[prefv_field] >= pref_range[0],
                                  ts[prefv_field] < pref_range[1])
    else:
        prefcond = np.ones(ts.shape) == 1
    if dispref_range is not None:
        disprefcond = np.logical_and((ts[disprefv_field] 
                                      >= dispref_range[0]),
                                     (ts[disprefv_field]
                                      < dispref_range[1]))
    else:
        disprefcond = np.ones(ts.shape) == 1
    mask = np.logical_and(prefcond, disprefcond)
    keeps = ts[mask]
    return keeps    
    
    
