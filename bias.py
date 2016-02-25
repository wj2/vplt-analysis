
import numpy as np
import matplotlib.pyplot as plt
from utility import *
import eyes as es

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
    plt.show(block=False)
    return f

def get_bias_tc(data, leftconds=(10,), rightconds=(7,), pref_range=None,
                dispref_range=None, winsize=50, winstep=20, fix_time=500,
                tlen=5000):
    for i, cond in enumerate(leftconds):
        condmask = data[condnum_field] == cond
        ts = data[condmask]
        keep_ts = get_fit_trials(ts, left_views_field, right_views_field,
                                 pref_range, dispref_range)
        if i == 0:
            left_ts = keep_ts
        else:
            left_ts = np.concatenate((left_ts, keep_ts))
    for i, cond in enumerate(rightconds):
        condmask = data[condnum_field] == cond
        ts = data[condmask]
        keep_ts = get_fit_trials(ts, right_views_field, left_views_field,
                                 pref_range, dispref_range)
        if i == 0:
            right_ts = keep_ts
        else:
            right_ts = np.concatenate((right_ts, keep_ts))
    look_course_l = get_look_img(left_ts, 1, 2, 0, fix_time=fix_time, 
                                 tlen=tlen)
    look_course_r = get_look_img(right_ts, 2, 1, 0, fix_time=fix_time,
                                 tlen=tlen)
    look_courses = np.concatenate((look_course_l, look_course_r), axis=0)
    xs = np.arange(-fix_time, tlen)
    return look_courses, xs

def pref_dispref_bias(look_courses, xs, winsize=50., winstep=1.):
    win = np.ones(winsize)
    mean_lc = np.mean(look_courses, axis=0)
    bias_tc = mean_lc[:, 0] - mean_lc[:, 1]
    if winsize > 1:
        smooth_tc = np.convolve(bias_tc, win, 'valid') / winsize
        xs = np.convolve(xs, win, 'valid') / winsize
    else:
        smooth_tc = bias_tc
        smooth_xs = xs
    skipped = smooth_tc[::winstep]
    skipped_xs = smooth_xs[::winstep]
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
                 tlen=5000):
    all_eyes = np.zeros((len(trials), fix_time+tlen, 3))
    for i, t in enumerate(trials):
        start_ind = t[eyemove_flag]
        end_ind = start_ind + fix_time + tlen
        eyeper = t[eye_field][start_ind:end_ind, :]
        all_eyes[i, :, :] = categ_eyepos(eyeper, left_ind, right_ind, off_ind,
                                         left_cent, right_cent, img_wid, 
                                         img_hei)
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
    
    
