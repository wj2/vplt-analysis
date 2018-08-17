
import numpy as np
import general.utility as u
import general.plotting as gpl
import matplotlib.pyplot as plt
import pref_looking.eyes as es

def get_fsc_bias(d, err_field='TrialError', oa_field='target_onset_time_diff',
                 user_field='UserVars', corr=0, incorr=6, boots=1000,
                 conds=None, condfield='ConditionNumber',
                 loc_field='first_target_location', target_num='target_num'):
    def _err_ratio_func(es):
        corr_errs = np.sum(es == corr)
        incorr_errs = np.sum(es == incorr)
        er = corr_errs/(corr_errs + incorr_errs)
        return er
    if conds is not None:
        cs = d[condfield][0,0]
        cond_mask = np.array(list([x in conds for x in cs]))
    else:
        cond_mask = np.ones(d[condfield][0,0].shape[0], dtype=bool)
    tnums = d[user_field][0,0][target_num] == 2
    cond_mask = np.logical_and(tnums[0, :], cond_mask)
    errs = d[err_field][0,0][cond_mask, 0]
    oas = d[user_field][0,0][oa_field][0, cond_mask]
    loc = d[user_field][0,0][loc_field][0, cond_mask]
    x_oas = np.unique(oas)
    x_oas = np.array([x[0,0] for x in x_oas])
    err_rate = np.zeros((len(x_oas), boots))
    for i, oa in enumerate(x_oas):
        err_types = errs[oas == oa]
        locs = loc[oas == oa]
        locs_arr = np.array([l for l in locs])
        er = u.bootstrap_list(err_types, _err_ratio_func, n=boots)
        if oa < 0:
            er = 1 - er
        err_rate[i] = er
    return x_oas, err_rate

def plot_fsc_bias(ds, labels=None, colors=None, ax=None, **kwargs):
    if labels is None:
        labels = ('',)*len(ds)
    if colors is None:
        colors = (None,)*len(ds)
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    for i, d in enumerate(ds):
        x_oas, err_rates = get_fsc_bias(d, **kwargs)
        gpl.plot_trace_werr(x_oas, err_rates.T, error_func=gpl.conf95_interval,
                            ax=ax, label=labels[i], color=colors[i])
    ax.set_ylabel('P(look left)')
    ax.set_xlabel('onset asynchrony (left - right)')
    return ax

def _make_ratio_function(func1, func2):
    def _ratio_func(ts):
        one = func1(ts)
        two = func2(ts)
        norm = one / (one + two)
        return norm
    return _ratio_func

def plot_spatial_bias(ds, conds, labels=None, colors=None, ax=None,
                      filt_func=None, left_field='left_first',
                      right_field='right_first', boots=1000):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    if labels is None:
        labels = ('',)*len(ds)
    if colors is None:
        colors = (None,)*len(ds)
    l_func = lambda x: np.sum(x[left_field])
    r_func = lambda x: np.sum(x[right_field])
    ratio_func = _make_ratio_function(l_func, r_func)
    for i, d in enumerate(ds):
        trs = u.get_only_conds(d, conds)
        if filt_func is not None:
            trs = filt_func(trs)
        v = u.bootstrap_list(trs, ratio_func, n=boots)
        v = np.array(v).reshape((-1, 1))
        gpl.plot_trace_werr(np.array([0]) + i, v,
                            error_func=gpl.conf95_interval, 
                            ax=ax, label=labs[i], fill=False, color=cols[i])
    return ax

def plot_sdmst_bias(ds, condlist, cond_labels=None, d_labels=None,
                    d_colors=None, ax=None, filt_func=None, boots=1000,
                    err_field='TrialError', corr=0, incorr=6, offset_div=6,
                    rotate_labels=True):
    if cond_labels is None:
        cond_labels = ('',)*len(condlist)
    if d_labels is None:
        d_labels = ('',)*len(ds)
    if d_colors is None:
        d_colors = ('',)*len(ds)
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    corr_func = lambda x: np.sum(x['TrialError'] == corr)
    incorr_func = lambda x: np.sum(x['TrialError'] == incorr)
    ratio_func = _make_ratio_function(corr_func, incorr_func)
    for i, d in enumerate(ds):
        for j, c in enumerate(condlist):
            trs = u.get_only_conds(d, (c,))
            c_r = u.bootstrap_list(trs, ratio_func, n=boots)
            c_r = np.array(c_r).reshape((-1, 1))
            offset = (i - len(ds)/2)/10
            if j == 0:
                use_label = d_labels[i]
            else:
                use_label = ''
            gpl.plot_trace_werr(np.array([j]) + offset, c_r,
                                error_func=gpl.conf95_interval, 
                                ax=ax, label=use_label, fill=False,
                                color=d_colors[i])
    ax.set_xticks(range(len(condlist)))
    ax.set_xlabel('condition')
    ax.set_ylabel('P(correct)')
    if rotate_labels:
        ax.set_xticklabels(cond_labels, rotation=90)
    else:
        ax.set_xticklabels(cond_labels)
    return ax
            
def plot_plt_bias(ds, labs=None, cols=None, filt_errs=True,
                  err_field='TrialError',
                  corr=0, sep_field='angular_separation', axs=None, cond_nf=22,
                  cond_fn=19, cond_nn=21, cond_ff=20, postthr='fixation_off',
                  sacc_vthr=.1, readdpost=False, lc=(-9, 0), rc=(9, 0), wid=3,
                  hei=3, centoffset=(0,0), use_bhv_img_params=True, boots=1000,
                  sep_filt=None, figsize=(12, 4)):
    if labs is None:
        labs = ('',)*len(ds)
    if cols is None:
        cols = ('',)*len(ds)
    if axs is None:
        f = plt.figure(figsize=figsize)
        ax_fs_nl = f.add_subplot(1, 3, 1)
        ax_fs_nr = f.add_subplot(1, 3, 2)
        ax_fs_bias = f.add_subplot(1, 3, 3)
    else:
        ax_fs_nl, ax_fs_nr, ax_fs_bias = axs
    ax_fs_nl.set_title('left novelty bias')
    ax_fs_nr.set_title('right novelty bias')
    ax_fs_bias.set_title('full bias')
    ax_fs_nl.set_ylabel('P(look left| novel vs familiar) -\n'
                        'P(look left | homogeneous)')
    ax_fs_bias.set_ylabel('P(look novel)')
    ax_fs_nl.set_xlabel('session')
    ax_fs_nr.set_xlabel('session')
    ax_fs_bias.set_xlabel('session')
    
    conds = (cond_nn, cond_fn, cond_nf, cond_ff)
    for i, d in enumerate(ds):
        seps = np.unique(d[sep_field])
        if sep_filt is not None:
            seps = sep_filt(seps)
        for j, s in enumerate(seps):
            d_sep = d[d[sep_field] == s]
            
            x = es.get_fixtimes(d, conds, postthr=postthr, thr=sacc_vthr,
                                readdpost=readdpost, lc=lc, rc=rc, wid=wid,
                                hei=hei, centoffset=centoffset,
                                use_bhv_img_params=use_bhv_img_params)
            ls, ts, begs, ends = x
            fls = es.get_first_sacc_latency_nocompute(begs, ts, onim=False,
                                                      first_n=1, sidesplit=True)
            sacc_arr1 = _make_fls_arr(fls, cond_nf)
            sacc_arr_nn = _make_fls_arr(fls, cond_nn)
            sacc_arr_ff = _make_fls_arr(fls, cond_ff)
            sacc_arr_null = np.concatenate((sacc_arr_nn, sacc_arr_ff))
            # look novel when on left
            f1 = lambda x: np.sum(x == 0)
            f2 = lambda x: np.sum(x == 1)
            rf1 = _make_ratio_function(f1, f2)
            nov_left = u.bootstrap_list(sacc_arr1, rf1, n=boots)
            nov_left = nov_left.reshape((-1, 1))
            sub1 = np.mean(u.bootstrap_list(sacc_arr_null, rf1, n=boots))
            gpl.plot_trace_werr(np.array([0]) + i, nov_left - sub1,
                                error_func=gpl.conf95_interval, ax=ax_fs_nl,
                                label=labs[i], fill=False, color=cols[i])

            sacc_arr2 = _make_fls_arr(fls, cond_fn, l=1, r=0)
            # look novel when on right
            nov_right = u.bootstrap_list(sacc_arr2, rf1, n=boots)
            nov_right = nov_right.reshape((-1, 1))
            rf2 = _make_ratio_function(f2, f1)
            sub2 = np.mean(u.bootstrap_list(sacc_arr_null, rf2, n=boots))
            gpl.plot_trace_werr(np.array([0]) + i, nov_right - sub2,
                                error_func=gpl.conf95_interval, ax=ax_fs_nr,
                                fill=False, color=cols[i])

            full_sacc_arr = np.concatenate((sacc_arr1, sacc_arr2))
            nov_full = u.bootstrap_list(full_sacc_arr, rf1, n=boots)
            nov_full = nov_full.reshape((-1, 1))
            gpl.plot_trace_werr(np.array([0]) + i, nov_full,
                                error_func=gpl.conf95_interval, ax=ax_fs_bias,
                                fill=False, color=cols[i])
            
            
            
def _make_fls_arr(fls, cond, l=0, r=1, o=2):
    c = fls[cond]
    return np.array((l,)*len(c['l']) + (r,)*len(c['r'])
                    + (o,)*len(c['o']))
