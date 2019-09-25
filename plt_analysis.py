
import numpy as np
import matplotlib.pyplot as plt
import general.utility as u
import general.neural_analysis as na
import general.plotting as gpl
import pref_looking.bias as b

def nanmean_axis1(x):
    return np.nanmean(x, axis=1)

def _get_leftright_conds(conds, li, ri, conds_ref='plt_conds'):
    left_conds = (conds[conds_ref][li],)
    right_conds = (conds[conds_ref][ri],)
    return left_conds, right_conds

def _get_ltrls_rtrls(data, conds, trial_type='trial_type', left_ind=3,
                     right_ind=0):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind)
    left_func = u.make_trial_constraint_func((trial_type,), (left_conds,),
                                             (np.isin,))
    right_func = u.make_trial_constraint_func((trial_type,), (right_conds,),
                                              (np.isin,))
    l_trls = data[left_func(data)]
    r_trls = data[right_func(data)]
    return l_trls, r_trls

def get_dwell_distribution(data, conds, trial_type='trial_type', left_ind=3,
                           right_ind=0, cutoff=-1, len_limit=None):
    l_trls, r_trls = _get_ltrls_rtrls(data, conds, trial_type=trial_type,
                                      left_ind=left_ind, right_ind=right_ind)
    nov_fixes = []
    fam_fixes = []
    for tr in l_trls:
        targs = tr['saccade_targ'][:-1]
        nov_fs = tr['saccade_lens'][:cutoff][targs[:cutoff] == b'l']
        nov_fixes = nov_fixes + list(nov_fs)
        fam_fs = tr['saccade_lens'][:cutoff][targs[:cutoff] == b'r']
        fam_fixes = fam_fixes + list(fam_fs)
    for tr in r_trls:
        targs = tr['saccade_targ'][:-1]
        nov_fs = tr['saccade_lens'][:cutoff][targs[:cutoff] == b'r']
        nov_fixes = nov_fixes + list(nov_fs)
        fam_fs = tr['saccade_lens'][:cutoff][targs[:cutoff] == b'l']
        fam_fixes = fam_fixes + list(fam_fs)
    nov_fixes = np.array(nov_fixes)
    fam_fixes = np.array(fam_fixes)
    if len_limit is not None:
        nov_fixes = nov_fixes[nov_fixes < len_limit]
        fam_fixes = fam_fixes[fam_fixes < len_limit]
    return nov_fixes, fam_fixes

def get_fixation_proportion(data, conds, n, trial_type='trial_type', left_ind=3,
                            right_ind=0):
    l_trls, r_trls = _get_ltrls_rtrls(data, conds, trial_type=trial_type,
                                      left_ind=left_ind, right_ind=right_ind)
    nov_props = []
    fam_props = []
    for tr in l_trls:
        n_sacc = tr['saccade_targ'][:n] == b'l'
        f_sacc = tr['saccade_targ'][:n] == b'r'
        n_prop = np.sum(n_sacc)/len(n_sacc)
        nov_props.append(n_prop)
        f_prop = np.sum(f_sacc)/len(f_sacc)
        fam_props.append(f_prop)
    for tr in r_trls:
        n_sacc = tr['saccade_targ'][:n] == b'r'
        f_sacc = tr['saccade_targ'][:n] == b'l'
        n_prop = np.sum(n_sacc)/len(n_sacc)
        nov_props.append(n_prop)
        f_prop = np.sum(f_sacc)/len(f_sacc)
        fam_props.append(f_prop)
    return nov_props, fam_props

def get_side_bias(data):
    lf = data['left_first']
    rf = data['right_first']
    side_bias = np.sum(lf)/(np.sum(lf) + np.sum(rf))
    return side_bias

def get_first_saccade_prob(data, conds, trial_type='trial_type',
                            left_ind=3, right_ind=0):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind)
    left_func = u.make_trial_constraint_func((trial_type,), (left_conds,),
                                             (np.isin,))
    right_func = u.make_trial_constraint_func((trial_type,), (right_conds,),
                                              (np.isin,))
    d_lmask = data[left_func(data)]
    total_l = np.sum(np.logical_or(d_lmask['left_first'],
                                   d_lmask['right_first']))
    d_rmask = data[right_func(data)]
    total_r = np.sum(np.logical_or(d_rmask['left_first'],
                                   d_rmask['right_first']))
    total_fs = np.sum(d_lmask['left_first']) + np.sum(d_rmask['right_first'])
    if total_l + total_r == 0:
        first_sacc_prob = np.nan
    else:
        first_sacc_prob = total_fs / (total_l + total_r)
    return first_sacc_prob

def get_bias_timecourse(data, conds, t_begin, t_end, winsize, winstep,
                         left_ind=3, right_ind=0):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind)
    out = b.get_bias_tc(data, left_conds, right_conds, use_bhv_img_params=True,
                        winsize=winsize, winstep=winstep, fix_time=-t_begin,
                        tlen=t_end)
    p, e, d, p_xs = out
    return p_xs, d
    
def plot_stanglm_collection(models, params, labels, param_funcs, link_strings,
                            panel_hei=6, panel_wid=6):
    n_ax = len(params)
    fig_side = np.ceil(np.sqrt(n_ax))
    f = plt.figure(figsize=(fig_side*panel_hei, fig_side*panel_wid))

    for i, p in enumerate(params):
        pf = param_funcs[i]
        ls = link_strings[i]
        ax_i = f.add_subplot(fig_side, fig_side, i + 1)
        try:
            _ = len(pf)
        except:
            pf = (pf,)*len(p)
        gpl.plot_stanglm_selectivity_scatter(models, p, labels,
                                             ax=ax_i, param_funcs=pf,
                                             link_string=ls)
    return f


def get_feat_tuning_index(a, b, boots=1000, ind_func=u.index_func,
                          with_replace=True):
    if len(a.shape) == 1:
        a = a.reshape((-1, 1))
    if len(b.shape) == 1:
        b = b.reshape((-1, 1))
    inds = np.zeros((boots, a.shape[1]))
    n_rs = min(a.shape[0], b.shape[0])
    for i in range(boots):
        a_samp = u.resample_on_axis(a, n_rs, axis=0,
                                    with_replace=with_replace)
        b_samp = u.resample_on_axis(b, n_rs, axis=0,
                                    with_replace=with_replace)
        inds[i] = ind_func(a_samp, b_samp)
    p_high = np.sum(inds > 0, axis=0).reshape((1, -1))/boots
    p_low =  np.sum(inds < 0, axis=0).reshape((1, -1))/boots
    p_arr = np.concatenate((p_high, p_low), axis=0)
    p = np.min(p_arr, axis=0)
    p[np.all(np.isnan(inds), axis=0)] = np.nan
    return inds, p

def get_sus_tuning(a_pop, b_pop, boots=1000, ind_func=u.index_func,
                   with_replace=True):
    ks = list(a_pop.keys())
    n_ts = a_pop[ks[0]].shape[1]
    n_nrs = len(ks)
    inds = np.zeros((n_nrs, boots, n_ts))
    ps = np.zeros((n_nrs, n_ts))
    for i, k in enumerate(ks):
        inds[i], ps[i] = get_feat_tuning_index(a_pop[k], b_pop[k], boots=boots,
                                               ind_func=ind_func,
                                               with_replace=with_replace)
    return inds, ps

def get_index_pairs(pop_pairs, labels, boots=1000, ind_func=u.index_func,
                    with_replace=True, axs=None, temporal_func=nanmean_axis1):
    inds_dict = {}
    for i, pp in enumerate(pop_pairs):
        inds, ps = get_sus_tuning(pp[0], pp[1], boots=boots, ind_func=ind_func,
                                  with_replace=with_replace)
        inds_dict[labels[i]] = (inds, ps)
    return inds_dict

def plot_hist_index_pairs(ind_dict, temporal_func=nanmean_axis1, axs=None,
                          plot_sig=True, plot_not_sig=True, figsize=(10, 3),
                          boot_central_func=np.nanmean, sig_thr=.05,
                          title='', f=None, sig_colors=None, all_color=None):
    if axs is None:
        f, axs = plt.subplots(1, len(ind_dict), figsize=figsize, sharex='all',
                              sharey='all')
    if sig_colors is None:
        sig_colors = {l:None for l in ind_dict.keys()}
    for i, k in enumerate(ind_dict.keys()):
        ax = axs[i]
        inds = ind_dict[k][0]
        ps = ind_dict[k][1]
        i_pl = temporal_func(boot_central_func(inds, axis=1))
        p_comp = temporal_func(ps)
        neur_mask = np.logical_or(np.logical_and(plot_sig,
                                                 p_comp < sig_thr/2),
                                  np.logical_and(plot_not_sig,
                                                 p_comp >= sig_thr/2))
        i_pl_allp = i_pl[neur_mask]
        h_extre = np.max(np.abs(i_pl_allp))
        h_range = (-h_extre, h_extre)
        _, bins, _ = ax.hist(i_pl_allp, color=all_color, range=h_range)
        sig_pl = i_pl[p_comp < sig_thr/2]
        color = sig_colors[k]
        s_extre = np.max(np.abs(sig_pl))
        s_range = (-s_extre, s_extre)
        ax.hist(sig_pl, bins=bins, color=color, range=s_range)
        ax.set_xlabel(k)
        if i == 0:
            ax.set_ylabel('neurons')
        gpl.clean_plot(ax, i)
    f.suptitle(title)
    return f

def plot_scatter_index(ind_dict, labels, temporal_func=nanmean_axis1, ax=None,
                       figsize=(3, 3), boot_central_func=np.nanmean,
                       sig_thr=.05, title='', sig_colors=None, all_color=None):
    if sig_colors is None:
        sig_colors = {l:None for l in labels}
    ind1, ps1 = ind_dict[labels[0]]
    ind2, ps2 = ind_dict[labels[1]]
    ind1 = temporal_func(boot_central_func(ind1, axis=1))
    ps1 = temporal_func(ps1)
    ind2 = temporal_func(boot_central_func(ind2, axis=1))
    ps2 = temporal_func(ps2)
    sig_ps1 = ps1 < sig_thr/2
    sig_ps2 = ps2 < sig_thr/2
    sig_both = np.logical_and(sig_ps1, sig_ps2)
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    ax.plot(ind1, ind2, 'o', color=all_color)
    color1 = sig_colors[labels[0]]
    color2 = sig_colors[labels[1]]
    color_both = np.mean((color1, color2), axis=0)
    ax.plot(ind1[sig_ps1], ind2[sig_ps1], 'o', color=color1)
    ax.plot(ind1[sig_ps2], ind2[sig_ps2], 'o', color=color2)
    ax.plot(ind1[sig_both], ind2[sig_both], 'o', color=color_both)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    gpl.clean_plot(ax, 0)
    return ax    

def plot_scatter_combos(ind_dict, labels, temporal_func=nanmean_axis1,
                        sig_colors=None, all_color=None, suptitle='',
                        figsize=(5.5, 2.5), sig_thr=.05):
    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    contra_labels = (labels[2], labels[0])
    plot_scatter_index(ind_dict, contra_labels, temporal_func=temporal_func,
                       ax=ax1, sig_colors=sig_colors, all_color=all_color,
                       sig_thr=.05)
    ipsi_labels = (labels[2], labels[1])
    plot_scatter_index(ind_dict, ipsi_labels, temporal_func=temporal_func,
                       ax=ax2, sig_colors=sig_colors, all_color=all_color,
                       sig_thr=sig_thr)
    f.suptitle(suptitle)
    f.tight_layout()
    return f

def plot_prop_selective(ind_dict, xs, labels, sig_colors=None, all_color=None,
                        figsize=(5.5, 2.5), ax=None, sig_thr=.05, boots=1000,
                        lw=5, title='', xlabel='time from image onset (ms)',
                        **kwargs):
    if sig_colors is None:
        sig_colors = {l:None for l in labels}
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(1,1,1)
    for k in labels:
        _, ps = ind_dict[k]
        prop_sel = np.zeros((boots, ps.shape[1]))
        for i in range(ps.shape[1]):
            sig_func = lambda x: np.sum(x < sig_thr/2)/len(x)
            prop_sel[:, i] = u.bootstrap_list(ps[:, i], sig_func, n=boots)
        gpl.plot_trace_werr(xs, prop_sel, color=sig_colors[k], label=k,
                            ax=ax, error_func=gpl.conf95_interval, **kwargs)
    ax.hlines(sig_thr, xs[0], xs[-1], linestyle='dashed', color=all_color,
              linewidth=lw)
    ax.set_ylabel('proportion of selective neurons')
    ax.set_xlabel(xlabel)        
    ax.set_title(title)
    return ax

def _resample_mean_distribs(md, boots):
    tr = np.zeros((md.shape[0], boots))
    for j in range(boots):
        md_samp = np.random.choice(md.shape[1],
                                   md.shape[0])
        inds = np.array(list(zip(range(md.shape[0]),
                                 md_samp)))
        tr[:, j] = np.nanmean(md[range(md.shape[0]), md_samp])
    avg_tr = np.nanmean(tr, axis=0)
    return avg_tr


def plot_strength_selective(ind_dict, xs, labels, sig_colors=None,
                            zero_color=None, figsize=(15, 4), axs=None,
                            sig_thr=.05, boots=1000, lw=5, title='',
                            xlabel='time from image onset (ms)', **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, len(labels), figsize=figsize, sharex=True,
                              sharey=True)
    for j, k in enumerate(labels):
        ax = axs[j]
        inds, ps = ind_dict[k]
        mean_inds = np.nanmean(inds, axis=1)
        pos_tr = np.zeros((boots, ps.shape[1]))
        neg_tr = np.zeros((boots, ps.shape[1]))
        for i in range(ps.shape[1]):
            sig_inds = ps[:, i] < 1.1 # sig_thr/2
            pos_inds = mean_inds[:, i] > 0
            neg_inds = mean_inds[:, i] < 0
            spos_inds = inds[np.logical_and(sig_inds, pos_inds), :, i]
            sneg_inds = inds[np.logical_and(sig_inds, neg_inds), :, i]
            if spos_inds.shape[0] > 0:
                pos_tr[:, i] = _resample_mean_distribs(spos_inds, boots)
            else:
                pos_tr[:, i] = 0
            if sneg_inds.shape[0] > 0:
                neg_tr[:, i] = _resample_mean_distribs(sneg_inds, boots)
            else:
                neg_tr[:, i] = 0
        gpl.plot_trace_werr(xs, neg_tr, ax=ax,
                            error_func=gpl.conf95_interval)
        gpl.plot_trace_werr(xs, pos_tr, ax=ax,
                            error_func=gpl.conf95_interval)
        ax.hlines(0, xs[0], xs[-1], linewidth=lw, color=zero_color,
                      linestyle='dashed')
        ax.set_ylabel(k)
        ax.set_xlabel(xlabel)


def organize_indices(data, labels):
    fin, nin, fout, nout, sin, sout = data
    pairs = ((nin, fin), (nout, fout), (sin, sout))
    ind_dict = get_index_pairs(pairs, labels)
    return ind_dict

default_basefolder = ('/Users/wjj/Dropbox/research/uc/freedman/'
                      'analysis/pref_looking/figs/')

def describe_population_indices(ns, xs, n_labels, ind_labels, time_inds,
                                binsize, color_dict=None, all_color=None,
                                supply_inds=True,
                                title_templ='{}',
                                period_templ='{} to {}ms after image onset',
                                basefolder=default_basefolder,
                                hist_size=(10, 3), scatter_size=(5.5,2.5),
                                prop_size=(5.5,2.5), sig_thr=.05,
                                prop_xlabel='time from image onset (ms)'):
    full_id = []
    for i, n in enumerate(ns):
        n_label = n_labels[i]
        if not supply_inds:
            ind_dict = organize_indices(n, ind_labels)
        else:
            ind_dict = n
        for j, ti in enumerate(time_inds):
            tf = lambda x: x[:, ti]
            form_info = (int(xs[ti] - binsize/2), 
                         int(xs[ti] + binsize/2))
            period = period_templ.format(*form_info)
            title = title_templ.format(period)
            
            f_hist = plot_hist_index_pairs(ind_dict, temporal_func=tf,
                                           title=title, all_color=all_color,
                                           sig_colors=color_dict, 
                                           figsize=hist_size, sig_thr=sig_thr)

            f_scatter = plot_scatter_combos(ind_dict, ind_labels,
                                            temporal_func=tf,
                                            sig_colors=color_dict,
                                            all_color=all_color, 
                                            suptitle=title, sig_thr=sig_thr,
                                            figsize=scatter_size)

            f_scatter_name = (basefolder
                              + '{}_scatter_{}ms.svg'.format(n_label, xs[ti]))
            f_scatter.savefig(f_scatter_name, bbox_inches='tight',
                              transparent=True)
            f_hist_name = basefolder + '{}_hist_{}ms.svg'.format(n_label,
                                                                 xs[ti])
            f_hist.savefig(f_hist_name, bbox_inches='tight', transparent=True)
        f_prop = plt.figure(figsize=prop_size)
        if i == 0:
            sharey = None
        ax_prop = f_prop.add_subplot(1,1,1, sharey=sharey)
        sharey = ax_prop
        ax_prop = plot_prop_selective(ind_dict, xs, ind_labels,
                                      sig_colors=color_dict,
                                      all_color=all_color, sig_thr=sig_thr,
                                      title=n_label, ax=ax_prop,
                                      xlabel=prop_xlabel)
        f_prop_name = basefolder + '{}_prop.svg'.format(n_label)
        f_prop.savefig(f_prop_name, bbox_inches='tight', transparent=True)
        full_id.append(ind_dict)
    return full_id

def plot_prop_figure(ind_dicts, xs, n_labels, labels, color_dict=None,
                     sig_thr=.05, all_color=None, figsize=None, basefolder='',
                     prop_xlabel='time from image onset (ms)'):
    f_prop = plt.figure(figsize=figsize)
    for i, ind_dict in enumerate(ind_dicts):
        if i == 0:
            sharey = None
            sharex = None
        n_label = n_labels[i]
        ax_prop = f_prop.add_subplot(len(ind_dicts), 1, i + 1,
                                     sharey=sharey, sharex=sharex)
        sharey = ax_prop
        sharex = ax_prop
        ax_prop = plot_prop_selective(ind_dict, xs, labels,
                                      sig_colors=color_dict,
                                      all_color=all_color, sig_thr=sig_thr,
                                      title=n_label, ax=ax_prop,
                                      xlabel=prop_xlabel)
        gpl.clean_plot(ax_prop, i, max_i=len(ind_dicts) - 1, horiz=False)
    f_prop_name = basefolder + 'all_prop.svg'
    f_prop.savefig(f_prop_name, bbox_inches='tight', transparent=True)
    return f_prop

def plot_single_unit_eg(ns, xs, neur_key, labels, colors=None, linestyles=None,
                        ax=None, title=False, legend=True, alphas=None):
    if ax is None:
        ax = f.add_subplot(1,1,1)
    if alphas is None:
        alphas = (1,)*len(ns)
    for i, n in enumerate(ns):
        neuron_n = n[neur_key]
        _ = gpl.plot_trace_werr(xs, neuron_n, ax=ax, label=labels[i],
                                color=colors[i], legend=legend,
                                linestyle=linestyles[i], line_alpha=alphas[i])
    if title:
        ax.set_title(neur_key)
    return ax

def plot_several_single_units(ns, xs, neur_keys, labels, colors=None,
                              linestyles=None, same_fig=True, figsize=(7, 3.5),
                              suptitle=None, title=False,
                              xlabel='time from image onset (ms)',
                              ylabel='spks/second', alphas=None,
                              suptitle_templ='{} single unit examples',
                              file_templ='su_eg_{}.svg', folder='', save=False):
    if same_fig:
        f, axs = plt.subplots(len(neur_keys), 1, figsize=figsize, sharex=True,
                              sharey=True)
    if alphas is None:
        alphas = (1,)*len(ns)
    for i, nk in enumerate(neur_keys):
        if same_fig:
            ax = axs[i]
        else:
            f = plt.figure(figsize=figsize)
            ax = f.add_subplot(1,1,1)
        if i == len(neur_keys) - 1:
            legend = True
        else:
            legend = False
        ax = plot_single_unit_eg(ns, xs, nk, labels, colors, linestyles,
                                 ax=ax, title=title, legend=legend,
                                 alphas=alphas)
        ax.set_ylabel(ylabel)
        if i == len(neur_keys) - 1:
            ax.set_xlabel(xlabel)
    if suptitle is not None:
        st = suptitle_templ.format(suptitle)
        f.suptitle(st)
    if save and suptitle is not None:
        fn = folder + file_templ.format(suptitle)
        f.savefig(fn, bbox_inches='tight', transparent=True)
    return f
    
def svm_decoding_helper(data_dict, min_trials=10, resample=100,
                        dec_pair_labels=None, shuffle=False,
                        kernel='linear', **params):
    out_dicts = {}
    for i, kv in enumerate(data_dict.items()):
        label, data = kv
        for j, p in enumerate(data):
            if dec_pair_labels is None:
                pl = j
            else:
                pl = dec_pair_labels[j]
            if i == 0:
                out_dicts[pl] = {}
            cat1, cat2 = p
            dec = na.svm_decoding(cat1, cat2, require_trials=min_trials, 
                                  resample=resample, shuff_labels=shuffle,
                                  kernel=kernel, **params)
            out_dicts[pl][label] = dec
    return out_dicts

def plot_svm_decoding(results_dict, xs, figsize=None, colordict=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    for k, data in results_dict.items():
        if colordict is not None:
            color = colordict[k]
        else:
            color = None
        gpl.plot_trace_werr(xs, data[0], color=color, ax=ax, label=k,
                            error_func=gpl.conf95_interval)
    return ax
    
def produce_proportions(ns, n_labels, ind_labels, t_filt=None, sig_thr=.01):
    for i, n in enumerate(ns):
        nl = n_labels[i]
        print('-----')
        print(nl)
        for il in ind_labels:
            _, ps = n[il]
            sig_neurs = np.any(ps < sig_thr/2, axis=1)
            num_selective = np.sum(sig_neurs)
            prop_selective = np.sum(sig_neurs) / sig_neurs.shape[0]
            print(il, '{}/{} ({}%)'.format(num_selective, sig_neurs.shape[0],
                                           prop_selective*100))
