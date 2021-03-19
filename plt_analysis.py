
import numpy as np
import matplotlib.pyplot as plt
import general.utility as u
import general.stan_utility as su
import general.neural_analysis as na
import general.plotting as gpl
import pref_looking.bias as b
import pref_looking.definitions as d
import pref_looking.image_selection as sel
import dPCA.dPCA as dPCA
import itertools
from sklearn import svm
import pickle
import pystan as ps
import arviz as az

monkey_paths = {'Stan':'pref_looking/data-stan-itc/',
                'Bootsy':'pref_looking/data-bootsy-itc/',
                'Rufus':'pref_looking/data-rufus/'}

def load_all_data(monkey_paths=monkey_paths):
    data = {monkey_name:load_monkey_data(path, monkey_name)
            for monkey_name, path in monkey_paths.items()}
    return data
    
def load_monkey_data(datadir, monkey_key, max_trials=None):
    pattern = d.data_patterns[monkey_key]

    pdict = d.reading_params[monkey_key]
    c = pdict
    reading_kwargs = d.reading_additional[monkey_key]
    dat = u.load_collection_bhvmats(datadir, pdict, pattern,
                                    make_log_name=False,
                                    max_trials=max_trials,
                                    use_data_name=True)
    return dat, c

def nanmean_axis1(x):
    return np.nanmean(x, axis=1)

def _make_fixation_mapping(data, sacc_targ='saccade_targ'):
    all_targ = np.concatenate(data[sacc_targ])
    options = np.unique(all_targ)
    ax_map = {o:i for i, o in enumerate(options)}
    return ax_map

def make_saccade_tree(trs, tree_depth=5, discard_short=False,
                      sacc_targ='saccade_targ', sacc_len='saccade_lens',
                      ax_map=None):
    if ax_map is None:
        ax_map = _make_fixation_mapping(trs, sacc_targ)
    n_options = len(ax_map.keys())
    ns = np.zeros(tree_depth)
    transition_tree = np.zeros((n_options, n_options, tree_depth - 1))
    occup_tree = np.zeros((n_options, tree_depth))
    len_occup_tree = np.zeros((n_options, tree_depth))
    for i, tr in enumerate(trs):
        st = tr[sacc_targ]
        sl = tr[sacc_len]
        if len(sl) >= tree_depth:
            td_i = tree_depth
        elif not discard_short:
            td_i = len(sl)
        for i in range(td_i):
            ns[i] += 1
            curr = ax_map[st[i]]
            occup_tree[curr, i] += 1
            len_occup_tree[curr, i] += sl[i]
            if i < td_i - 1:
                next_ = ax_map[st[i+1]]
                transition_tree[curr, next_, i] += 1
    return transition_tree, occup_tree, len_occup_tree, ns, ax_map    

def get_nov_fam_tree(data, conds, tree_depth=5, discard_short=False,
                     sacc_targ='saccade_targ', sacc_len='saccade_lens',
                     **trial_kwargs):
    labels = ('n', 'f', 'o')
    ltrs, rtrs = _get_ltrls_rtrls(data, conds, **trial_kwargs)
    ax_map_r = {b'l':1, b'r':0, b'o':2}
    out_r = make_saccade_tree(rtrs, tree_depth, discard_short, sacc_targ,
                              sacc_len, ax_map_r)
    tt_r, ot_r, lot_r, ns_r, _ = out_r
    ax_map_l = {b'l':0, b'r':1, b'o':2}
    out_l = make_saccade_tree(ltrs, tree_depth, discard_short, sacc_targ,
                              sacc_len, ax_map_l)
    tt_l, ot_l, lot_l, ns_l, _ = out_l
    tt = tt_l + tt_r
    ot = ot_l + ot_r
    lot = lot_l + lot_r
    ns = ns_l + ns_r
    return tt, ot, lot, ns, labels

def _get_leftright_conds(conds, li, ri, conds_ref='plt_conds'):
    left_conds = (conds[conds_ref][li],)
    right_conds = (conds[conds_ref][ri],)
    return left_conds, right_conds

def _get_ltrls_rtrls(data, conds, trial_type='trial_type', left_ind=3,
                     right_ind=0, noerr=True, errfield='TrialError', errtarg=0):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind)
    left_func = u.make_trial_constraint_func((trial_type, errfield),
                                             (left_conds, errtarg),
                                             (np.isin, np.equal))
    right_func = u.make_trial_constraint_func((trial_type, errfield),
                                              (right_conds, errtarg),
                                              (np.isin, np.equal))
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

def get_look_times_nsacc(data, conds, n, trial_type='trial_type', left_ind=3,
                            right_ind=0):
    l_trls, r_trls = _get_ltrls_rtrls(data, conds, trial_type=trial_type,
                                      left_ind=left_ind, right_ind=right_ind)
    nov_cumu = []
    fam_cumu = []
    for tr in l_trls:
        if len(tr['saccade_lens']) >= n:
            n_sacc = tr['saccade_targ'][:n] == b'l'
            f_sacc = tr['saccade_targ'][:n] == b'r'
            n_times = tr['saccade_lens'][:n][n_sacc]
            f_times = tr['saccade_lens'][:n][f_sacc]
            nov_cumu.append(np.sum(n_times)/n)
            fam_cumu.append(np.sum(f_times)/n)
    for tr in r_trls:
        if len(tr['saccade_lens']) >= n:
            n_sacc = tr['saccade_targ'][:n] == b'r'
            f_sacc = tr['saccade_targ'][:n] == b'l'
            n_times = tr['saccade_lens'][:n][n_sacc]
            f_times = tr['saccade_lens'][:n][f_sacc]
            nov_cumu.append(np.sum(n_times)/n)
            fam_cumu.append(np.sum(f_times)/n)
    return np.array(nov_cumu), np.array(fam_cumu)

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

def compute_simple_sdms_performance(data, cg, **kwargs):
    red_group = {'all':cg['all']}
    perform_dict = compute_sdms_performance(data, red_group, **kwargs)
    out = np.mean(perform_dict['all'])
    return out

def compute_sdms_performance(data, cgroups, n_boots=1000, errfield='TrialError',
                             corr=0, incorr=6):
    corr_func = lambda x: np.sum(x[errfield] == corr)
    incorr_func = lambda x: np.sum(x[errfield] == incorr)
    ratio_func = u.make_ratio_function(corr_func, incorr_func)
    out_dict = {}
    for key, cgroup in cgroups.items():
        trs = u.get_only_conds(data, cgroup)
        if n_boots == 1:
            c_r = ratio_func(trs)
        else:
            c_r = u.bootstrap_list(trs, ratio_func, n=n_boots)
        c_r = np.array(c_r).reshape((-1, 1))
        out_dict[key] = c_r
    return out_dict

def plot_sdms_performance(perf_dict, offset, ax, color=None,
                          rotate_labels=True):
    namelist = []
    for j, (label, perf) in enumerate(perf_dict.items()):
        gpl.plot_trace_werr(np.array([j]) + offset, perf,
                            error_func=gpl.conf95_interval, 
                            ax=ax, fill=False, points=True,
                            color=color)
        namelist.append(label)
    ax.set_xticks(range(len(namelist)))
    ax.set_ylabel('P(correct)')
    if rotate_labels:
        ax.set_xticklabels(namelist, rotation=90)
    else:
        ax.set_xticklabels(namelist)
    return ax

def format_data_stan_models(data, time_cent, time_window, marker_func,
                            constr_funcs, cond_inds, labels, min_spks=1,
                            min_trials=5, interactions=(), double_factors=True,
                            full_interactions=True, single_time=True,
                            zscore=True, **kwargs):
    start_time = time_cent - time_window/2
    end_time = time_cent + time_window/2
    binsize = time_window
    binstep = time_window*2
    marker_funcs = (marker_func,)*len(constr_funcs)
    out = na.organize_spiking_data(data, constr_funcs, marker_funcs,
                                   start_time, end_time, binsize, binstep,
                                   min_spks=min_spks, zscore=zscore, **kwargs)
    spks, xs = out
    neur_form_shape = np.max(cond_inds, axis=0) + 1
    data_list = []
    cond_list = []
    
    for k in spks[0].keys():
        neur_form = np.zeros(neur_form_shape, dtype=dict)
        trls = np.inf
        for j, d_j in enumerate(spks):
            trls = int(np.min((trls, len(d_j[k]))))
            neur_form[cond_inds[j]] = {k:d_j[k]}
        if trls >= min_trials:
            out = na.glm_asymm_trials_format(
                neur_form, k, cond_inds,
                cond_labels=labels,
                interactions=interactions,
                double_factors=double_factors,
                full_interactions=full_interactions)
            glm_dat, glm_conds, glm_labels = out
            if single_time:
                glm_dat = glm_dat[:1]
            data_list.append(glm_dat)
            cond_list.append(glm_conds)
    return data_list, cond_list, glm_labels, xs

def _get_label_filter(labels, label_set):
    if label_set is None:
        mask = np.ones(len(labels), dtype=bool)
    else:
        mask = np.array(list(l in label_set for l in labels))
    return mask       

def fit_stan_model(data, conds, labels, model_path, only_labels=None,
                   sdmst_num=(('task', 0),), stan_iters=2000, stan_chains=4,
                   sigma_var=1, beta_var=1, modul_var=1, arviz=na.glm_arviz,
                   adapt_delta=.8, max_treedepth=10, reduce_fit=True,
                   **stan_params):
    data = np.squeeze(data)
    conds = np.squeeze(conds)
    sdmst_ind = list(labels).index(sdmst_num)
    task_var = conds[:, sdmst_ind]
    mask = _get_label_filter(labels, only_labels)
    conds = conds[..., mask]
    if conds.shape[1] == 0:
        conds = np.zeros((len(data), 1))
    sm = pickle.load(open(model_path, 'rb'))
    stan_data = dict(N=data.shape[0], K=conds.shape[1], x=conds,
                     y=data, beta_var=beta_var, sigma_var=sigma_var,
                     context=task_var, modul_var=modul_var)
    control = dict(adapt_delta=adapt_delta, max_treedepth=max_treedepth)
    fit = sm.sampling(data=stan_data, iter=stan_iters, chains=stan_chains,
                      control=control, **stan_params)
    diag = ps.diagnostics.check_hmc_diagnostics(fit)
    av = az.from_pystan(posterior=fit, **arviz)
    fit = su.ModelFitContainer(fit)
    if reduce_fit:
        fit = fit_params_only(fit)
    return fit, diag, av

def fit_params_only(fit, pop_keys=('log_lik', 'err_hat')):
    for pk in pop_keys:
        fit.samples.pop(pk)
    return fit

def fit_comparison_models(data, conds, labels, model_path=None,
                          model_path_modu=None, **fit_params):
    if model_path is None:
        model_path = na.stan_file_glm_nomean
    if model_path_modu is None:
        model_path_modu = na.stan_file_glm_modu_nomean

    out_null = fit_stan_model(data, conds, labels, model_path, only_labels=(),
                              **fit_params)
    
    pure_conds = list(filter(lambda x: len(x) == 1 and x[0][0] != 'task',
                             labels))
    out_pure = fit_stan_model(data, conds, labels, model_path,
                              only_labels=pure_conds, **fit_params)

    modulated_conds = list(filter(lambda x: len(x) == 1,
                                  labels))
    out_mod = fit_stan_model(data, conds, labels, model_path_modu,
                             only_labels=modulated_conds, **fit_params)

    second_conds = list(filter(lambda x: len(x) < 3,
                               labels))
    out_sec = fit_stan_model(data, conds, labels, model_path,
                             only_labels=second_conds)

    out_third = fit_stan_model(data, conds, labels, model_path, **fit_params)

    models = {'pure':out_pure[:2], 'modulated':out_mod[:2],
              'second':out_sec[:2], 'third':out_third[:2],
              'null':out_null[:2]}
    models_ar = {'pure':out_pure[2], 'modulated':out_mod[2],
                 'second':out_sec[2], 'third':out_third[2],
                 'null':out_null[2]}
    return models, models_ar

def compare_models(data, tc, tw, marker_func, constr_funcs, constr_inds, labels,
                   av_only=False, store_comp=True, max_fit=None, min_trials=5,
                   **fit_params):
    out = format_data_stan_models(data, tc, tw, marker_func, constr_funcs,
                                  constr_inds, labels, min_trials=min_trials)
    glm_dat, glm_cond, glm_lab, xs = out
    comp_all = []
    fits_all = []
    n_neurs = len(glm_dat)
    for i, gd in enumerate(glm_dat):
        print('{} / {}'.format(i+1, n_neurs))
        fit, comp = fit_comparison_models(gd, glm_cond[i], glm_lab, **fit_params)
        if not av_only:
            fits_all.append(fit)
        if store_comp:
            comp_all.append(az.compare(comp))
        else:
            comp_all.append(comp)
        if max_fit is not None and i + 1 > max_fit:
            break
    return fits_all, comp_all

def _get_rel_loss(cf, model_name, deviance='d_loo', deviance_uncertainty='dse'):
    pl = cf.loc[model_name, deviance]
    if pl > 0:
        pl = pl/cf.loc[model_name, deviance_uncertainty]
    return pl

def summarize_model_comparison(
        comps,
        model_names=('pure', 'modulated', 'second', 'third'),
        n_boots=1000):
    comps_f = list(filter(lambda x: not np.any(x.loc[:, 'warning']), comps))
    out_loss = {}
    out_weight = {}
    out_weightsums = {}
    for cf in comps_f:
        null_loss = _get_rel_loss(cf, 'null')
        if null_loss > 1:
            for mn in model_names:
                mnl = out_loss.get(mn, [])
                mnl.append(_get_rel_loss(cf, mn))
                out_loss[mn] = mnl
                mnw = out_weight.get(mn, [])
                mnw.append(cf.loc[mn, 'weight'])
                out_weight[mn] = mnw
    for mn in model_names:
        w_b = u.bootstrap_list(np.array(out_weight[mn]), np.mean, n=n_boots)
        out_weightsums[mn] = w_b
    return out_loss, out_weight, out_weightsums

def plot_model_comparison(loss, weight, weightsums, axs=None,
                          keys=('pure', 'modulated', 'second'),
                          fwid=1.5, n_boots=1000, eps=.001,
                          labels=None):
    if labels is None:
        labels = {'second':'nonlinear'}
    if axs is None:
        f = plt.figure(figsize=(fwid, 2*fwid))
        ax_l = f.add_subplot(2, 1, 1)
        ax_ws = f.add_subplot(2, 1, 2)
    else:
        ax_l, ax_ws = axs
    ax_vals = []
    ax_labels = []
    cols = []
    vp_seq = []
    for i, k in enumerate(keys[::-1]):
        frac_b = u.bootstrap_list(np.array(loss[k]), lambda x: np.mean(x < 1),
                                  n=n_boots)
        l = gpl.plot_horiz_conf_interval(i, frac_b, ax_l)
        col = l[0].get_color()
        cols.append(col)
        vp_seq.append(weight[k])
        ax_vals.append(i)
        if labels.get(k, None) is None:
            ax_labels.append(k)
        else:
            ax_labels.append(labels[k])
    print(np.sum(np.logical_or(np.array(loss['second']) < eps,
                               np.array(loss['third']) < eps)))
    subset = np.logical_and(np.logical_or(np.array(loss['second']) < eps,
                                          np.array(loss['third']) < eps),
                            np.array(loss['modulated']) > 1)
    print('{} / {}'.format(np.sum(subset), len(subset)))
    vps = gpl.violinplot(vp_seq, positions=ax_vals, vert=False, ax=ax_ws,
                         showextrema=False, showmedians=True)
    ax_l.set_yticks(ax_vals)
    ax_l.set_yticklabels(ax_labels)
    ax_ws.set_yticks(ax_vals)
    ax_ws.set_yticklabels(ax_labels)

    ax_l.set_xlim([0, .9])
    # gpl.clean_plot_bottom(ax_l)
    gpl.clean_plot(ax_l, 1, ticks=False)
    gpl.make_xaxis_scale_bar(ax_l, magnitude=.75, double=False,
                             label='fraction best fit',
                             text_buff=.23)
    
    gpl.clean_plot(ax_ws, 1, horiz=True, ticks=False)
    gpl.make_xaxis_scale_bar(ax_ws, magnitude=.5, double=False,
                             label='relative probability', text_buff=.23)
    # gpl.clean_plot_bottom(ax_ws)
        
def get_first_saccade_prob_bs(data, *args, n_boots=1000, remove_errs=True,
                              errfield='TrialError', angular_separation=180,
                              ang_field='angular_separation', **kwargs):
    if remove_errs:
        data = data[data[errfield] == 0]
    if angular_separation is not None:
        data = data[data[ang_field] == angular_separation]
    f = lambda x: get_first_saccade_prob(x, *args, **kwargs)
    fsps = u.bootstrap_list(data, f, n=n_boots)
    return fsps

def _compute_looking_difference(pref, dispref, t_beg, t_end):
    b = np.sum(pref[t_beg:t_end]) - np.sum(dispref[t_beg:t_end])
    return b

def get_time_bias(data, conds, trial_type='trial_type', left_ind=3, right_ind=0,
                  conds_ref='plt_conds', min_trials=0, left_mark=b'l',
                  right_mark=b'r', start_lag=100, winsize=250,
                  norm=True):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind,
                                                   conds_ref=conds_ref)
    left_func = u.make_trial_constraint_func((trial_type,), (left_conds,),
                                             (np.isin,))
    right_func = u.make_trial_constraint_func((trial_type,), (right_conds,),
                                              (np.isin,))
    biases = []
    d_lmask = data[left_func(data)]
    for d in d_lmask:
        onpref = d['on_left_img']
        ondispref = d['on_right_img']
        t_beg = start_lag
        t_end = t_beg + winsize
        if len(np.array(onpref).shape) > 0 and len(onpref) > winsize:
            b = _compute_looking_difference(onpref, ondispref, t_beg, t_end)
            biases.append(b)
    d_rmask = data[right_func(data)]
    for d in d_rmask:
        on_dispref = d['on_left_img']
        on_pref = d['on_right_img']
        t_beg = start_lag 
        t_end = t_beg + winsize
        if len(np.array(onpref).shape) > 0 and len(onpref) > winsize:
            b = _compute_looking_difference(onpref, ondispref, t_beg, t_end)
            biases.append(b)
    biases = np.array(biases)
    if norm:
        biases = biases/winsize
    return np.mean(biases)

def get_image_saccade_prob(data, conds, trial_type='trial_type',
                           conds_ref='plt_conds', min_trials=0):
    use_conds = conds[conds_ref]
    all_func = u.make_trial_constraint_func((trial_type, 'TrialError'),
                                            (use_conds, 0),
                                            (np.isin, np.equal))
    d_mask = data[all_func(data)]
    total = len(d_mask)
    image_look = np.sum(np.logical_or(d_mask['left_first'],
                                      d_mask['right_first']))
    if total == 0 or total < min_trials:
        first_sacc_prob = np.nan
    else:
        image_sacc_prob = image_look / total
    return image_sacc_prob


def get_first_saccade_prob(data, conds, trial_type='trial_type',
                           left_ind=3, right_ind=0, conds_ref='plt_conds',
                           min_trials=0):
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind,
                                                   conds_ref=conds_ref)
    left_func = u.make_trial_constraint_func((trial_type, 'TrialError'),
                                             (left_conds, 0),
                                             (np.isin, np.equal))
    right_func = u.make_trial_constraint_func((trial_type, 'TrialError'),
                                              (right_conds, 0),
                                              (np.isin, np.equal))
    d_lmask = data[left_func(data)]
    total_l = np.sum(np.logical_or(d_lmask['left_first'],
                                   d_lmask['right_first']))
    d_rmask = data[right_func(data)]
    total_r = np.sum(np.logical_or(d_rmask['left_first'],
                                   d_rmask['right_first']))
    total_fs = np.sum(d_lmask['left_first']) + np.sum(d_rmask['right_first'])
    if total_l + total_r == 0 or total_l + total_r < min_trials:
        first_sacc_prob = np.nan
    else:
        first_sacc_prob = total_fs / (total_l + total_r)
    return first_sacc_prob

def get_sal_bias(data, conds, folder, trial_type='trial_type',
                 conds_ref='plt_conds', use_inds=(1,), time=False,
                 winsize=250, start=100, left_look='on_left_img',
                 right_look='on_right_img', sal_reduce_func=np.mean):
    t_conds = conds[conds_ref]
    use_conds = list(t_conds[i] for i in use_inds)
    t_func = u.make_trial_constraint_func(('TrialError', trial_type,),
                                          (0, use_conds),
                                          (np.equal, np.isin))
    d_t = data[t_func(data)]
    ef = generate_lowlevel_diff_func(folder, d_t, reduce_func=sal_reduce_func)
    sal_diffs = []
    sacc_side = []
    for t in d_t:
        sal_diffs.append(ef(t))
        if time:
            t_end = start + winsize
            left_looks = np.sum(t[left_look][start:t_end])
            right_looks = np.sum(t[right_look][start:t_end])
            ss = (left_looks - right_looks)/winsize
        else:
            if t['left_first']:
                ss = 1
            elif t['right_first']:
                ss = -1
            else:
                ss = 0
        sacc_side.append(ss)
    sal_diffs = np.array(sal_diffs)
    sacc_side = np.array(sacc_side)
    if time:
        mult_mask = np.ones_like(sal_diffs)
        mult_mask[sal_diffs < 0] = -1
        sal_look_times = sacc_side*mult_mask
        bias = np.mean(sal_look_times)
    else:
        ll = np.sum(sacc_side[sal_diffs > 0] == 1)
        lr = np.sum(sacc_side[sal_diffs < 0] == -1)
        lt = np.sum(sacc_side != 0)
        bias = (ll + lr)/lt
    return bias

def get_side_bias(data, conds, trial_type='trial_type',
                  conds_ref='plt_conds', min_trials=0):
    t_conds = conds[conds_ref]
    t_func = u.make_trial_constraint_func((trial_type,), t_conds,
                                          (np.isin,))
    d_t = data[t_func(data)]
    total_l = np.sum(d_t['left_first'])
    total_r = np.sum(d_t['right_first'])
    if total_l + total_r == 0 or total_l + total_r < min_trials:
        first_sacc_prob = np.nan
    else:
        first_sacc_prob = total_l / (total_l + total_r)
    return first_sacc_prob

def get_bias_timecourse(data, conds, t_begin, t_end, winsize, winstep,
                        left_ind=3, right_ind=0, remove_errs=True,
                        errfield='TrialError', conds_ref='plt_conds',
                        angular_separation=180, ang_field='angular_separation'):
    if remove_errs:
        data = data[data[errfield] == 0]
    if angular_separation is not None:
        data = data[data[ang_field] == angular_separation]
    left_conds, right_conds = _get_leftright_conds(conds, left_ind, right_ind,
                                                   conds_ref=conds_ref)
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
    p_high = np.sum(inds >= 0, axis=0).reshape((1, -1))/boots
    p_low =  np.sum(inds <= 0, axis=0).reshape((1, -1))/boots
    p_arr = np.concatenate((p_high, p_low), axis=0)
    p = np.min(p_arr, axis=0)
    p[np.all(np.isnan(inds), axis=0)] = np.nan
    return inds, p

def get_sus_tuning_data(afunc, bfunc, data, mfunc, start_time, end_time,
                        binsize, binstep, boots=1000, ind_func=u.index_func,
                        min_trials=15, min_spks=1, zscore=True,
                        collapse_time_zscore=False):
    out = na.organize_spiking_data(data, (afunc, bfunc), (mfunc,)*2,
                                   start_time, end_time, binsize, binstep,
                                   min_trials=min_trials, min_spks=min_spks,
                                   zscore=zscore,
                                   collapse_time_zscore=collapse_time_zscore)
    (a_pop, b_pop), xs = out
    out = get_sus_tuning(a_pop, b_pop, boots=boots, ind_func=ind_func)
    return out, xs

def get_sus_tuning(a_pop, b_pop, boots=1000, ind_func=u.index_func,
                   with_replace=True):
    ks = list(a_pop.keys())
    n_ts = a_pop[ks[0]].shape[1]
    n_nrs = len(ks)
    inds = np.zeros((n_nrs, boots, n_ts))
    ps = np.zeros((n_nrs, n_ts))
    store_ks = np.zeros(n_nrs, dtype=tuple)
    for i, k in enumerate(ks):
        inds[i], ps[i] = get_feat_tuning_index(a_pop[k], b_pop[k], boots=boots,
                                               ind_func=ind_func,
                                               with_replace=with_replace)
        store_ks[i] = k
    return inds, ps, store_ks

def get_index_pairs(pop_pairs, labels, boots=1000, ind_func=u.index_func,
                    with_replace=True, axs=None, temporal_func=nanmean_axis1):
    inds_dict = {}
    for i, pp in enumerate(pop_pairs):
        inds, ps, _ = get_sus_tuning(pp[0], pp[1], boots=boots, ind_func=ind_func,
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

default_basefolder = ('/Users/wjj/Dropbox/research/'
                      'analysis/pref_looking/figs/')


def plot_eyepos_at_time(data, trl_funcs, timeflag, rel_t=0, ax=None,
                        eyefield='eyepos'):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    for i, tf in enumerate(trl_funcs):
        col = None
        trl_mask = tf(data)
        print(np.sum(trl_mask))
        data_sub = data[trl_mask]
        ts = data_sub[timeflag] + rel_t
        eyedat = data_sub[eyefield]
        for j, trl in enumerate(eyedat):
            if trl.shape[0] > ts[j] and ts[j] > 0:
                pos = trl[ts[j]]
                l = ax.plot(pos[0], pos[1], 'o', color=col)
                col = l[0].get_color()
    return ax

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

def compile_saccade_latencies(d, dfuncs, latt_groups, saccfield='first_sacc_time',
                              fixoff_field='fixation_off'):
    assert len(dfuncs) == len(latt_groups)
    ts = np.unique(latt_groups)
    storage = tuple([] for i in ts)
    for i, ind in enumerate(latt_groups):
        d_i = d[dfuncs[i](d)]
        storage[ind].append(d_i[saccfield] - d_i[fixoff_field])
    return storage

def compute_average_diff(neurs, xs, x_pt, cent_func=np.mean):
    x_ind = np.argmin(np.abs(xs - x_pt))
    assert len(neurs) == 2
    diffs = {}
    for n_key, spks1 in neurs[0].items():
        spks2 = neurs[1][n_key]
        diffs[n_key] = cent_func(spks2[:, x_ind]) - cent_func(spks1[:, x_ind]) 
    return diffs

def plot_neuron_diffs(d1, d2, boots=1000, ax=None, color=None, ax_buff=.25,
                      cent_func=np.mean, pt_alpha=.1, plot_indiv=False,
                      offset=0):
    if ax is None:
        f, ax = plt.subplots(1,1)
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    if plot_indiv:
        for k in d1_keys.intersection(d2_keys):
            ax.plot([0 + offset, 1 + offset], [d1[k], d2[k]], color=color,
                    alpha=pt_alpha)
            ax.plot([0 + offset], [d1[k]], 'o', color=color, alpha=pt_alpha)
            ax.plot([1 + offset], [d2[k]], 'o', color=color, alpha=pt_alpha)
    d1_arr = np.array(list(d1.values()))
    d1_boots = u.bootstrap_list(d1_arr, cent_func, n=boots)
    gpl.plot_trace_werr(np.array([0 + offset]), np.expand_dims(d1_boots, 1),
                        ax=ax, color=color, fill=False,
                        error_func=gpl.conf95_interval,
                        points=True)
    d2_arr = np.array(list(d2.values()))
    d2_boots = u.bootstrap_list(d2_arr, cent_func, n=boots)
    gpl.plot_trace_werr(np.array([1 + offset]), np.expand_dims(d2_boots, 1),
                        ax=ax, color=color, fill=False,
                        error_func=gpl.conf95_interval,
                        points=True)
    xl = ax.get_xlim()
    ax.set_xlim(xl[0] - ax_buff, xl[1] + ax_buff)
    gpl.add_hlines(0, ax)
    return d1_boots, d2_boots

def compile_saccade_velocities(d, dfuncs, latt_groups, s_beg='saccade_begs',
                               s_end='saccade_ends', ep='eyepos',
                               fo='fixation_off', dist_thr=2):
    assert len(dfuncs) == len(latt_groups)
    ts = np.unique(latt_groups)
    storage = tuple([] for i in ts)
    for i, ind in enumerate(latt_groups):
        d_i = d[dfuncs[i](d)]
        for t in d_i:
            b = t[s_beg][0] + t[fo]
            e = t[s_end][0] + t[fo]
            distance = np.sqrt(np.sum((t[ep][b] - t[ep][e])**2))
            if distance > dist_thr:
                vel = 1000*distance/(e - b)
                storage[ind].append(vel)
    return storage

def get_sacc_vel(t, s_beg='saccade_begs', s_end='saccade_ends',
                 ep='eyepos', fo='fixation_off', dist_thr=2):
    b = t[s_beg][0] + t[fo] 
    e = t[s_end][0] + t[fo]
    distance = np.sqrt(np.sum((t[ep][b] - t[ep][e])**2))
    if distance > dist_thr:
        vel = 1000*distance/(e - b)
    else:
        vel = np.nan
    return vel

def plot_single_unit_scatter(ns, bhv, xs, x_pt, neur_key, labels=None, ax=None,
                             colors=None, alphas=None, linestyles=None, ax_i=0):
    x_ind = np.argmin(np.abs(xs - x_pt))
    if ax is None:
        ax = f.add_subplot(1,1,1)
    if alphas is None:
        alphas = (1,)*len(ns)
    if linestyles is None:
        linestyles = (None,)*len(ns)
    if colors is None:
        colors = (None,)*len(ns)
    for i, n in enumerate(ns):
        neuron_n = n[neur_key][:, x_ind]
        bhv_n = bhv[i][neur_key]
        ax.plot(bhv_n, neuron_n, 'o', color=colors[i])
    gpl.clean_plot(ax, ax_i)

def compute_velocity_firing_correlation(neurs, xs, bhvs, x_pt):
    assert len(bhvs) == len(neurs)
    x_ind = np.argmin(np.abs(xs - x_pt))
    storage = []
    for i, neur in enumerate(neurs):
        bhv = bhvs[i]
        storage.append({})
        for k, spks in neur.items():
            behav = np.array(bhv[k], dtype=float)
            mask = np.logical_not(np.isnan(behav))
            behav = behav[mask]
            spks = spks[mask, x_ind]
            cc = np.corrcoef(spks, behav)
            storage[i][k] = cc[1, 0]
    return storage

def compile_fixation_latencies(d, dfuncs, latt_groups, fix_on='fixation_on',
                               fix_acq='fixation_acquired', max_latt=1000):
    assert len(dfuncs) == len(latt_groups)
    ts = np.unique(latt_groups)
    storage = tuple([] for i in ts)
    for i, ind in enumerate(latt_groups):
        d_i = d[dfuncs[i](d)]
        fix_latts = d_i[fix_acq] - d_i[fix_on]
        fix_latts = fix_latts[fix_latts < max_latt]
        storage[ind].append(fix_latts)
    return storage

def plot_single_unit_eg(ns, xs, neur_key, labels, colors=None, linestyles=None,
                        ax=None, title=False, legend=True, alphas=None,
                        error_func=gpl.sem):
    if ax is None:
        ax = f.add_subplot(1,1,1)
    if alphas is None:
        alphas = (1,)*len(ns)
    if linestyles is None:
        linestyles = (None,)*len(ns)
    if colors is None:
        colors = (None,)*len(ns)
    for i, n in enumerate(ns):
        neuron_n = n[neur_key]
        _ = gpl.plot_trace_werr(xs, neuron_n, ax=ax, label=labels[i],
                                color=colors[i], legend=legend, 
                                linestyle=linestyles[i], line_alpha=alphas[i],
                                error_func=error_func)
    if title:
        ax.set_title(neur_key)
    return ax

def plot_decoding_info(decs, pt, ax1, ax2, colors=None,
                       pb_color=(.4, .4, .4), pb_width=2):
    pts_all = {}
    for i, (m, res) in enumerate(decs.items()):
        plot_svm_decoding(res[1], res[2], ax=ax1, colordict=colors)
        gpl.add_vlines(0, ax1)
        gpl.add_hlines(.5, ax1, linestyle='dashed')
        gpl.add_hlines(.5, ax2, linestyle='dashed')
        _, pts, x_ind = plot_svm_decoding_point(res[1], res[2], pt, ax=ax2,
                                                legend=False, colordict=colors,
                                                pb_color=pb_color,
                                                pb_width=pb_width, pb_ax=ax1)
        pts_all[m] = pts
        gpl.clean_plot(ax2, 1)
    return pts_all

def plot_svm_bhv_scatter(decs_pop, bias_dict, pt, ax1, ax2, low_lim=.4,
                         colordict=None, unity_color=(.9, .9, .9),
                         n_boots=1000, print_=False):
    bd_all = []
    sd_all = []
    sb_all = []
    dp_set = set(decs_pop.keys())
    bd_set = set(bias_dict.keys())
    use_keys = dp_set.intersection(bd_set)
    for i, m in enumerate(use_keys):
        pops = decs_pop[m]
        if colordict is None:
            color = None
        else:
            color = colordict[m]
        cond_keys = list(pops[1].keys())
        pop_keys = pops[1][cond_keys[0]][0].keys()
        xs = pops[2]
        x_ind = np.argmin(np.abs(xs - pt))
        for j, pop_num in enumerate(pop_keys):
            r1 = {ck:(pops[1][ck][0][pop_num],) for ck in cond_keys}
            dec_bd = r1[1][0][:, x_ind:x_ind+1]
            dec_sd = r1[0][0][:, x_ind:x_ind+1]
            bd, sd, sb = bias_dict[m]
            bias_b = np.abs(bd[pop_num] - .5) + .5
            bias_s = np.abs(sd[pop_num] - .5) + .5
            bias_side = np.abs(sb[pop_num] - .5) + .5
            gpl.plot_trace_werr(np.array([bias_b]), dec_bd,
                                points=True, color=color,
                                ax=ax1)
            gpl.plot_trace_werr(np.array([bias_s]), dec_sd,
                                points=True, color=color,
                                ax=ax2)
            bd_all.append((bias_b, np.mean(dec_bd)))
            sd_all.append((bias_s, np.mean(dec_sd)))
            sb_all.append((bias_side, np.mean(dec_bd)))
    bd_all = np.array(bd_all)
    sd_all = np.array(sd_all)
    sb_all = np.array(sb_all)
    fs_b_side_b = np.stack((bd_all[:, 0], sb_all[:, 0]), axis=1)
    f = lambda x: np.corrcoef(x[:, 0], x[:, 1])[1, 0]
    if print_:
        bd_boot = u.bootstrap_list(bd_all, f, n=n_boots)
        sd_boot = u.bootstrap_list(sd_all, f, n=n_boots)
        sb_boot = u.bootstrap_list(sb_all, f, n=n_boots)
        bd_sb_boot = u.bootstrap_list(fs_b_side_b, f, n=n_boots)
        gpl.print_mean_conf95(bd_boot, 'combined', 'FS bias-decoding corr',
                              preboot=True)
        gpl.print_mean_conf95(sd_boot, 'combined', 'sDMST perf-decoding corr',
                              preboot=True)
        gpl.print_mean_conf95(sb_boot, 'combined', 'side bias-decoding corr',
                              preboot=True)
        gpl.print_mean_conf95(bd_sb_boot, 'combined', 'side bias-FS bias corr',
                              preboot=True)
    return ax1, ax2

def plot_svm_session_scatter(decs_pop, pt, ax, low_lim=.4, colordict=None,
                             unity_color=(.9, .9, .9)):
    pts_all = {}
    for i, (m, pops) in enumerate(decs_pop.items()):
        if colordict is None:
            color = None
        else:
            color = colordict[m]
        cond_keys = list(pops[1].keys())
        pop_keys = pops[1][cond_keys[0]][0].keys()
        xs = pops[2]
        pts = []
        for j, pop_num in enumerate(pop_keys):
            r1 = {ck:(pops[1][ck][0][pop_num],) for ck in cond_keys}
            pt_d = plot_svm_decoding_scatter(r1[0], r1[1], xs, pt, ax=ax,
                                             color=color)
            pts.append(pt_d)
        pts_all[m] = pts
    ax.plot([low_lim, 1], [low_lim, 1], linestyle='dashed', 
            color=unity_color)
    ax.set_xlim([low_lim, 1])
    ax.set_ylim([low_lim, 1])
    return ax, pts_all

def plot_several_single_units(ns, xs, neur_keys, labels, colors=None,
                              linestyles=None, same_fig=True, figsize=(7, 3.5),
                              suptitle=None, title=False, error_func=gpl.sem,
                              xlabel='time from image onset (ms)',
                              ylabel='spks/second', alphas=None,
                              suptitle_templ='{} single unit examples',
                              file_templ='su_eg_{}.svg', folder='', save=False,
                              same_fig_square=False, sharex=True, sharey=False):
    if same_fig:
        n = len(neur_keys)
        if same_fig_square:
            s = int(np.ceil(np.sqrt(n)))
            figsize = (figsize[0]*s, figsize[1]*s)
            f, axs = plt.subplots(s, s, figsize=figsize, sharex=sharex,
                                  sharey=sharey)
            axs = axs.flatten()
        else:
            figsize = (figsize[0], figsize[1]*len(neur_keys))
            f, axs = plt.subplots(len(neur_keys), 1, figsize=figsize,
                                  sharex=sharex, sharey=sharey)
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
                                 alphas=alphas, error_func=error_func)
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

def cross_svm_decoding(data_dict, pair_inds, monkeys=None,
                       min_trials=10, resample=100, model=svm.SVC,
                       dec_pair_labels=None, shuffle=False,
                       kernel='linear', **params):
    out_dicts = {}
    if monkeys is None:
        monkeys = data_dict.keys()
    for m in monkeys:
        data = data_dict[m]
        for i, (train_ind, test_ind) in enumerate(pair_inds):
            if dec_pair_labels is None:
                pl = (train_ind, test_ind)
            else:
                pl = (dec_pair_labels[train_ind], dec_pair_labels[test_ind])
            train_c1, train_c2 = data[train_ind]
            test_c1, test_c2 = data[test_ind]
            out = na.svm_cross_decoding(train_c1, test_c1, train_c2, test_c2,
                                        require_trials=min_trials,
                                        resample=resample, shuff_labels=shuffle,
                                        kernel=kernel, model=model, **params)
            if i == 0:
                out_dicts[pl] = {}
            out_dicts[pl][m] = out
    return out_dicts


def generate_lowlevel_diff_func(folder, trls, reduce_func=np.mean,
                                imgfield1='leftimg', imgfield2='rightimg'):
    imgdict = sel.compute_lowlevel_salience(folder)
    trl_mask1 = trls['leftimg_type'] == b'FamImgList'
    trl_mask2 = trls['rightimg_type'] == b'FamImgList'
    fam_names = np.unique(trls[trl_mask1]['leftimg'])
    assert np.all(list([fn in imgdict.keys() for fn in fam_names]))

    def extract_func(trl):
        img1_sal = reduce_func(imgdict[trl[imgfield1]])
        img2_sal = reduce_func(imgdict[trl[imgfield2]])
        return img1_sal - img2_sal
    
    return extract_func

def generate_lowlevel_extract_func(folder, trls, reduce_func=np.mean,
                                   imgfield='leftimg'):
    imgdict = sel.compute_lowlevel_salience(folder)
    trl_mask = trls['leftimg_type'] == b'FamImgList'
    fam_names = np.unique(trls[trl_mask]['leftimg'])
    assert np.all(list([fn in imgdict.keys() for fn in fam_names]))

    def extract_func(trl):
        ll_sal = reduce_func(imgdict[trl[imgfield]])
        return ll_sal
    
    return extract_func

def get_eyeslices(eps, ts, pre, post, diff=False, diff_n=0):
    pres = ts + pre
    posts = ts + post 
    arrlen = 2*(post - pre)
    if diff:
        arrlen = arrlen - 2*diff_n
    ep_arr = np.zeros((len(eps), arrlen))
    for i, ep in enumerate(eps):
        if len(ep) > posts[i]:
            ei = ep[pres[i]:posts[i]]
            if diff:
                ei = np.diff(ei, n=diff_n, axis=0)
            ep_arr[i] = np.reshape(ei, -1)
        else:
            ep_arr[i] = np.nan
    mask = np.all(np.logical_not(np.isnan(ep_arr)), axis=1)
    return ep_arr[mask].T

def decode_task_from_eyepos(eps1, ts1, eps2, ts2, pretime, posttime,
                            n_folds=20, diff=True, **params):
    sls1 = get_eyeslices(eps1, ts1, pretime, posttime, diff=diff)
    sls2 = get_eyeslices(eps2, ts2, pretime, posttime, diff=diff)
    out = na.fold_skl(np.expand_dims(sls1, (1, -1)),
                      np.expand_dims(sls2, (-1, 1)),
                      n_folds, params=params)
    print(out)
    out = na.fold_skl(np.expand_dims(sls1, (1, -1)),
                      np.expand_dims(sls2, (-1, 1)),
                      n_folds, shuffle=True, params=params)
    return out

def organize_dpca_transform(data, dfunc_group, mf, start, end, binsize,
                            binstep, min_trials, cond_labels,
                            dfunc_pts, min_spks=5, resample=100,
                            shuff_labels=False, with_replace=False,
                            pop=False, causal_timing=True, min_population=1,
                            use_avail_trials=False, use_max_trials=False,
                            **kwargs):
    mfs = (mf,)*len(dfunc_group)
    if pop:
        out = na.organize_spiking_data_pop(data, dfunc_group, mfs,
                                           start, end, binsize,
                                           binstep=binstep,
                                           min_trials=min_trials,
                                           min_spks=None, 
                                           causal_timing=causal_timing,
                                           **kwargs)
    else:
        out = na.organize_spiking_data(data, dfunc_group, mfs,
                                       start, end, binsize, binstep=binstep,
                                       min_trials=min_trials,
                                       min_spks=min_spks, 
                                       causal_timing=causal_timing,
                                       **kwargs)
    dat, xs = out
    dfunc_maxes = np.max(dfunc_pts, axis=0) + 1
    if use_avail_trials:
        all_mins = {k:np.inf for k in dat[0].keys()}
        all_maxes = {k:0 for k in dat[0].keys()}
        for d in dat:
            all_mins = {k:min(all_mins[k], len(d[k])) for k in d.keys()}
            all_maxes = {k:max(all_maxes[k], len(d[k])) for k in d.keys()}
        min_trials = np.min(list(all_mins.values()))
        max_trials = np.max(list(all_maxes.values()))
    orgs = {}
    if pop:
        for k in dat[0].keys():
            org = np.zeros(dfunc_maxes, dtype=object)
            for i, dp in enumerate(dfunc_pts):
                org[dp] = dat[i][k]
            orgs[k] = org
    else:
        org = np.zeros(dfunc_maxes, dtype=object)
        for i, dp in enumerate(dfunc_pts):
            org[dp] = dat[i]
        orgs['all'] = org
    dpca_fits = {}
    for k, org in orgs.items():
        fits = []
        print('{}: {} neurons'.format(k, len(org[0, 0, 0])))
        for i in range(resample):
            if use_max_trials:
                trls = max_trials
                fill_nan = True
            else:
                trls = min_trials
                fill_nan = False
            arr_form_i = na.array_format(org, trls, fill_nan=fill_nan,
                                         with_replace=with_replace)
            print('{} / {}'.format(i+1, resample))
            arr_form_i = np.moveaxis(arr_form_i, 2, -1)
            arr_form_mean = np.nanmean(arr_form_i, axis=0)
            with u.HiddenPrints():
                d_ = dPCA.dPCA(cond_labels, regularizer='auto')
                d_.protect = ['t']
                d_fit = d_.fit_transform(arr_form_mean, trialX=arr_form_i)
                out = d_.significance_analysis(arr_form_mean, arr_form_i,
                                               axis=True, full=True)
                s_mask, scores, shuff_scores = out
                ev = d_.explained_variance_ratio_
            fits.append((d_fit, s_mask, scores, shuff_scores, ev))
        dpca_fits[k] = fits
    if not pop:
        dpca_fits = dpca_fits['all']
    return orgs, dpca_fits, xs

def plot_dpca_kernels(dpca, xs, axs_keys, dim_dict=0, arr_labels=None,
                      signif_level=.01, color_dict=None, style_dict=None,
                      signif_heights=None, signif_height_default=5,
                      task_ind=None):
    kerns, _, scores, shuff_scores, ev = dpca
    if signif_heights is None:
        signif_heights = {k[0]:signif_height_default for k in axs_keys}
    for (k, ax) in axs_keys:
        print(kerns['t'].shape)
        dim = dim_dict[k]
        if k in shuff_scores.keys():
            num_shuffs = len(shuff_scores[k])
            score = np.expand_dims(scores[k][dim], 0)
            shuff_score = np.array(list(shuff_scores[k][i][dim]
                                        for i in range(num_shuffs)))
            pval = 1 - np.sum(score > shuff_score, axis=0)/num_shuffs
        else:
            pval = np.ones_like(xs)
        # kern = kerns[k][dim]
        kern = kerns[k][0]
        kern2 = kerns[k][1]
        if task_ind is not None:
            kern = kern[task_ind]
            kern2 = kern2[task_ind]
        sig = pval < signif_level
        ax_shape = kern.shape[:-1]
        ind_combs = itertools.product(*(range(x) for x in ax_shape))
        
        for ic in ind_combs:
            k1 = kern[ic]
            k2 = kern2[ic]
            if color_dict is not None:
                color = color_dict[k]
            else:
                color = None
            if style_dict is not None:
                style = style_dict[k][ic]
            else:
                style=None
            # l = gpl.plot_trace_werr(xs, k1, ax=ax, color=color, linestyle=style)
            l = gpl.plot_trace_werr(k1, k2, ax=ax, color=color, linestyle=style)
        xs_sig = xs[sig]
        ys_sig = np.ones_like(xs_sig)*signif_heights[k]
        # ax.plot(xs_sig, ys_sig, 'o', color=l[0].get_color())

def _compute_latency_across_dim(scores, shuffs, xs, signif_level=.01,
                                n_consecutive=5):
        num_shuffs = len(shuffs)
        latencies = np.zeros(scores.shape[0])
        for dim in range(scores.shape[0]):
            score = np.expand_dims(scores[dim], 0)
            shuff_score = np.array(list(shuffs[i][dim]
                                        for i in range(num_shuffs)))
            pval = 1 - np.sum(score > shuff_score, axis=0)/num_shuffs
            signif_mask = pval < signif_level
            m = np.ones(n_consecutive)
            out_conv = np.convolve(signif_mask, m, mode='valid')
            out_mask = out_conv == n_consecutive
            if np.any(out_mask):
                first_ind = np.argmax(out_mask)
                latencies[dim] = xs[first_ind]
            else:
                latencies[dim] = np.nan
        if np.all(np.isnan(latencies)):
            min_latency = np.nan
        else:
            min_latency = np.nanmin(latencies)
        return min_latency            
            
def compute_dpca_latencies(dpcas, xs, key, signif_level=.01,
                           n_consecutive=5):
    latencies = np.zeros(len(dpcas))
    for i, dpca in enumerate(dpcas):
        kerns, _, scores, shuff_scores, ev = dpca
        latencies[i] = _compute_latency_across_dim(scores[key],
                                                   shuff_scores[key], xs,
                                                   signif_level=signif_level,
                                                   n_consecutive=n_consecutive)        
    return latencies

def compute_dpca_ev(dpcas):
    evs = np.zeros(len(dpcas))
    for i, dpca in enumerate(dpcas):
        ev = dpca[-1]
        for v in ev.values():
            evs[i] += np.sum(v)
    return evs

def plot_dpca_kernels_resample(dpcas, xs, axs_keys, dim=0, arr_labels=None):
    for (k, ax) in axs_keys:
        kerns = [d[0][k][dim] for d in dpcas]
        full_arr = np.stack(kerns, axis=0)
        ax_shape = full_arr.shape[1:-1]
        ind_combs = itertools.product(*(range(x) for x in ax_shape))
        full_arr = np.swapaxes(full_arr, 0, len(ax_shape))
        for ic in ind_combs:
            gpl.plot_trace_werr(xs, full_arr[ic], ax=ax,
                                error_func=gpl.conf95_interval)

def organize_salience_decoding(data, dfunc_group, mf, start, end, binsize,
                               binstep, min_trials, bhv_extract_func,
                               cond_labels=None,
                               dfunc_pairs=None, min_spks=5, resample=100,
                               shuff_labels=False, kernel='rbf',
                               model=svm.SVR, with_replace=False, penalty=1,
                               leave_out=4, collapse_time=False, zscore=True,
                               pop=False, causal_timing=True, equal_fold=False,
                               min_population=1, collapse_time_zscore=False,
                               **kwargs):
    org_min_trials = min_trials
    if dfunc_pairs is None:
        dfunc_pairs = (0,)*len(dfunc_group)
    else:
        dfp = np.array(dfunc_pairs)
        org_min_trials = tuple(2*min_trials/np.sum(dfp == x) for x in dfunc_pairs)
    num_groups = len(np.unique(dfunc_pairs))
    if cond_labels is None:
        cond_labels = tuple(range(num_groups))

    mfs = (mf,)*len(dfunc_group)
    ctzs = collapse_time_zscore
    if pop:
        out = na.organize_spiking_data_pop(data, dfunc_group, mfs,
                                           start, end, binsize,
                                           binstep=binstep,
                                           min_trials=org_min_trials,
                                           min_spks=None, zscore=zscore,
                                           causal_timing=causal_timing,
                                           collapse_time_zscore=ctzs,
                                           bhv_extract_func=bhv_extract_func,
                                           **kwargs)
    else:
        out = na.organize_spiking_data(data, dfunc_group, mfs,
                                       start, end, binsize, binstep=binstep,
                                       min_trials=org_min_trials,
                                       min_spks=min_spks, zscore=zscore,
                                       causal_timing=causal_timing,
                                       collapse_time_zscore=ctzs,
                                       bhv_extract_func=bhv_extract_func,
                                       **kwargs)
    dat, xs, bhv = out
    org = tuple([] for x in np.unique(dfunc_pairs))
    tuple(org[dfunc_pairs[i]].append(x) for i, x in enumerate(dat))
    org_bhv = tuple([] for x in np.unique(dfunc_pairs))
    tuple(org_bhv[dfunc_pairs[i]].append(b) for i, b in enumerate(bhv))
    dec = {}
    if zscore or (ctzs and collapse_time):
        norm = False
    else:
        norm = True
    if collapse_time and zscore:
        print('data were zscored in each time bin, but time was collapsed')
    for i, ds in enumerate(org):
        bhv = org_bhv[i]
        n_cs = len(ds)
        out = na.svm_regression(ds, bhv, require_trials=min_trials,
                                resample=resample, leave_out=leave_out,
                                shuff_labels=shuff_labels, multi_cond=True,
                                kernel=kernel, model=model, pop=pop,
                                with_replace=with_replace,
                                equal_fold=equal_fold, norm=norm,
                                collapse_time=collapse_time,
                                min_population=min_population)
        dec[cond_labels[i]] = out
    return org, dec, xs
        

def organize_svm_pairs_prelim(data, dfunc_group, mf, start, end, binsize,
                              binstep, min_trials, cond_labels=None,
                              dfunc_pairs=None, min_spks=5, collapse_time=False,
                              zscore=True, pop=False, causal_timing=True,
                              collapse_time_zscore=False, cross_dec=False,
                              full_mfs=False, **kwargs):
    num_pairs = int(np.floor(len(dfunc_group)/2))
    if len(dfunc_group) > 2 and dfunc_pairs is None:
        dfunc_pairs = tuple(int(np.floor(i/2)) for i in range(2*num_pairs))
    if cross_dec:
        org_min_trials = min_trials
    else:
        dfp = np.array(dfunc_pairs)
        org_min_trials = tuple(2*min_trials/np.sum(dfp == x)
                               for x in dfunc_pairs)
    if cond_labels is None:
        if cross_dec:
            cond_labels = tuple((0, 1) for i in range(num_pairs))
        else:
            cond_labels = tuple(range(num_pairs))
    if not full_mfs:
        mfs = (mf,)*len(dfunc_group)
    else:
        mfs = mf
    ctzs = collapse_time_zscore
    if pop:
        out = na.organize_spiking_data_pop(data, dfunc_group, mfs,
                                           start, end, binsize,
                                           binstep=binstep,
                                           min_trials=org_min_trials,
                                           min_spks=None, zscore=zscore,
                                           causal_timing=causal_timing,
                                           collapse_time_zscore=ctzs,
                                           **kwargs)
    else:
        out = na.organize_spiking_data(data, dfunc_group, mfs,
                                       start, end, binsize, binstep=binstep,
                                       min_trials=org_min_trials,
                                       min_spks=min_spks, zscore=zscore,
                                       causal_timing=causal_timing,
                                       collapse_time_zscore=ctzs,
                                       **kwargs)
    return out, dfunc_pairs, cond_labels

def organized_decoding(dat, dfunc_pairs, cond_labels, require_trials=20,
                       norm=True, resample=20, leave_out=4, shuff_labels=False,
                       kernel='linear', model=svm.SVC, pop=False,
                       with_replace=False, equal_fold=False, cross_dec=False,
                       collapse_time=False, min_population=1,
                       use_avail_trials=False):
    org = tuple([] for x in np.unique(dfunc_pairs))
    tuple(org[dfunc_pairs[i]].append(x) for i, x in enumerate(dat))
    dec = {}
    for i, ds in enumerate(org):
        n_cs = len(ds)
        c1 = ds[:int(n_cs/2)]
        c2 = ds[int(n_cs/2):]
        req_trls = int(np.floor(require_trials/min(len(c1), len(c2))))
        if not cross_dec:
            out = na.svm_decoding(c1, c2, require_trials=req_trls,
                                  resample=resample, leave_out=leave_out,
                                  shuff_labels=shuff_labels, multi_cond=True,
                                  kernel=kernel, model=model, pop=pop,
                                  with_replace=with_replace,
                                  use_avail_trials=use_avail_trials,
                                  equal_fold=equal_fold, norm=norm,
                                  collapse_time=collapse_time,
                                  min_population=min_population)
            dec[cond_labels[i]] = out
        else:
            d1_label, d2_label = cond_labels[i]
            label1 = '{} -> {}'.format(d1_label, d2_label)
            train_c1, test_c1 = c1
            train_c2, test_c2 = c2
            out = na.svm_cross_decoding(train_c1, test_c1, train_c2, test_c2,
                                        require_trials=require_trials,
                                        resample=resample, leave_out=leave_out,
                                        shuff_labels=shuff_labels,
                                        use_avail_trials=use_avail_trials,
                                        kernel=kernel, model=model, pop=pop,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold, norm=norm,
                                        collapse_time=collapse_time,
                                        min_population=min_population)
            dec[label1] = out
            
            label2 = '{} -> {}'.format(d2_label, d1_label)
            out = na.svm_cross_decoding(test_c1, train_c1, test_c2, train_c2,
                                        require_trials=require_trials,
                                        resample=resample, leave_out=leave_out,
                                        shuff_labels=shuff_labels, 
                                        kernel=kernel, model=model, pop=pop,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold, norm=norm,
                                        collapse_time=collapse_time,
                                        min_population=min_population)
            dec[label2] = out
    return dec, org

def organize_svm_pairs(data, dfunc_group, mf, start, end, binsize, binstep,
                       min_trials, cond_labels=None, dfunc_pairs=None,
                       min_spks=5, resample=20, shuff_labels=False,
                       kernel='linear', model=svm.SVC, with_replace=False,
                       penalty=1, leave_out=4, collapse_time=False, zscore=True,
                       pop=False, causal_timing=True, equal_fold=False,
                       min_population=1, cross_dec=False,
                       collapse_time_zscore=False, **kwargs):
    out = organize_svm_pairs_prelim(data, dfunc_group, mf, start, end, binsize,
                                    binstep, min_trials, cond_labels=cond_labels,
                                    dfunc_pairs=dfunc_pairs, min_spks=min_spks,
                                    collapse_time=collapse_time, zscore=zscore,
                                    pop=pop, causal_timing=causal_timing,
                                    collapse_time_zscore=collapse_time_zscore,
                                    cross_dec=cross_dec, **kwargs)
    (dat, xs), dfunc_pairs, cond_labels = out 
    if collapse_time and zscore:
        print('data were zscored in each time bin, but time was collapsed')
    if zscore or (collapse_time_zscore and collapse_time):
        norm = False
    else:
        norm = True
    dec, org = organized_decoding(dat, dfunc_pairs, cond_labels,
                                  require_trials=min_trials,
                                  norm=norm, cross_dec=cross_dec,
                                  resample=resample, leave_out=leave_out,
                                  shuff_labels=shuff_labels, kernel=kernel,
                                  model=model, pop=pop,
                                  with_replace=with_replace,
                                  equal_fold=equal_fold,
                                  collapse_time=collapse_time,
                                  min_population=min_population)
    return org, dec, xs

def print_decoding_info(decs):
    for m, (org, _, _) in decs.items():
        for i, dec_org in enumerate(org):
            n_neurs = len(dec_org[0].keys())
            print('{}: {} neurons in condition {}'.format(m, n_neurs, i+1))
        

def combine_svm_format(og_dict, nk='combined'):
    new_dict = {}
    for i, (k, v) in enumerate(og_dict.items()):
        (dat, xs), dp, cl = v
        new_dat = []
        for d in dat:
            n_neurdict = {(k, nk):d[nk] for nk in d.keys()}
            new_dat.append(n_neurdict)
        if i == 0:
            new_dict[nk] = (new_dat, xs), dp, cl
        else:
            prev = new_dict[nk][0][0]
            for j, nd in enumerate(new_dat):
                prev[j].update(nd)
    return new_dict
            
def plot_neuron_scatters(data, pt, ax=None, central_tend=np.nanmean,
                         error_func=gpl.sem, boots=None, lims=.6,
                         l_color=(.8, .8, .8), min_zero=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw=dict(aspect='equal'))
    ((d1, d2), xs) = data
    x_ind = np.argmin(np.abs(xs - pt))
    pts_d = []
    for k, spks1 in d1.items():
        spks2 = d2[k]
        if boots is not None:
            f = lambda x: central_tend(x, axis=0)
            spks1 = u.bootstrap_list(spks1, f, n=boots,
                                     out_shape=spks1.shape[1:])
            spks2 = u.bootstrap_list(spks2, f, n=boots,
                                     out_shape=spks2.shape[1:])
            error_func = gpl.conf95_interval
        pts_d.append((spks1[:, x_ind:x_ind+1], spks2[:, x_ind:x_ind+1]))
        gpl.plot_trace_werr(spks1[:, x_ind:x_ind+1], spks2[:, x_ind:x_ind+1],
                            ax=ax, fill=False, central_tendency=central_tend,
                            error_func=error_func, points=True, **kwargs)
    if lims is not None:
        if min_zero:
            lim_tup = (0, lims)
        else:
            lim_tup = (-lims, lims)
    else:
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        lim_tup = (min(xl[0], yl[0]), max(xl[1], yl[1]))
    ax.plot(lim_tup, lim_tup, color=l_color, linestyle='dashed')
    ax.set_xlim(lim_tup)
    ax.set_ylim(lim_tup)
    pts_d = np.array(pts_d)
    return ax, pts_d

def svm_decoding_helper(data_dict, min_trials=10, resample=100,
                        dec_pair_labels=None, shuffle=False,
                        kernel='linear', model=svm.SVC, **params):
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
                                  kernel=kernel, model=model, **params)
            out_dicts[pl][label] = dec
    return out_dicts

def plot_svm_pairs(res, plot_keys, axs, **kwargs):
    _, dec, xs = res
    for i, ks in enumerate(plot_keys):
        ax = axs[i]
        nd = {k:dec[k] for k in ks}
        plot_svm_decoding(nd, xs, ax=ax, **kwargs)
    return axs

def get_svm_trajectory(res, ax_names, resamples=100, pop=False, with_replace=True):
    ax_vectors = []
    inters = []
    for an in ax_names:
        vecs = res[1][an][3]
        vec = np.mean(np.mean(vecs, axis=1), axis=0)[0]
        ax_vectors.append(vec)
        intercepts = res[1][an][4]
        intercept = np.mean(np.mean(intercepts, axis=1), axis=0)[0]
        inters.append(intercept)
    org_trajs = []
    for i, org in enumerate(res[0]):
        data, lens = na.neural_format(org, pop=pop)
        d_samp = na.sample_trials_svm(data, resamples,
                                      with_replace=with_replace)
        trajs = np.zeros((len(org), resamples, len(ax_vectors),
                          d_samp.shape[3]))
        for j in range(len(org)):
            data_j = np.swapaxes(d_samp[:, j], 0, 1)
            trajs[j] = na.basis_trajectories(data_j, ax_vectors)
            trajs[j] = trajs[j] + np.expand_dims(inters, (0, 2))
        org_trajs.append(trajs)
    return org_trajs

def plot_svm_trajectory(cond_trajs, ax, alpha=.2, central_tend=np.nanmean,
                        plot_other=False):
    for trajs in cond_trajs:
        if plot_other:
            for traj in trajs:
                ax.plot(traj[0], traj[1], traj[2], alpha=alpha)
        cent_traj = central_tend(trajs, axis=0)
        ax.plot(cent_traj[0], cent_traj[1], cent_traj[2])
    return ax
        
def plot_svm_decoding(results_dict, xs, figsize=None, colordict=None, ax=None,
                      legend=True, shuffled_results=None,
                      legend_text=None, boots=1000):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    for k, data in results_dict.items():
        if colordict is not None:
            color = colordict[k]
        else:
            color = None
        if legend and legend_text is None:
            label = str(k)
        elif legend and legend_text is not None:
            label = legend_text
        else:
            label = ''
        tcs = data[0]
        ef = gpl.conf95_interval
        if shuffled_results is not None:
            tcs_shuff = shuffled_results[k][0]
            tcs = tcs - np.expand_dims(np.mean(tcs_shuff, axis=0), 0)
        gpl.plot_trace_werr(xs, tcs, color=color, ax=ax, label=label,
                            error_func=ef)
    return ax

def plot_svm_decoding_angs(decs, pt, ax, rand_pt=0, cross_color=None,
                           wi_color=None, rand_color=None):
    for m, (org, dec, xs) in decs.items():
        assert len(dec) == 2
        k1, k2 = list(dec.keys())
        _, ms1, inter1 = dec[k1]
        _, ms2, inter2 = dec[k2]
        ms1 = np.mean(ms1, axis=1)
        ms2 = np.mean(ms2, axis=1)
        x_pt = np.argmin(np.abs(xs - pt))
        min_trls = min(ms1.shape[0], ms2.shape[0])
        angs = [u.vector_angle(ms1[i, :, x_pt], ms2[i, :, x_pt])
                for i in range(min_trls)]
        wi_ind_pairs = list(itertools.product(range(min_trls), repeat=2))
        angs_wi1 = list(u.vector_angle(ms1[i, :, x_pt], ms1[j, :, x_pt])
                        for (i, j) in wi_ind_pairs)
        angs_wi2 = list(u.vector_angle(ms2[i, :, x_pt], ms2[j, :, x_pt])
                        for (i, j) in wi_ind_pairs)
        angs_wi = angs_wi1 + angs_wi2
        angs_rand =  [u.vector_angle(ms1[i, :, rand_pt], ms2[i, :, rand_pt])
                      for i in range(min_trls)]
        ax.hist(angs, histtype='step', density=True, color=cross_color)
        ax.hist(angs_wi, histtype='step', density=True, color=wi_color)
        ax.hist(angs_rand, histtype='step', density=True, color=rand_color)
        gpl.clean_plot(ax, 0)
        
def plot_svm_decoding_point(results_dict, xs, pt, figsize=None, colordict=None,
                            ax=None, color=None, legend=True, tick_labels=False,
                            shuffled_results=None, legend_text=None,
                            ax_offset=.15, pb_color=(.4, .4, .4), pb_width=1.5,
                            pb_ax=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    x_ind = np.argmin(np.abs(xs - pt))
    n_pts = len(results_dict)
    pts = []
    xs_pts = []
    labels = []
    y_pt_list = []
    for i, (k, data) in enumerate(results_dict.items()):
        if colordict is not None:
            color = colordict[k]
        if legend and legend_text is None:
            label = str(k)
        elif legend and legend_text is not None:
            label = legend_text
        else:
            label = ''
        tcs = data[0]
        if shuffled_results is not None:
            tcs_shuff = shuffled_results[k][0]
            tcs = tcs - np.expand_dims(np.mean(tcs_shuff, axis=0), 0)
        xs_pts.append(i)
        pts.append(tcs[:, x_ind])
        gpl.plot_trace_werr(np.array([i]), tcs[:, x_ind:x_ind+1], color=color,
                            ax=ax, label=label, error_func=gpl.conf95_interval,
                            fill=False, points=True)
        y_pt_list.append(np.nanmean(tcs[:, x_ind]))
        labels.append(k)
    if pb_ax is not None:
        ymin = np.min(y_pt_list)
        ymax = np.max(y_pt_list)
        pb_ax.vlines(xs[x_ind], ymin, ymax, color=pb_color, linewidth=pb_width)
    if tick_labels:
        xs_arr = np.array(xs_pts)
        ax.set_xticks(xs_arr)
        ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim([xs_pts[0] - ax_offset, xs_pts[-1] + ax_offset])
    return ax, pts, x_ind

def _get_all_diffs(p1, p2):
    ds = np.squeeze(np.diff(list(itertools.product(p2, p1)), axis=1))
    return ds

def print_svm_decoding_diff(pts, text, labels):
    for m, pts in pts.items():
        t1 = text.format(labels[0])
        gpl.print_mean_conf95(pts[0], m, t1, preboot=True)
        t2 = text.format(labels[1])
        gpl.print_mean_conf95(pts[1], m, t2, preboot=True)           
        t3 = text.format('diff')
        ds = _get_all_diffs(pts[0], pts[1])
        gpl.print_mean_conf95(ds, m, t3, preboot=True)

def print_svm_scatter(pts, text, combine=True):
    pd_all = np.zeros((0, 1))
    for m, pts in pts.items():
        pts = np.array(pts)
        pts_mean = np.mean(pts[..., 0], axis=2)
        pts_diff = -np.diff(pts_mean, axis=1)
        pd_all = np.concatenate((pd_all, pts_diff), 0)
        gpl.print_mean_conf95(pts_mean[:, 0], m, text + ' sDMST')
        gpl.print_mean_conf95(pts_mean[:, 1], m, text + ' PLT')
        gpl.print_mean_conf95(pts_diff, m, text)
    gpl.print_mean_conf95(pd_all, 'combined', text)
        
def plot_svm_decoding_scatter(r1, r2, xs, pt, figsize=None, 
                              ax=None, color=None, legend=True,
                              shuffled_results=None, legend_text=None,
                              central_tend=np.nanmean):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    tcs1 = r1[0]
    tcs2 = r2[0]
    x_ind = np.argmin(np.abs(xs - pt))
    x_pt_d = tcs1[:, x_ind:x_ind+1]
    y_pt_d = tcs2[:, x_ind:x_ind+1]
    l = gpl.plot_trace_werr(x_pt_d, y_pt_d,
                            ax=ax, fill=False, central_tendency=central_tend,
                            error_func=gpl.conf95_interval, points=True,
                            color=color)
    pt = (x_pt_d, y_pt_d)
    return pt
                        
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
