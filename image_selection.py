
import numpy as np
import pystan as ps
import pickle
import os
import re
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools as it

import general.plotting as gpl
import general.stan_utility as su
import pref_looking.plt_analysis as pl

model_path = 'pref_looking/stan_models/image_selection.pkl'
model_path_notau = 'pref_looking/stan_models/image_selection_notau.pkl'
model_path_notau_eps = 'pref_looking/stan_models/image_selection_notau_eps.pkl'
model_path_notau_cat = 'pref_looking/stan_models/image_selection_notau_cat.pkl'
model_path_nt_all = 'pref_looking/stan_models/image_selection_nt_all.pkl'

def get_novfam_sal_diff(fit, fit_params, param='s\[.*', central_func=np.mean,
                        sal_central_func=np.mean, nov_val=1, fam_val=0,
                        fam_field='img_cats'):
    nov_mask = fit_params[fam_field] == nov_val
    fam_mask = fit_params[fam_field] == fam_val
    nov_sals = su.get_stan_params(fit, param, mask=nov_mask)
    fam_sals = su.get_stan_params(fit, param, mask=fam_mask)
    nov_means = sal_central_func(nov_sals, axis=0)
    fam_means = sal_central_func(fam_sals, axis=0)
    avg_diff = central_func(nov_means - fam_means)
    return avg_diff

def get_bias_diff(fit, fit_params, param='bias\[.*', central_func=np.mean,
                  lim=np.inf):
    both_bias = su.get_stan_params(fit, param)
    bias_diff = -np.diff(both_bias, axis=0)
    ret = central_func(bias_diff)
    if np.abs(ret) > lim:
        ret = np.nan
    return ret

def get_nov_bias(fit, fit_params, param='eps.*', central_func=np.mean,
                 lim=np.inf):
    eps = su.get_stan_params(fit, param)
    ret = central_func(eps)
    if np.abs(ret) > lim:
        ret = np.nan
    return ret

def get_full_nov_effect(fit, fit_params, sal_param='s\[.*', bias_param='eps.*'):
    eps = get_nov_bias(fit, fit_params, param=bias_param)
    avg_sdiff = get_novfam_sal_diff(fit, fit_params, param=sal_param)
    return eps + avg_sdiff

def stan_scatter_plot(model_dict, analysis_dict, func1, func2,
                      ax=None, func1_stan=True, func2_stan=False):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    stan_vals = np.zeros(len(model_dict))
    analy_vals = np.zeros_like(stan_vals)
    for i, (k, stan_v) in enumerate(model_dict.items()):
        fit, params, diags = stan_v
        analy_v = analysis_dict[k]
        if func1_stan:
            args1 = (fit, params)
        else:
            args1 = (analy_v,)
        if func2_stan:
            args2 = (fit, params)
        else:
            args2 = (analy_v,)
        stan_vals[i] = func1(*args1)
        analy_vals[i] = func2(*args2)
    nan_mask = (np.logical_not(np.isnan(stan_vals))
                *np.logical_not(np.isnan(analy_vals)))
    stan_vals = stan_vals[nan_mask]
    analy_vals = analy_vals[nan_mask]
    cc = np.corrcoef(stan_vals, analy_vals)
    ax.plot(stan_vals, analy_vals, 'o')
    rc = np.round(cc[1,0]**2, 2)
    ax.set_title(r'$R^{2} = '+'{}$'.format(rc))
    return stan_vals, analy_vals

def _remove_errs_models(model_dict, errfield, diag_ind=2):
    models = {}
    for k, v in model_dict.items():
        if errfield is not None:
            if v[diag_ind][errfield]:
                models[k] = v
        else:
            if np.product(list(v[diag_ind].values())):
                models[k] = v
    return models

def plot_stan_models(model_dict, f=None, fsize=(12,4), lw=10, chains=4,
                     nov_param='eps.*', bias_param='bias\[.*', sal_param='s\[.*',
                     lil=.1, sort_by_nov=True, sal_pairs=1000,
                     remove_errs=True, errfield=None, diag_ind=2):
    if f is None:
        f = plt.figure(figsize=fsize)
    if remove_errs:
        model_dict = _remove_errs_models(model_dict, errfield,
                                         diag_ind=diag_ind)
    ax_sal = f.add_subplot(2, 1, 1)
    ax_par = f.add_subplot(2, 1, 2)
    nov_col = None
    fam_col = None
    nov_fits = np.zeros((len(model_dict), chains))
    bias_fits = np.zeros((len(model_dict), 2, chains))
    expsal_fits = np.zeros((len(model_dict), chains))
    sal_diff = np.zeros((len(model_dict), chains))
    for i, (k, v) in enumerate(model_dict.items()):
        fit, params, diags = v
        nov_sals = su.get_stan_params(fit, sal_param, params['img_cats'] == 1)
        nov_sals = np.expand_dims(np.mean(nov_sals, axis=1), axis=0).T
        fam_sals = su.get_stan_params(fit, sal_param, params['img_cats'] == 0)
        fam_sals = np.expand_dims(np.mean(fam_sals, axis=1), axis=0).T
        xs = np.array([i])
        nl = gpl.plot_trace_werr(xs - lil, nov_sals, ax=ax_sal, linewidth=lw,
                                 color=nov_col, error_func=gpl.conf95_interval)
        fl = gpl.plot_trace_werr(xs + lil, fam_sals, ax=ax_sal, linewidth=lw,
                                 color=fam_col, error_func=gpl.conf95_interval)
        nov_col = nl[0].get_color()
        fam_col = fl[0].get_color()

        nov_fits[i] = su.get_stan_params(fit, param=nov_param)
        bias_fits[i] = su.get_stan_params(fit, param=bias_param)
        expsal_fits[i] = np.mean(np.abs(np.concatenate((nov_sals, fam_sals))))
        sals = _sample_pairs(np.concatenate((nov_sals, fam_sals)),
                             n=sal_pairs)
        sal_diff[i] = np.mean(sals)
    xs = np.arange(len(model_dict))
    nov_mag = np.abs(nov_fits)
    bias_diff = np.abs(np.diff(bias_fits, axis=1))
    norm = nov_mag + bias_diff[:, 0] + sal_diff
    if sort_by_nov:
        sort_inds = np.argsort(np.mean(nov_mag/norm, axis=1))
    norm = norm[sort_inds].T
    gpl.plot_trace_werr(xs, nov_mag[sort_inds].T/norm, ax=ax_par, label='novel',
                        marker='o')
    gpl.plot_trace_werr(xs, bias_diff[sort_inds, 0].T/norm, ax=ax_par,
                        label='bias diff', marker='o')
    gpl.plot_trace_werr(xs, sal_diff[sort_inds].T/norm, ax=ax_par, marker='o',
                        label='salience')
    return f

def _sample_pairs(sals, n=500):
    all_pairs = list(it.combinations(range(len(sals)), 2))
    permuted_pairs = np.random.permutation(all_pairs)
    dists = np.zeros(n)
    for i, (s1, s2) in enumerate(permuted_pairs[:n]):
        dists[i] = np.abs(sals[s1] - sals[s2])
    return dists

def fit_run_models(run_dict, model_path=model_path, prior_dict=None,
                   stan_params=None, parallel=False):
    sm = pickle.load(open(model_path, 'rb'))
    out_models = {}
    if stan_params is None:
        stan_params = {}
    if prior_dict is None:
        prior_dict = {}
    args_list = []
    for i, run_k in enumerate(run_dict.keys()):
        params, mappings = run_dict[run_k]
        args_list.append((run_k, params, prior_dict, stan_params, sm,
                          parallel))
    if parallel:
        try:
            pool = mp.Pool(processes=mp.cpu_count())
            out_list = pool.map(_map_stan_fitting, args_list)
        finally:
            pool.close()
            pool.join()
    else:
        out_list = map(_map_stan_fitting, args_list)
    out_models = dict(out_list)
    return sm, out_models

def _map_stan_fitting(args):
    run_i, params, prior_dict, stan_params, sm, parallel = args
    if parallel:
        n_jobs = 1
    else:
        n_jobs = -1
    params.update(prior_dict)
    fit = sm.sampling(data=params, n_jobs=n_jobs, **stan_params)
    diags = ps.diagnostics.check_hmc_diagnostics(fit)
    return run_i, (fit, params, diags)

def _swap_string_for_levels(arr, mapping=None, nonzero=False):
    types = np.unique(arr)
    lvl_arr = np.zeros_like(arr, dtype=int)
    back_mapping = {}
    forw_mapping = {}
    for i, t in enumerate(types):
        if mapping is not None:
            lvl_rpl = mapping[t]
        elif nonzero:
            lvl_rpl = i + 1
        else:
            lvl_rpl = i 
        lvl_arr[arr == t] = lvl_rpl
        back_mapping[lvl_rpl] = t
        forw_mapping[t] = lvl_rpl
    return lvl_arr, back_mapping, forw_mapping

def _get_common_swap(arrs):
    total_arrs = np.concatenate(arrs)
    lvl_arr, bm, fm = _swap_string_for_levels(total_arrs)
    output_arrs = []
    start_ind = 0
    for arr in arrs:
        end_ind = start_ind + arr.shape[0]
        lvl_a = lvl_arr[start_ind:end_ind]
        output_arrs.append(lvl_a)
        start_ind = end_ind
    return output_arrs, bm, fm

def generate_stan_datasets(data, constraint_func, conds=None,
                           run_field='datanum', collapse=False,
                           **params):
    data = data[constraint_func(data)]
    runs = np.unique(data[run_field])
    run_dict = {}
    analysis_dict = {}
    if collapse:
        runs = ('all',)
    for i, run in enumerate(runs):
        if collapse:
            data_run = data
        else:
            data_run = data[data[run_field] == run]
        out = format_predictors_outcomes(data_run, **params)
        run_dict[run] = out
        if conds is not None:
            fsp = pl.get_first_saccade_prob(data_run, conds)
        else:
            fsp = None
        side_bias = pl.get_side_bias(data_run)
        analysis_dict[run] = (fsp, side_bias)
    return run_dict, analysis_dict

outcome_mapping = {b'l':1, b'r':2, b'o':3}
def format_predictors_outcomes(data, outcome='first_look', li='leftimg',
                               ri='rightimg', lv='leftviews',
                               rv='rightviews', lc='leftimg_type',
                               rc='rightimg_type', drunfield='datafile',
                               outcome_mapping=outcome_mapping,
                               outcome_types=(b'l', b'r', b'o')):
    outcomes = data[outcome]
    valid_outcome_mask = np.isin(outcomes, outcome_types)
    outcomes = outcomes[valid_outcome_mask]
    n = len(outcomes)
    k = len(outcome_types)
    mappings = {}
    out = _swap_string_for_levels(outcomes, mapping=outcome_mapping)
    outcomes, outcome_bm, outcome_fm = out
    mappings['outcomes'] = (outcome_bm, outcome_fm)
    data = data[valid_outcome_mask]

    # days array
    days = data[drunfield]
    out = _swap_string_for_levels(days, nonzero=True)
    days, days_bm, days_fm = out
    mappings['day'] = (days_bm, days_fm)
    n_days = len(np.unique(days))
    
    # image array
    out = _get_common_swap((data[li], data[ri]))
    l = len(out[1].keys())
    imgs = np.zeros((n, l, k), dtype=int)
    inds1 = (np.arange(imgs.shape[0], dtype=int), out[0][0], (0,)*n)
    imgs[inds1[0], inds1[1], inds1[2]] = 1
    inds2 = (np.arange(imgs.shape[0], dtype=int), out[0][1], (1,)*n)
    imgs[inds2[0], inds2[1], inds2[2]] = 1
    img_bm, img_fm = out[1], out[2]
    mappings['img'] = (img_bm, img_fm)
    
    # novelty indicator array
    novs = np.zeros((n, k))
    out = _get_common_swap((data[lc], data[rc]))
    novs[:, 0], novs[:, 1] = out[0]
    nov_bm, nov_fm = out[1], out[2]
    mappings['nov'] = (nov_bm, nov_fm)

    # img novelty
    all_imgs = np.concatenate((data[li], data[ri]))
    all_cats = np.concatenate((data[lc], data[rc]))
    unique_imgs, img_inds = np.unique(all_imgs, return_index=True)
    img_cats = all_cats[img_inds]
    mapped_cats = np.array(list([nov_fm[x] for x in img_cats]))

    # number of views array
    views = np.zeros((n, k))    
    views[:, 0] = data[lv]
    views[:, 1] = data[rv]

    param_dict = {'N':n, 'K':k, 'L':l, 'imgs':imgs, 'novs':novs,
                  'views':views, 'y':outcomes, 'img_cats':mapped_cats,
                  'day':days, 'D':n_days}
    return param_dict, mappings
