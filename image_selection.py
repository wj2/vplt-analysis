
import numpy as np
import pystan as ps
import pickle
import os
import re
import matplotlib.pyplot as plt
import multiprocessing as mp

import general.plotting as gpl
import pref_looking.plt_analysis as pl

model_path = 'pref_looking/stan_models/image_selection.pkl'
model_path_notau = 'pref_looking/stan_models/image_selection_notau.pkl'

def store_models(model_collection):
    new_collection = {}
    for k, (fit, params, diags) in model_collection.items():
        new_fit = ModelFitContainer(fit)
        new_collection[k] = (new_fit, params, diags)
    return new_collection

def get_stan_params(mf, param, mask=None, skip_end=1):
    names = mf.flatnames
    means = mf.get_posterior_mean()[:-skip_end]
    param = '\A' + param
    par_mask = np.array(list([re.match(param, x) is not None for x in names]))
    par_means = means[par_mask]
    if mask is not None:
        par_means = par_means[mask]
    return par_means

def get_novfam_sal_diff(fit, fit_params, param='s.*', central_func=np.mean,
                        sal_central_func=np.mean, nov_val=1, fam_val=0,
                        fam_field='img_cats'):
    nov_mask = fit_params[fam_field] == nov_val
    fam_mask = fit_params[fam_field] == fam_val
    nov_sals = get_stan_params(fit, param, mask=nov_mask)
    fam_sals = get_stan_params(fit, param, mask=fam_mask)
    nov_means = sal_central_func(nov_sals, axis=0)
    fam_means = sal_central_func(fam_sals, axis=0)
    avg_diff = central_func(nov_means - fam_means)
    return avg_diff

def get_bias_diff(fit, fit_params, param='bias.*', central_func=np.mean):
    both_bias = get_stan_params(fit, param)
    bias_diff = -np.diff(both_bias, axis=0)
    return central_func(bias_diff)

def get_nov_bias(fit, fit_params, param='eps.*', central_func=np.mean):
    eps = get_stan_params(fit, param)
    return central_func(eps)

def get_full_nov_effect(fit, fit_params, sal_param='s.*', bias_param='eps.*'):
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
    cc = np.corrcoef(stan_vals, analy_vals)
    ax.plot(stan_vals, analy_vals, 'o')
    rc = np.round(cc[1,0]**2, 2)
    ax.set_title(r'$R^{2} = '+'{}$'.format(rc))
    return stan_vals, analy_vals

def plot_stan_models(model_dict, f=None, fsize=(12,4), lw=10, chains=4,
                     nov_param='eps.*', bias_param='bias.*', sal_param='s.*',
                     lil=.1):
    if f is None:
        f = plt.figure(figsize=fsize)
    ax_sal = f.add_subplot(2, 1, 1)
    ax_par = f.add_subplot(2, 1, 2)
    nov_col = None
    fam_col = None
    nov_fits = np.zeros((len(model_dict), chains))
    bias_fits = np.zeros((len(model_dict), 2, chains))
    expsal_fits = np.zeros((len(model_dict), chains))
    for i, (k, v) in enumerate(model_dict.items()):
        fit, params, diags = v
        nov_sals = get_stan_params(fit, sal_param, params['img_cats'] == 1)
        nov_sals = np.expand_dims(np.mean(nov_sals, axis=1), axis=0).T
        fam_sals = get_stan_params(fit, sal_param, params['img_cats'] == 0)
        fam_sals = np.expand_dims(np.mean(fam_sals, axis=1), axis=0).T
        xs = np.array([i])
        nl = gpl.plot_trace_werr(xs - lil, nov_sals, ax=ax_sal, linewidth=lw,
                                 color=nov_col, error_func=gpl.conf95_interval)
        fl = gpl.plot_trace_werr(xs + lil, fam_sals, ax=ax_sal, linewidth=lw,
                                 color=fam_col, error_func=gpl.conf95_interval)
        nov_col = nl[0].get_color()
        fam_col = fl[0].get_color()

        nov_fits[i] = get_stan_params(fit, param=nov_param)
        bias_fits[i] = get_stan_params(fit, param=bias_param)
        expsal_fits[i] = np.mean(np.abs(np.concatenate((nov_sals, fam_sals))))
    xs = np.arange(len(model_dict))
    gpl.plot_trace_werr(xs, nov_fits.T, ax=ax_par, label='novel')
    l = gpl.plot_trace_werr(xs, bias_fits[:, 0, :].T, ax=ax_par,
                            label='bias 1')
    col = l[0].get_color()
    gpl.plot_trace_werr(xs, bias_fits[:, 1, :].T, ax=ax_par, label='bias 2',
                        color=col)
    gpl.plot_trace_werr(xs, expsal_fits.T, ax=ax_par, label='salience')
    return f
        
def recompile_model(mp=model_path):
    p, ext = os.path.splitext(mp)
    stan_path = p + '.stan'
    sm = ps.StanModel(file=stan_path)
    pickle.dump(sm, open(mp, 'wb'))
    return mp

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

def _swap_string_for_levels(arr, mapping=None):
    types = np.unique(arr)
    lvl_arr = np.zeros_like(arr, dtype=int)
    back_mapping = {}
    forw_mapping = {}
    for i, t in enumerate(types):
        if mapping is not None:
            lvl_arr[arr == t] = mapping[t]
        else:
            lvl_arr[arr == t] = i
        back_mapping[i] = t
        forw_mapping[t] = i
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
                           run_field='datanum', **params):
    data = data[constraint_func(data)]
    runs = np.unique(data[run_field])
    run_dict = {}
    analysis_dict = {}
    for i, run in enumerate(runs):
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
                               rc='rightimg_type',
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
                  'views':views, 'y':outcomes, 'img_cats':mapped_cats}
    return param_dict, mappings

class ModelFitContainer(object):

    def __init__(self, fit):
        self.flatnames = fit.flatnames
        self._posterior_means = fit.get_posterior_mean()
        self.samples = fit.extract()
        self._summary = fit.stansummary()

    def get_posterior_mean(self):
        return self._posterior_means

    def stansummary(self):
        return self._summary

    def __repr__(self):
        return self._summary
