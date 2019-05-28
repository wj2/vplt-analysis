
import numpy as np
import pystan as ps
import pickle
import os
import matplotlib.pyplot as plt

import general.plotting as gpl

model_path = 'pref_looking/stan_models/image_selection.pkl'
model_path_notau = 'pref_looking/stan_models/image_selection_notau.pkl'


def plot_stan_model(outmodel, f=None, fsize=(12, 4), lw=5):
    if f is None:
        f = plt.figure(figsize=fsize)
    fit, params, diags = outmodel
    means = fit.get_posterior_mean()
    names = fit.flatnames
    it = range(len(names))
    sal_mask = list(filter(lambda i: names[i][0] == 's', it))
    img_nov_mask = params['img_cats'] == 1
    img_fam_mask = params['img_cats'] == 0
    sals = means[sal_mask]
    biases = means[sal_mask[-1]+1:sal_mask[-1]+3]
    novelty = means[sal_mask[-1]+3]
    tau_nov = means[-3]
    tau_img = means[-2]
    ax1 = f.add_subplot(1, 3, 1)
    ax2 = f.add_subplot(1, 3, 2)
    ax1.hist(np.mean(sals[img_nov_mask], axis=1), histtype='step',
             linewidth=lw, label='nov', density=True)
    ax1.hist(np.mean(sals[img_fam_mask], axis=1), histtype='step',
             linewidth=lw, label='fam', density=True)
    ax1.legend(frameon=False)
    ax1.set_xlabel('inferred salience')
    ax1.set_ylabel('number of images')
    mean_abs_sals = np.mean(np.abs(sals), axis=0)
    mean_bias = -np.diff(biases, axis=0)[0]
    param_list = (mean_abs_sals, novelty, mean_bias)
    param_label = (r'$E[|s|]$', r'$\epsilon_{nov}$', r'$\Delta b$')
    for i, par in enumerate(param_list):
        pair = np.array((min(par), max(par)))
        xs = np.array([i, i])
        gpl.plot_trace_werr(xs, pair, ax=ax2, linewidth=lw)
    gpl.clean_plot(ax1, 0)
    ax2.set_xticks(range(len(par)))
    ax2.set_xticklabels(param_label)
    ax2.set_xlim([-.5, i + .5])
    ax2.set_ylabel('weighting')

    views = np.arange(30)
    salience_nov = (np.mean(mean_abs_sals)*np.exp(-views/np.mean(tau_img))
                    + np.mean(novelty)*np.exp(-views/np.mean(tau_nov)))
    salience_fam = np.mean(mean_abs_sals)*np.exp(-views/np.mean(tau_img))
    ax3 = f.add_subplot(1, 3, 3)
    gpl.plot_trace_werr(views, salience_nov, ax=ax3, linewidth=lw,
                        label='novel')
    gpl.plot_trace_werr(views, salience_fam, ax=ax3, linewidth=lw,
                        label='familiar')
    ax3.set_xlabel('number of views')
    ax3.set_ylabel('average image weighting')
    return f 
    
def recompile_model(mp=model_path):
    p, ext = os.path.splitext(mp)
    stan_path = p + '.stan'
    sm = ps.StanModel(file=stan_path)
    pickle.dump(sm, open(mp, 'wb'))
    return mp

def fit_run_models(run_dict, model_path=model_path, prior_dict=None):
    sm = pickle.load(open(model_path, 'rb'))
    out_models = {}
    if prior_dict is None:
        prior_dict = {}
    for i, run_k in enumerate(run_dict.keys()):
        params, mappings = run_dict[run_k]
        params.update(prior_dict)
        fit = sm.sampling(data=params)
        diags = ps.diagnostics.check_hmc_diagnostics(fit)
        out_models[run_k] = (fit, params, diags)
    return out_models

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

def generate_stan_datasets(data, constraint_func, run_field='datanum',
                           **params):
    data = data[constraint_func(data)]
    runs = np.unique(data[run_field])
    run_dict = {}
    for i, run in enumerate(runs):
        data_run = data[data[run_field] == run]
        out = format_predictors_outcomes(data_run, **params)
        run_dict[run] = out
    return run_dict

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

