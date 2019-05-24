
import numpy as np
import pystan as ps
import pickle
import os

model_path = 'pref_looking/stan_models/image_selection.pkl'

def recompile_model(model_path=model_path):
    p, ext = os.path.splitext(model_path)
    stan_path = p + '.stan'
    sm = ps.StanModel(file=stan_path)
    pickle.dump(sm, open(model_path, 'wb'))
    return model_path

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

    # number of views array
    views = np.zeros((n, k))    
    views[:, 0] = data[lv]
    views[:, 1] = data[rv]

    param_dict = {'N':n, 'K':k, 'L':l, 'imgs':imgs, 'novs':novs,
                  'views':views, 'y':outcomes}
    return param_dict, mappings

