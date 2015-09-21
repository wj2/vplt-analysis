
import statsmodels.api as sm
import numpy as np

from utility import *

def organize_dim_data(dimdat, drunfield='datanum', neurfield='spike_times',
                      usefuncs=[]):
    """ 
    dimdat - data to use for glm
    usefuncs - list of functions that take a single trial and return one or 
      more columns in the exogenous matrix for that trial 
    """
    druns = get_data_run_nums(dimdat, drunfield)
    glmdat = []
    for dr in druns:
        drdat = dimdat[dimdat[drunfield] == dr]
        # prep indep vars
        if len(drdat[0][neurfield]) > 0:
            for i, tr in enumerate(drdat):
                for j, func in enumerate(usefuncs):
                    cols = func(tr, params)
                    if j == 0:
                        allcols = cols
                    else:
                        allcols = np.concatenate((allcols, cols), axis=1)
                # have all exog columns for single trial, need endog
                for j, neur in tr[neurfield]:
                    bn = bins_tr(neur, params)
                    if j == 0:
                        neurblock = bn
                    else:
                        neurblock = np.concatenate((neurblock, bn), axis=1)
                if i == 0:
                    endog = neurblock
                    exog = allcols
                else:
                    endog = np.concatenate((endog, neurblock), axis=0)
                    exog = np.concatenate((exog, allcols), axis=0)
            for i in xrange(len(tr[neurfield])):
                neurdic = {}
                neurdic['endog'] = endog[:, i]
                neurdic['exog'] = exog
                glmdat.append(neurdic)
    return glmdat

def extract_ar(trl, params):
    q_ar = params['q_ar']
    binsize = params['binsize']
    neurf = params['neurfield']
    for i, neur in enumerate(trl[neurf]):
        spkbins = bins_tr(neur, params)
        if i == 0:
            pass

def fit_glm(glmdic, link='log', 
            family=sm.families.Poisson(sm.families.links.log)):
    endog = glmdic['endog']
    exog = glmdic['exog']
    exog = sm.add_constant(exog)
    gpoiss = sm.GLM(endog, exog, family=family)
    gp_results = gpoiss.fit()
    return gp_results
                
