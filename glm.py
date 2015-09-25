
import statsmodels.api as sm
import numpy as np

from utility import *

def organize_dim_data(dimdat, drunfield='datanum', neurfield='spike_times',
                      usefuncs=None, params=None, tnumfield='trialnum', 
                      novfunc=views_to_member, tau1=50.):
    """ 
    dimdat - data to use for glm
    usefuncs - list of functions that take a single trial and return one or 
      more columns in the exogenous matrix for that trial 
    """
    if usefuncs is None:
        usefuncs = []
    if params is None:
        params = {'begtime':-200., 'endtime':0., 'binsize':5., 'q_ar':10,
                  'neurf':neurfield, 'winsize':10., 'endflag':'reward_time',
                  'lviewf':'leftviews', 'rviewf':'rightviews', 
                  'ltimef':'left_img_on', 'rtimef':'right_img_on', 
                  'centimgf_on':'centimgon', 'centimgf_off':'centimgoff',
                  'eyepos':'eyepos', 'tau1':tau1, 'tcodef':'code_numbers',
                  'tnumfield':'trial_type', 'novfunc':novfunc}
    druns = get_data_run_nums(dimdat, drunfield)
    glmdat = []
    glmlabels = ['const']
    beg = params['begtime']
    endadd = params['endtime']
    binsize = params['binsize']
    for k, dr in enumerate(druns):
        drdat = dimdat[dimdat[drunfield] == dr]
        # prep indep vars
        if len(drdat[0][neurfield]) > 0:
            for i, tr in enumerate(drdat):
                for j, func in enumerate(usefuncs):
                    cols, labels = func(tr, params)
                    if i == 0 and k == 0:
                        glmlabels = glmlabels + labels
                    if j == 0:
                        allcols = cols
                    else:
                        allcols = np.concatenate((allcols, cols), axis=1)
                # have all exog columns for single trial, need endog
                end = tr[params['endflag']][0, 0] + endadd
                for j, neur in enumerate(tr[neurfield][:, 0]):
                    bn = bins_tr(neur, beg, end, binsize, column=True)
                    if np.all(bn == 0):
                        bn[:, :] = np.nan
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
                nt = endog[:, i]
                mask = np.logical_not(np.isnan(nt))
                neurdic['endog'] = nt[mask].reshape((-1, 1))
                exo = exog[mask, :]
                neurdic['exog'] = np.concatenate((np.ones((exo.shape[0], 1)), 
                                                  exo), axis=1)
                glmdat.append(neurdic)
    return glmdat, glmlabels

def extract_ar(trl, params):
    q_ar = params['q_ar']
    neurf = params['neurf']
    binsize = params['binsize']
    endt = params['endflag']
    beg = params['begtime']
    end = params['endtime'] + trl[endt][0,0]
    for i, neur in enumerate(trl[neurf][:, 0]):
        spkbins = bins_tr(neur, beg - q_ar*binsize, end, binsize, column=True)
        for j in xrange(q_ar):
            lag = j + 1
            label = 'neuron {}, lag {}'.format(i, lag)
            neurtr = spkbins[q_ar - lag:-lag, :]
            if j == 0:
                allbins = neurtr
                all_labels = [label]
            else:
                allbins = np.concatenate((allbins, neurtr), axis=1)
                all_labels.append(label)
    return allbins, all_labels

def extract_nf(trl, params, lastcodebuff=400.):
    tnumf = params['tnumfield']
    neurf = params['neurf']
    tau1 = params['tau1']
    novfunc = params['novfunc']
    binsize = params['binsize']
    beg = params['begtime']
    endadd = params['endtime']
    end = trl[params['endflag']][0,0] + endadd
    mask = np.arange(beg, end, binsize).reshape((-1, 1))
    nbins = np.ceil((end - beg) / binsize)
    if trl[tnumf] > 6:
        lvf = params['lviewf']
        lv = trl[lvf][0, 0]
        rvf = params['rviewf']
        rv = trl[rvf][0, 0]
        ltf = params['ltimef']
        lt = trl[ltf][0, 0]
        rtf = params['rtimef']
        rt = trl[rtf][0, 0]
        lnov = novfunc(lv)
        lnov_r = np.repeat(lnov, nbins, axis=0)*(mask > lt + tau1)

        rnov = novfunc(rv)
        rnov_r = np.repeat(rnov, nbins, axis=0)*(mask > rt + tau1)
        novpars = np.concatenate((lnov_r, rnov_r, np.zeros(lnov_r.shape)), 
                                 axis=1)
    else:
        centif_on = params['centimgf_on']
        centif_off = params['centimgf_off']
        tcodef = params['tcodef']
        tcodes = trl[tcodef]
        centcodes = get_cent_codes(tcodes)
        centimgs_on = trl[centif_on] + tau1
        centimgs_off = trl[centif_off] + tau1
        if centimgs_on.shape[0] > centimgs_off.shape[0]:
            lastcode = centimgs_on[-1, 0] + lastcodebuff
            centimgs_off = np.concatenate((centimgs_off, 
                                           [[lastcode]]), axis=0)
        for i, img_on in enumerate(centimgs_on):
            img_off = centimgs_off[i]
            centcode = centcodes[i]
            cn = novfunc(get_code_views(centcode))
            if i == 0:
                novpars = np.zeros((nbins, cn.shape[1]))
            fmask = ((img_on <= mask)*(mask <= img_off))[:, 0]
            novpars[fmask, :] = cn
        novpars = np.concatenate((np.zeros(novpars.shape), 
                                  np.zeros(novpars.shape),
                                  novpars), axis=1)
    if novpars.shape[0] > 3:
        labels = ['left nov', 'left int', 'left fam', 'right nov', 'right int', 
                  'right fam', 'cent nov', 'cent int', 'cent fam']
    else:
        labels = ['left novelty', 'right novelty', 'cent novelty']
    return novpars, labels

def fit_glm(glmdic, labels, link='log', 
            family=sm.families.Poisson(sm.families.links.log)):
    endog = glmdic['endog']
    exog = glmdic['exog']
    gpoiss = sm.GLM(endog, exog, family=family)
    gp_results = gpoiss.fit()
    gp_results.model.xnames = labels
    return gp_results
                
