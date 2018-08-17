
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from utility import *

def trl_glm_vars(tr, winsize, winstep):
    # check trial type
    # generate matrix of trial variables
    # split dim trials into multiple trials
    # get neural dat

def organize_data_glm(data, drunfield='datanum', neurfield='spike_times',
                      tnumfield='trialnum'):
    druns = get_data_run_nums(data, drunfield)
    for k, dr in enumerate(druns):
        drdata = data[data[drunfield] == dr]
        for i, tr in enumerate(drdata):
            ns, ls, typ, ni = trl_glm_vars(tr, winsize, winstep)
            if k == 0 and i == 0:
                all_ns = ns
                all_ni = ni
                all_ls = ls
                all_typ = typ
            else:
                all_ns = np.concatenate((all_ns, ns), axis=0)
                all_ni = np.concatenate((all_ni, ni), axis=0)
                all_ls = np.concatenate((all_ls, ls), axis=0)
                all_typ = np.concatenate((all_typ, typ), axis=0)
    return all_ns, all_ls


def organize_dim_data(dimdat, drunfield='datanum', neurfield='spike_times',
                      usefuncs=None, params=None, tnumfield='trialnum', 
                      novfunc=views_to_member, tau1=50., bigwindows=False):
    """ 
    dimdat - data to use for glm
    usefuncs - list of functions that take a single trial and return one or 
      more columns in the exogenous matrix for that trial 
    """
    if usefuncs is None:
        usefuncs = []
    if params is None:
        params = {'begtime':200., 'endtime':0., 'binsize':50., 'q_ar':20,
                  'neurf':neurfield, 'winsize':50., 'endflag':'reward_time',
                  'lviewf':'leftviews', 'rviewf':'rightviews', 
                  'ltimef':'left_img_on', 'rtimef':'right_img_on', 
                  'centimgf_on':'centimgon', 'centimgf_off':'centimgoff',
                  'eyepos':'eyepos', 'tau1':tau1, 'tcodef':'code_numbers',
                  'tnumfield':'trial_type', 'novfunc':novfunc, 
                  'bigwind_absend':1000}
    druns = get_data_run_nums(dimdat, drunfield)
    glmdat = []
    glmlabels = ['const']
    beg = params['begtime']
    endadd = params['endtime']
    binsize = params['binsize']
    nruns = len(druns)
    neurnum = 0
    for k, dr in enumerate(druns):
        print 'organizing run {} of {}'.format(k+1, nruns)
        drdat = dimdat[dimdat[drunfield] == dr]
        # prep indep vars
        if len(drdat[0][neurfield]) > 0:
            ntrs = drdat.shape[0]
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
                for j, neur in enumerate(tr[neurfield][0, :]):
                    bn = bins_tr(neur, beg, end, binsize, column=True)
                    if np.all(bn == 0):
                        bn[:, :] = np.nan
                    if j == 0:
                        neurblock = bn
                    else:
                        neurblock = np.concatenate((neurblock, bn), axis=1)
                if bigwindows:
                    windiff = (params['bigwind_absend'] - beg) / binsize
                    neurblock = np.reshape(neurblock[:windiff, :], 
                                           (windiff, -1, 1))
                    allcols = np.reshape(allcols[:windiff, :], 
                                         (windiff, -1, 1))
                if i == 0:
                    endog = neurblock
                    exog = allcols
                else:
                    # if bigwindowing, want to keep trials separate and same
                    # length
                    if bigwindows:
                        endog = np.concatenate((endog, neurblock), 
                                               axis=2)
                        exog = np.concatenate((exog, allcols), 
                                              axis=2)
                    else:
                        endog = np.concatenate((endog, neurblock), axis=0)
                        exog = np.concatenate((exog, allcols), axis=0)
            for i in xrange(tr[neurfield].shape[1]):
                neurdic = {}
                if bigwindows:
                    nt = endog[:, i, :]
                    mask = np.logical_not(np.isnan(nt))
                    print mask.shape
                    neurdic['endog'] = nt[mask].reshape((windiff, 1, -1))
                    exo = exog[:, :, mask.sum(0) >= windiff]
                    print 'endo', neurdic['endog'].shape
                    print 'exo', exo.shape
                    exo = np.concatenate((np.ones((exo.shape[0], 1, exo.shape[2])),
                                          exo), axis=1)
                    print exo.shape
                    neurdic['exog'] = exo
                    
                else:
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

def extract_nf_vplt(trl, params, nbins, mask, tau1):
    lvf = params['lviewf']
    lv = trl[lvf][0, 0]
    rvf = params['rviewf']
    rv = trl[rvf][0, 0]
    ltf = params['ltimef']
    lt = trl[ltf][0, 0]
    rtf = params['rtimef']
    rt = trl[rtf][0, 0]
    novfunc = params['novfunc']
    lnov = novfunc(lv)
    lnov_r = np.repeat(lnov, nbins, axis=0)*(mask > lt + tau1)
        
    rnov = novfunc(rv)
    rnov_r = np.repeat(rnov, nbins, axis=0)*(mask > rt + tau1)
    return np.concatenate((lnov_r, rnov_r, np.zeros(lnov_r.shape)), 
                          axis=1)

def extract_dim_stim(trl, params, nbins, mask, tau1):
    centif_on = params['centimgf_on']
    centif_off = params['centimgf_off']
    tcodef = params['tcodef']
    novfunc = params['novfunc']
    tcodes = trl[tcodef]
    centcodes = get_cent_codes(tcodes)
    centimgs_on = trl[centif_on] + tau1
    centimgs_off = trl[centif_off] + tau1
    

def extract_nf_dimming(trl, params, nbins, mask, tau1, lastcodebuff=400.):
    centif_on = params['centimgf_on']
    centif_off = params['centimgf_off']
    tcodef = params['tcodef']
    novfunc = params['novfunc']
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
    return np.concatenate((np.zeros(novpars.shape), 
                           np.zeros(novpars.shape),
                           novpars), axis=1)

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
        print 'vplt'
        novpars = extract_nf_vplt(trl, params, nbins, mask, tau1)
        print novpars
    else:
        print 'dim'
        novpars = extract_nf_dimming(trl, params, nbins, mask, tau1, 
                                     lastcodebuff)
        print novpars
    if novpars.shape[0] > 3:
        labels = ['left nov', 'left int', 'left fam', 'right nov', 'right int', 
                  'right fam', 'cent nov', 'cent int', 'cent fam']
    else:
        labels = ['left novelty', 'right novelty', 'cent novelty']
    labels = ['visual stim'] + labels
    visstim = np.sum(novpars, axis=1)
    visstim = np.reshape(visstim > 0, (-1, 1))
    novpars = np.concatenate((visstim, novpars), axis=1)
    return novpars, labels

def fit_glm(glmdic, labels, link='log', 
            family=sm.families.Poisson(sm.families.links.log)):
    endog = glmdic['endog']
    exog = glmdic['exog']
    gpoiss = sm.GLM(endog, exog, family=family)
    gp_results = gpoiss.fit()
    gp_results.model.xnames = labels
    return gp_results
                
def fit_windglm(glmdic, labels, link='log', bigwin=5, step=2.,
                family=sm.families.Poisson(sm.families.links.log)):
    # bigwin is bigwin*orig step in actual ms
    endog = glmdic['endog']
    exog = glmdic['exog']
    numts = endog.shape[0]
    glms = []
    nsteps = (numts - bigwin)/step + bigwin
    coeffs = np.zeros((nsteps, exog.shape[1]))
    for i, s in enumerate(np.arange(0, numts - np.ceil(bigwin/step), step)):
        sendog = endog[s:s+bigwin, : :]
        longendog = np.concatenate([x for x in sendog], axis=1)
        sexog = exog[s:s+bigwin, :, :]
        longexog = np.concatenate([x for x in sexog], axis=1)
        sg = sm.GLM(longendog.T, longexog.T, family=family)
        try:
            sg_r = sg.fit()
        except Exception:
            x = np.ones(coeffs.shape[1])
            x[:] = np.nan
            coeffs[i, :] = x
            print 'glm failed on {}, {}:{}'.format(i, s, s+bigwin)
        finally:
            coeffs[i, :] = sg_r.params
        glms.append(sg_r)
    return glms, coeffs
    
def plot_windglm(coeffs, labels, plot_inds=[0, 21, 28, 29, 30]):
    if len(plot_inds) == 0:
        plot_inds = np.arange(len(labels))
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    for i in plot_inds:
        l = labels[i]
        coe = coeffs[:, i]
        if i > 1: 
            coe = coe + coeffs[:, 0] + coeffs[:, 1]
        ax.plot(coe, label=l)
    ax.legend()
    plt.show(block=False)
