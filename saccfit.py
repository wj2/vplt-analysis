
import pymc
from saccader import Saccader, plot_presentation
import eyes as es
import utility as u
import os 
import numpy as np
import cPickle as cp
import scipy.stats as sts

ld_name = 'lendist.pkl'
save = False

def save_distribs(lendistribname=ld_name):
    errorfield = 'TrialError'
    tnumfield = 'trial_type'
    eyefield = 'eyepos'
    os.chdir('data')
    d = u.load_separate(['.'], 'bootsy-bhvit-run[0-9]*.mat')
    ed = d[d[errorfield] == 0]
    os.chdir('..')
    fn, ff, nn, nf = 7, 8, 9, 10
    lens, looks, s_bs, s_es = es.get_fixtimes(ed, [fn, ff, nn, nf], 
                                              postthr='fixation_off')
    observed_lendistrib = np.concatenate(lens[nf] + lens[fn], axis=0)
    cp.dump(observed_lendistrib, open(lendistribname, 'wb'))
    return observed_lendistrib

if save:
    observed_lendistrib = save_distribs()
else:
    observed_lendistrib = cp.load(open(ld_name, 'rb'))

use_informative_priors = False
# priors and parameters

if use_informative_priors:
    pass
    # long-term novel function and params
    
    # short-term novel function and params
    
    # salience time constant

    # saccade probability time constant
    
    # off target novelty
    
    # saccade probability growth constant

    # bias toward same-image saccade
else:
    # long-term novel function and params
    lt_novfunc = lambda x, a, b: a*np.exp(-x) + b
    lt_novpars = (pymc.Uniform('lt_novpar1', lower=0., upper=5,
                               doc='lt nov multiplier'),
                  pymc.Uniform('lt_novpar2', lower=0., upper=2.,
                               doc='lt nov add'))

    # short-term novel function and params
    st_novfunc = lambda x, a: -a*x
    st_novpars = (pymc.Uniform('st_novpar1', lower=0., upper=5.,
                               doc='st nov multiplier'),)

    # salience time constant
    sal_tc = pymc.DiscreteUniform('sal_tc', lower=100, upper=1000, 
                                  doc='salience time constant')

    # saccade probability time constant
    prob_tc = pymc.DiscreteUniform('prob_tc', lower=100, upper=1000,
                                   doc='salience time constant')

    # off target novelty
    off_tnov = pymc.Uniform('off_tnov', lower=0, upper=1.,
                                    doc='off target novelty')

    # saccade probability growth constant
    sacc_grow = pymc.Uniform('sacc_grow', lower=0, upper=1.,
                             doc='saccade probability grow constant')

    # bias toward same-image saccade
    samebias = pymc.Uniform('samebias', lower=0, upper=1.,
                            doc='same image saccade bias')

leftviews = 0.
rightviews = 10000.
prestime = 5000
tstep = 1.

use_const_samp_logp = True
plot_preses = False
samp_pres = 5

sacclen = prestime*samp_pres
saccval = np.zeros(sacclen)
saccval[:] = np.nan

def compare_fixlens(samp_fixlen, fixlendist, eps=.000000001):
    nonan_samp_fixlen = samp_fixlen[np.logical_not(np.isnan(samp_fixlen))]
    nonan_fixlendist = fixlendist[np.logical_not(np.isnan(fixlendist))]
    print nonan_samp_fixlen, nonan_fixlendist
    ks, p = sts.ks_2samp(nonan_samp_fixlen, nonan_fixlendist)
    print ks, p
    return np.log(p + eps)

@pymc.deterministic
def sample_eyetrace(lt_novpars=lt_novpars, st_novpars=st_novpars, 
                    sal_tc=sal_tc, prob_tc=prob_tc, off_tnov=off_tnov, 
                    sacc_grow=sacc_grow, samebias=samebias):
    print 'getting samp'
    sac = Saccader(lt_novfunc, lt_novpars, st_novfunc, st_novpars, sal_tc, 
                   off_tnov, prob_tc, sacc_grow, samebias)
    out = sac.present_many([leftviews, rightviews], samp_pres, 
                           prestime=prestime, tstep=tstep)
    ts, sals, looks, saccs, lookprobs = out
    if plot_preses:
        plot_presentation(ts, sals[0], looks[0], saccs[0], lookprobs[0])
    fixes = np.concatenate(map(np.diff, saccs), axis=0)
    fixarr = np.zeros(sacclen)
    fixarr[:] = np.nan
    fixarr[:len(fixes)] = fixes
    return fixarr

# @pymc.stochastic
# def sample_eyetrace(lt_novpars=lt_novpars, st_novpars=st_novpars, 
#                     sal_tc=sal_tc, prob_tc=prob_tc, off_tnov=off_tnov, 
#                     sacc_grow=sacc_grow, samebias=samebias):
    
#     def logp(value, lt_novpars=lt_novpars, st_novpars=st_novpars, 
#              sal_tc=sal_tc, prob_tc=prob_tc, off_tnov=off_tnov, 
#              sacc_grow=sacc_grow, samebias=samebias):
#         if not use_const_samp_logp:
#             sac = Saccader(lt_novfunc, lt_novpars, st_novfunc, st_novpars, sal_tc, 
#                            off_tnov, prob_tc, sacc_grow, samebias)

#             ts, sals, looks, saccs, lookprobs = sac.present_many([leftviews, 
#                                                                   rightviews],
#                                                                  samp_pres, 
#                                                                  prestime)
#             samp_fixlens = np.concatenate(map(np.diff, saccs), axis=0)
#             logp = compare_fixlens(samp_fixlens, value)
#         else:
#             logp = 0.0
#         print 'logp', value, lt_novpars, st_novpars, sal_tc
#         return logp
        
#     def random(lt_novpars=lt_novpars, st_novpars=st_novpars, 
#                sal_tc=sal_tc, prob_tc=prob_tc, off_tnov=off_tnov, 
#                sacc_grow=sacc_grow, samebias=samebias):
#         print 'getting samp'
#         sac = Saccader(lt_novfunc, lt_novpars, st_novfunc, st_novpars, sal_tc, 
#                        off_tnov, prob_tc, sacc_grow, samebias)
#         ts, sals, looks, saccs, lookprobs = sac.present([leftviews, rightviews],
#                                                         prestime)
#         fixes = np.diff(saccs)
#         fixarr = np.zeros(sacclen)
#         fixarr[:] = np.nan
#         fixarr[:len(fixes)] = fixes
#         return fixarr

@pymc.stochastic(observed=True)
def model_fixlen_to_actual(model_distrib=sample_eyetrace, 
                           value=observed_lendistrib):

    def logp(model_distrib=sample_eyetrace, value=observed_lendistrib):
        logp = compare_fixlens(model_distrib, observed_lendistrib)
        print 'compfixlen', logp, model_distrib
        return logp
