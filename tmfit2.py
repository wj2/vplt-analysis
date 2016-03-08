
import pymc
from tm import SalSacc, proc_many_outs
import eyes as es
import utility as u
import os 
import numpy as np
import cPickle as cp
import scipy.stats as sts
import multiprocessing as mp
import random

info_name_temp = 'proc_guides_{}ms.npy'

tau_h = pymc.TruncatedNormal('tau_h', 30., 1/(10.**2), 5., 10000)
eff = pymc.TruncatedNormal('eff', .122, 1/(2.**2), .001, 1.) # big U
tau_f = pymc.TruncatedNormal('tau_f', 20., 1/(10.**2), 1., 10000)
tau_d = pymc.TruncatedNormal('tau_d', 100., 1/(20.**2), 1., 10000)
a = pymc.TruncatedNormal('a', 3.96, 1/(1.**2), 0.01, 10.) # big J
tau_x = pymc.TruncatedNormal('tau_x', 200., 1/(50.**2), 1., 10000)
tau_u = pymc.TruncatedNormal('tau_u', 200., 1/(50.**2), 1., 10000)

prob_tc = pymc.DiscreteUniform('prob_tc', lower=50., upper=100000.)
prob_gc = pymc.Uniform('prob_gc', lower=0., upper=10.)
prob_dp = pymc.Uniform('prob_dp', lower=0., upper=10.)
prob_sb = pymc.Uniform('prob_sb', lower=0., upper=.5)

look_mod = pymc.Uniform('look_mod', lower=0., upper=5.)
off_img = pymc.Uniform('off_img', lower=0., upper=1.5)
nov_img = pymc.Uniform('nov_img', lower=0., upper=1.5)
fam_img = pymc.Uniform('fam_img', lower=0., upper=1.5)

prestime = 4000
tstep = 10.
observations = np.load(info_name_temp.format(int(tstep)))
guide_buff = pymc.TruncatedNormal('guide_buff', 600/tstep, 1/((100./tstep)**2),
                                  500/tstep, 1000/tstep)

eps = np.log(10.**(-200))
plot_preses = False
samp_pres = 80

par = False
if par:
    send_pool = mp.Pool(processes=mp.cpu_count())

@pymc.stochastic(observed=True, dtype=object)
def eyetrace(prob_tc=prob_tc, prob_gc=prob_gc, prob_dp=prob_dp, 
             samebias=prob_sb, tau_h=tau_h, eff=eff, tau_f=tau_f, 
             tau_d=tau_d, a=a, tau_x=tau_x, tau_u=tau_u, 
             nov_img=nov_img, fam_img=fam_img, off_img=off_img,
             guide_buff=guide_buff, look_mod=look_mod, value=observations):
    
    def logp(prob_tc=prob_tc, prob_gc=prob_gc, prob_dp=prob_dp, 
             samebias=prob_sb, tau_h=tau_h, eff=eff, tau_f=tau_f, 
             tau_d=tau_d, a=a, tau_x=tau_x, tau_u=tau_u, 
             nov_img=nov_img, fam_img=fam_img, off_img=off_img,
             guide_buff=guide_buff, look_mod=look_mod, value=observations):
        rtf_func = lambda x: x
        sac = SalSacc(prob_tc, prob_gc, prob_dp, samebias, tau_h, rtf_func, eff, 
                      tau_f, tau_d, a, tau_x=tau_x, tau_u=tau_u)
        init_h = [0., 0., 0.]
        rs = [off_img, nov_img, fam_img]
        pardict = {'tend':prestime, 'tstep':tstep, 'gbuff':guide_buff}
        useguides = random.sample(observations, samp_pres)
        if par:
            outs = sac.simulate_many_par(samp_pres, rs, look_mod, init_h, 
                                         pardict, pool=send_pool, 
                                         guides=useguides)
        else:
            outs = sac.simulate_many(samp_pres, rs, look_mod, init_h, 
                                     pardict, guides=useguides)
        ts, hs_3d, lps_2d, looks_2d, saccts, fixes, ps = proc_many_outs(outs)
        logps = np.log(np.mean(ps))
        if np.isnan(logps):
            print 'nantrip'
            logps = eps
        print logps
        return logps
