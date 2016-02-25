
import pymc
from tm import SalSacc, proc_many_outs
import eyes as es
import utility as u
import os 
import numpy as np
import cPickle as cp
import scipy.stats as sts
import multiprocessing as mp

info_name = 'rs_lc_ft_info.npz'

obs = np.load(info_name)
observations = (obs['look_course'].mean(0), obs['fix_lens'])

tau_h = pymc.TruncatedNormal('tau_h', 30., 10., 0.01, 10000)
eff = pymc.Uniform('eff', lower=.001, upper=1.)
tau_f = pymc.TruncatedNormal('tau_f', 20., 10., 0.01, 10000)
tau_d = pymc.TruncatedNormal('tau_d', 100., 20., 0.01, 10000)
a = pymc.TruncatedNormal('a', 5., 1., 0.01, 10000)
tau_x = pymc.TruncatedNormal('tau_x', 200., 50., 0.01, 10000)
tau_u = pymc.TruncatedNormal('tau_u', 200., 50., 0.01, 10000)

prob_tc = pymc.Uniform('prob_tc', lower=10., upper=100000.)
prob_gc = pymc.Uniform('prob_gc', lower=0., upper=10.)
prob_dp = pymc.Uniform('prob_dp', lower=0., upper=10.)
prob_sb = pymc.Uniform('prob_sb', lower=0., upper=.5)

look_mod = pymc.Uniform('look_mod', lower=0., upper=5.)
off_img = pymc.Uniform('off_img', lower=0., upper=1.5)
nov_img = pymc.Uniform('nov_img', lower=0., upper=1.5)
fam_img = pymc.Uniform('fam_img', lower=0., upper=1.5)

prestime = 4500
tstep = 1.

use_const_samp_logp = True
plot_preses = False
samp_pres = 5
test_interval = 50

sacclen = prestime*samp_pres
saccval = np.zeros(sacclen)
saccval[:] = np.nan

send_pool = mp.Pool(processes=mp.cpu_count())

def compare_fixlens(samp_fixlen, fixlendist, eps=.000000001):
    nonan_samp_fixlen = samp_fixlen[np.logical_not(np.isnan(samp_fixlen))]
    nonan_fixlendist = fixlendist[np.logical_not(np.isnan(fixlendist))]
    print nonan_samp_fixlen, nonan_fixlendist
    ks, p = sts.ks_2samp(nonan_samp_fixlen, nonan_fixlendist)
    print ks, p
    return np.log(p + eps)

@pymc.deterministic
def sample_eyetrace(prob_tc=prob_tc, prob_gc=prob_gc, prob_dp=prob_dp, 
                    samebias=prob_sb, tau_h=tau_h, eff=eff, tau_f=tau_f, 
                    tau_d=tau_d, a=a, tau_x=tau_x, tau_u=tau_u, 
                    nov_img=nov_img, fam_img=fam_img, off_img=off_img):
    rtf_func = lambda x: x
    sac = SalSacc(prob_tc, prob_gc, prob_dp, samebias, tau_h, rtf_func, eff, 
                  tau_f, tau_d, a, tau_x=tau_x, tau_u=tau_u)
    init_h = [0., 0., 0.]
    rs = [off_img, nov_img, fam_img]
    pardict = {'tend':prestime, 'tstep':tstep}
    print 'beg sim'
    outs = sac.simulate_many_par(samp_pres, rs, look_mod, init_h, pardict,
                                 pool=send_pool)
    ts, hs_3d, lps_2d, looks_2d, saccts, fixes = proc_many_outs(outs)
    print 'end sim'
    hs = hs_3d.mean(2)
    plooks = np.zeros(hs.shape)
    for i, row in enumerate(plooks):
        row[:] = np.mean(looks_2d == i, axis=0)
    fixarr = np.zeros(sacclen)
    fixarr[:] = np.nan
    fixarr[:len(fixes)] = fixes
    return plooks, fixarr

@pymc.stochastic(observed=True, dtype=object)
def model_fixlen_to_actual(model_distrib=sample_eyetrace, 
                           value=observations):

    def logp(model_distrib=sample_eyetrace, value=observations):
        fix_logp = compare_fixlens(model_distrib[1], value[1])
        tr = model_distrib[0]*samp_pres
        tr_true = value[0]
        zt_0 = [sts.binom_test(x, samp_pres, tr_true[i*test_interval, 0]) 
                for i, x in enumerate(tr[::test_interval, 0])]
        zt_1 = [sts.binom_test(x, samp_pres, tr_true[i*test_interval, 1]) 
                for i, x in enumerate(tr[::test_interval, 1])]
        traj_p = np.product(zt_0)*np.product(zt_1)
        print 'traj_p', traj_p
        full_logp = np.log(traj_p) + fix_logp
        print 'compfixlen', full_logp, model_distrib
        return full_logp
