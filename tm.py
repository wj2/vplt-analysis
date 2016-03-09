
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools as ft
import scipy.stats as sts
import random
import sys

def euler_integrate(f, init, inputs, tbeg=0, tend=4000, tstep=.1):
    ts = np.arange(tbeg, tend, tstep)
    out = np.zeros((len(init), len(ts)))
    out[:, 0] = init
    for i in xrange(1, len(ts)):
        out[:, i] = out[:, i-1] + tstep*f(ts[i], out[:, i-1], *inputs)
    return ts, out

def dtmdt(t, x, v, u, a, tau_d, tau_f, tau_s):
    out = np.zeros(len(x))
    out[0] = ((1 - x[0])/tau_d) - x[1]*x[0]*v(t)
    out[1] = -x[1]/tau_f + u*(1 - x[1])*v(t)
    out[2] = - x[2]/tau_s + a*x[0]*x[1]*v(t)
    return out
    
def plot_dtmdt():
    v_rate = .05
    v_cut = 500.
    v = lambda t: v_rate*(t < v_cut)
    u = .5
    a = 1.
    tau_d = 1000.
    tau_f = 100.
    tau_s = 50.
    ts, outc = euler_integrate(dtmdt, [1., 0., 0.], 
                               [v, u, a, tau_d, tau_f, tau_s])
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(ts, outc[0, :], label='x')
    ax.plot(ts, outc[1, :], label='u')
    ax.plot(ts, outc[2, :], label='I')
    ax.legend(frameon=False)
    plt.show(block=False)

def mode_filter(tc, winsize):
    filted = np.zeros(tc.shape)
    halfw_fore = np.ceil(winsize / 2.)
    halfw_post = np.floor(winsize / 2.)
    for i, x in enumerate(tc):
        if i < halfw_post:
            coll = tc[:i+halfw_fore]
        elif i > tc.shape[0] - halfw_fore:
            coll = tc[i-halfw_post:]
        else:
            coll = tc[i-halfw_post:i+halfw_fore]
        m = sts.mode(coll)[0]
        filted[i] = m
    return filted

def process_guides(guides, tstep, currstep=1, filt=True, window=50.):
    guide_list = []
    for tc in guides:
        lc = np.where(tc)[1]
        slc = lc[::tstep/currstep]
        mslc = mode_filter(slc, window/(tstep/currstep))
        guide_list.append(mslc)
    return guide_list

def plot_estrunvar(range_, allstds, allmeans):
    f = plt.figure()
    ax1 = f.add_subplot(2, 1, 1)
    ax1.plot(range_, allstds)
    ax1.set_ylabel('std log p')
    ax2 = f.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(range_, allmeans)
    ax2.set_xlabel('samples')
    ax2.set_ylabel('mean log p')
    f.tight_layout()
    plt.show(block=False)

def get_range_estrunvar(range_, rns, par=False, tstep=1., guides=None, proc=True,
                        show=True):
    allps = np.zeros((len(range_), rns))
    allstds = np.zeros(len(range_))
    allmeans = np.zeros(len(range_))
    for i, x in enumerate(range_):
        ps, pstd = estimate_runvar(rns, x, par=par, tstep=tstep, guides=guides,
                                   proc=proc, show=False)
        allps[i, :] = ps
        allstds[i] = pstd
        allmeans[i] = np.mean(ps)
    if show:
        plot_estrunvar(range_, allstds, allmeans)
    return allps, allmeans, allstds

def estimate_runvar(est_n, n, par=False, tstep=1., guides=None, proc=True,
                    show=False):
    ps = np.zeros(est_n)
    for i in xrange(est_n):
        outs = do_and_plot(n, par=par, tstep=tstep, guides=guides, proc=proc, 
                           show=show)
        ps[i] = np.mean(np.log(outs[-1]))
    return ps, np.std(ps)

def do_and_plot(n=1, par=True, tstep=1., guides=None, proc=False, show=True,
                params=None):
    rtf_func = identity_func
    tend = 4000.
    if params is None:
        tau_h = 30.
        eff = .5
        tau_f = 10.
        tau_d = 100.
        a = 2.
        tau_x = 200.
        tau_u = 200.
        
        prob_tc = 10000.
        prob_growconst = .05
        prob_diffpar = 1.
        samebias = 0.
        gbuff = 600./tstep
        
        look_mod = .2
        off_img = .02
        nov_img = .8
        fam_img = .1
    else:
        tau_h = params['tau_h']
        eff = params['eff']
        tau_f = params['tau_f']
        tau_d = params['tau_d']
        a = params['a']
        tau_x = params['tau_x']
        tau_u = params['tau_u']
        prob_tc = params['prob_tc']
        prob_growconst = params['prob_gc']
        prob_diffpar = params['prob_dp']
        samebias = params['prob_sb']
        gbuff = params['guide_buff']
        look_mod = params['look_mod']
        off_img = params['off_img']
        nov_img = params['nov_img']
        fam_img = params['fam_img']    
    rs = np.array([off_img, nov_img, fam_img])
    ss = SalSacc(prob_tc, prob_growconst, prob_diffpar, samebias,
                 tau_h, rtf_func, eff, tau_f, tau_d, a, tau_x=tau_x, 
                 tau_u=tau_u, tf_func=thresh_linear)
    init_h = [0., 0., 0.]
    manysimdic = {'tstep':tstep, 'tend':tend, 'gbuff':gbuff}
    if guides is not None and not proc:
        proc_guides = process_guides(guides, tstep)
        select_guides = random.sample(proc_guides, n)
    elif proc:
        select_guides = random.sample(guides, n)
    else:
        select_guides = None
    if n == 1:
        ts, hs, saccts, lp, looks = ss.simulate(rs, look_mod, init_h, 
                                                tstep=tstep, 
                                                guide=select_guides)
    else:
        if par:
            # potentially using same seed for each proc, investigate
            outs = ss.simulate_many_par(n, rs, look_mod, init_h, manysimdic,
                                        guides=select_guides)
        else:
            outs = ss.simulate_many(n, rs, look_mod, init_h, manysimdic,
                                    guides=select_guides)
        ts, hs_3d, lps_2d, looks_2d, saccts, fix_lens, ps = proc_many_outs(outs)
        hs = hs_3d.mean(2)
        lp = lps_2d.mean(0)
        looks = looks_2d.mean(0)
        plooks = np.zeros(hs.shape)
        for i, row in enumerate(plooks):
            row[:] = np.mean(looks_2d == i, axis=0)
    if show:
        plot_ss(ts, hs, plooks, saccts, lp)
    return ts, hs, looks, saccts, lp, ps

def proc_many_outs(outs):
    xs = outs[0][0]
    hs_3d = np.dstack([x[1] for x in outs])
    saccts = np.concatenate([x[2] for x in outs])
    fix_lens = np.concatenate([np.diff(x[2]) for x in outs])
    lps_2d = np.vstack([x[3] for x in outs])
    looks_2d = np.vstack([x[4] for x in outs])
    ps = [x[5] for x in outs]
    return xs, hs_3d, lps_2d, looks_2d, saccts, fix_lens, ps

def plot_ss(ts, sals, looks, saccs, lookprob, axbuff=.5,
            sallabs=('off img', 'left', 'right'), sal_ax=None,
            sal_ts=None, xtickstep=500, show=True):
    f = plt.figure(figsize=(8, 10))
    salax = f.add_subplot(3, 1, 1)
    probax = f.add_subplot(3, 1, 2, sharex=salax)
    lookax = f.add_subplot(3, 1, 3, sharex=salax)
    xticks = np.arange(ts[0], ts[-1], xtickstep)
    sum_sals = sals.sum(0)
    sum_sals = sum_sals
    for i, row in enumerate(sals):
        if sal_ax is not None:
            print 'hello'
            sal_ax.plot(ts, row / sum_sals, label=sallabs[i])
        salax.plot(ts, row / sum_sals, label=sallabs[i])
    if sal_ax is not None:
        print 'hey'
        sal_ax.set_ylabel('salience')
        sal_ax.legend(frameon=False, loc=2)
        sal_ax.set_xticks(sal_ts)
    salax.set_ylabel('target salience')
    salax.legend()
    salax.set_xticklabels(xticks, visible=False)

    probax.plot(ts, lookprob)
    probax.set_xticklabels(xticks, visible=False)
    probax.set_ylabel('saccade prob')

    for i, row in enumerate(looks):
        lookax.plot(ts, row, label=sallabs[i])
    lookax.set_ylabel('fixation probability')
    lookax.set_xticks(xticks)
    if show:
        plt.show(block=False)    

def plot_imgsal(ts, hs, xs, us, const_div=.05):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    h_labs = ['h{}'.format(i) for i in xrange(hs.shape[0])]
    x_labs = ['x{}'.format(i) for i in xrange(xs.shape[0])]
    u_labs = ['u{}'.format(i) for i in xrange(xs.shape[0])]
    ax.plot(ts, hs.T, label=h_labs)
    # ax.plot(ts, xs.T, '--', label=x_labs)
    # ax.plot(ts, us.T, '--', label=u_labs)
    const = np.ones((1, hs.shape[1]))*const_div
    hs_cat = np.concatenate((hs, const), axis=0)
    div_hs = hs_cat / hs_cat.sum(0)
    ax.plot(ts, div_hs.T, label='dived')
    ax.plot(ts, hs[0,:] - hs[1, :], label='subbed')
    ax.set_xlabel('time (ms)')
    ax.legend()
    plt.show(block=False)

class ImgSalience(object):
    
    def __init__(self, tau_h, rtf_func, eff, tau_f, tau_d, a, 
                 tf_func=lambda x: np.max([x, 0]), tau_x=100., tau_u=100.):
        self.tau_h = tau_h
        self.rtf_func = rtf_func
        self.tf_func = tf_func
        self.tm_eff = eff
        self.tm_tau_f = tau_f
        self.tm_tau_d = tau_d
        self.tm_a = a
        self.tm_tau_x = tau_x
        self.tm_tau_u = tau_u

    def _dhdt(self, h, curr):
        return (-h + self.tf_func(curr))/self.tau_h

    def simulate(self, stims, init_h=0., init_x=1., tend=5000,
                 tstep=1, tbeg=0):
        ts = np.arange(tbeg, tend+tstep, tstep)
        hs = np.zeros((len(stims), ts.shape[0]))
        xs = np.zeros((len(stims), ts.shape[0]))
        us = np.zeros((len(stims), ts.shape[0]))
        hs[:, 0] = init_h
        xs[:, 0] = init_x
        us[:, 0] = self.tm_eff
        rs = self.rtf_func(np.array(stims))
        tm = TMStp(self.tm_eff, self.tm_tau_f, self.tm_tau_d, self.tm_a, 
                   self.tm_tau_x, self.tm_tau_u)
        for i in xrange(1, len(ts)):
            for j, r in enumerate(rs):
                h, x, u, _ = self.step(hs[j, i-1], xs[j, i-1], us[j, i-1], r,
                                       tstep, tm)
                hs[j, i] = h
                xs[j, i] = x
                us[j, i] = u
        return ts, hs, xs, us

    def step(self, h, x, u, r, delt, tm):
        nx, nu, r, curr = tm.step(x, u, r, delt)
        nh = h + delt*self._dhdt(h, curr)
        return nh, nx, nu, r

    def make_tmstp(self):
        tm = TMStp(self.tm_eff, self.tm_tau_f, self.tm_tau_d, self.tm_a, 
                   self.tm_tau_x, self.tm_tau_u)
        return tm

    def do_internal_step(self, i, hs, xs, us, rs, tstep, tm):
        for j, r in enumerate(rs):
            h, x, u, _ = self.step(hs[j, i-1], xs[j, i-1], us[j, i-1], r,
                                   tstep, tm)
            hs[j, i] = h
            xs[j, i] = x
            us[j, i] = u
        return hs, xs, us


def _simulate_top(simmer, stims, look_mod, init_h, params, guides, n):
    np.random.seed()
    if guides is not None:
        guide = guides[n]
    else:
        guide = None
    return simmer.simulate(stims, look_mod, init_h, guide_traj=guide,
                           **params)
    
def identity_func(x):
    return x

def thresh_linear(x):
    return np.max([x, 0])

class SalSacc(object):
    
    def __init__(self, prob_timeconst, prob_growconst, prob_diffpar, samebias, 
                 tau_h, rtf_func, eff, tau_f, tau_d, a, tf_func=thresh_linear, 
                 tau_x=100., tau_u=100.):
        self.is_m = ImgSalience(tau_h, rtf_func, eff, tau_f, tau_d, a,
                                tf_func, tau_x, tau_u)
        self.prob_tau = prob_timeconst
        self.probgrow = prob_growconst
        self.prob_diffpar = prob_diffpar
        self.samebias = samebias
        self._param_dict = {'prob_tc':prob_timeconst,
                            'prob_gc':prob_growconst,
                            'prob_dp':prob_diffpar, 'prob_sb':samebias,
                            'tau_h':tau_h, 'eff':eff, 'tau_f':tau_f,
                            'tau_d':tau_d, 'a':a, 'tau_x':tau_x,
                            'tau_u':tau_u}

    def _dpldt(self, lprob, lsal, nlsals):
        x = nlsals - lsal
        salgrow = self.prob_diffpar*np.max([np.max(x), 0])
        dpldt = (-(lprob - 1)*(self.probgrow + salgrow)) / self.prob_tau
        return dpldt

    def simulate_many(self, n, stims, look_mod, init_h, params, guides=None):
        outs = []
        for x in xrange(n):
            if guides is not None:
                guide = guides[x]
            else:
                guide = None
            out = self.simulate(stims, look_mod, init_h, guide_traj=guide, 
                                **params)
            outs.append(out)
        return outs

    def simulate_many_par(self, n, stims, look_mod, init_h, params, pool=None,
                          guides=None):
        simfunc = ft.partial(_simulate_top, self, stims, look_mod, init_h,
                             params, guides)
        if guides is not None:
            assert len(guides) == n
        if pool is None:
            pool = mp.Pool(processes=mp.cpu_count())
            origin_pool = True
        else:
            origin_pool = False
        outs = pool.map(simfunc, xrange(n))
        if origin_pool:
            pool.close()
            pool.join()
        return outs

    def simulate(self, stims, look_mod, init_h, init_x=1., init_pl=0., 
                 tend=5000, tstep=1, tbeg=0, look_def=0, gbuff=0,
                 start_eq=(True, False, False), guide_traj=None,
                 eps=.0000000001):
        ts = np.arange(tbeg, tend+tstep, tstep)
        hs = np.zeros((len(stims), ts.shape[0]))
        xs = np.zeros((len(stims), ts.shape[0]))
        us = np.zeros((len(stims), ts.shape[0]))
        pls = np.zeros(ts.shape)
        looks = np.zeros(ts.shape)
        sacc_ts = []
        hs[:, 0] = init_h
        xs[:, 0] = init_x
        us[:, 0] = self.is_m.tm_eff
        rs = self.is_m.rtf_func(np.array(stims))
        rs_mod = np.zeros(rs.shape)
        rs_mod[look_def] = look_mod
        tm = self.is_m.make_tmstp()
        p = np.ones(ts.shape)
        for i in xrange(1, len(ts)):
            hs, xs, us = self.is_m.do_internal_step(i, hs, xs, us, rs + rs_mod,
                                                    tstep, tm)
            l = looks[i]
            pls[i] = np.min([pls[i-1] + tstep*self._dpldt(pls[i-1], 
                                                          hs[looks[i], i], 
                                                          hs[:, i]),
                             1.])
            if guide_traj is None:
                if pls[i] > np.random.rand():
                    looks[i] = self._look_change(looks[i-1], 
                                                 hs[:, i])
                    pls[i] = eps
                    rs_mod[looks[i-1]] = 0
                    rs_mod[looks[i]] = look_mod
                    sacc_ts.append(ts[i])
                else:
                    looks[i] = looks[i-1]
                lps = self._get_look_pchanges(looks[i-1], hs[:, i])
                pchange_i = pls[i]*lps[looks[i]]
                pnochange_i = (1 - pls[i])
                if looks[i] != looks[i-1]:
                    p[i] = pchange_i
                else:
                    p[i] = pchange_i + pnochange_i
            else:
                looks[i] = guide_traj[gbuff+i]
                lps = self._get_look_pchanges(looks[i-1], hs[:, i])
                if np.any(lps > 1):
                    sys.stderr.write('\nerr hs, lps:'+str(hs[:, i])
                                     +' '+str(lps))
                    pd = self._get_param_dict(stims, look_mod, gbuff)
                    sys.stderr.write('\nparams: '+str(pd)+'\n')

                pchange_i = pls[i]*lps[looks[i]]
                if guide_traj[gbuff+i-1] != guide_traj[gbuff+i]:
                    p[i] = pchange_i
                    pls[i] = eps
                    rs_mod[looks[i-1]] = 0
                    rs_mod[looks[i]] = look_mod
                    sacc_ts.append(ts[i])
                else:
                    pnochange_i = (1 - pls[i])
                    p[i] = pchange_i + pnochange_i
        if guide_traj is not None:
            guide_p = np.prod(p)
        else:
            guide_p = None
        return ts, hs, sacc_ts, pls, looks, guide_p
            
    def _get_look_pchanges(self, lookind, sals):
        sump = np.sum(sals) + self.samebias
        lps = np.zeros(sals.size)
        for i, ls in enumerate(sals):
            if i == lookind:
                ls = ls + self.samebias
            lps[i] = ls/sump
        return lps
            
    def _look_change(self, lookind, sals):
        lps = self._get_look_pchanges(lookind, sals)
        return np.random.choice(np.arange(sals.size), p=lps)

    def _get_param_dict(self, stims, look_mod, gbuff):
        pd = self._param_dict
        pd['stims'] = stims
        pd['look_mod'] = look_mod
        pd['gbuff'] = gbuff
        return pd
        
class TMStp(object):
    
    def __init__(self, eff, tau_f, tau_d, a, tau_x, tau_u):
        self.eff = eff
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.a = a
        self.tau_x = tau_x
        self.tau_u = tau_u

    def _dxdt(self, x, u, r):
        return (((1 - x)/self.tau_d) - u*x*r)/self.tau_x

    def _dudt(self, x, u, r):
        return (((self.eff - u)/self.tau_f) + self.eff*(1 - u)*r)/self.tau_u

    def _current(self, x, u, r):
        return self.a*x*u*r

    def step(self, x, u, r, delt):
        nx = np.max([np.min([1, x + delt*self._dxdt(x, u, r)]), 0])
        nu = np.max([np.min([1, u + delt*self._dudt(x, u, r)]), 0])
        curr = self._current(nx, nu, r)
        return nx, nu, r, curr
