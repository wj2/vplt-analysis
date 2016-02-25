
import numpy as np
import cPickle as cp
import saccader as s
import saccfit as sf
import matplotlib.pyplot as plt
import scipy.stats as sts

full_params = ('sal_tc', 'off_tnov', 'lt_novpar1', 'lt_novpar2', 'st_novpar1', 
               'sacc_grow', 'samebias', 'prob_saccpar', 'prob_tc')
reduced_params = ('sal_tc', 'off_tnov', 'lt_novpar1', 'lt_novpar2', 'st_novpar1', 
                  'sacc_grow', 'samebias', 'prob_tc')

st_novfunc2 = lambda v, x, a: -sf.lt_novfunc(v, x, a)*v

def get_saccaders(pardict, lt_novfunc=sf.lt_novfunc, st_novfunc=st_novfunc2):
    sacs = []
    for i in xrange(len(pardict['lt_novpar1'])):
        # sac = s.Saccader(lt_novfunc=lt_novfunc, 
        #                  lt_novpars=(pardict['lt_novpar1'][i], 
        #                              pardict['lt_novpar2'][i]),
        #                  st_novfunc=st_novfunc,
        #                  st_novpars=(pardict['st_novpar1'][i],),
        #                  sal_timeconst=pardict['sal_tc'][i],
        #                  out_targnov=pardict['off_tnov'][i],
        #                  prob_timeconst=pardict['prob_tc'][i],
        #                  prob_diffpar=pardict['prob_saccpar'][i],
        #                  prob_growconst=pardict['sacc_grow'][i],
        #                  samebias=pardict['samebias'][i])
        sac = s.Saccader(lt_novfunc, 
                         (pardict['lt_novpar1'][i], 
                          pardict['lt_novpar2'][i]),
                         st_novfunc,
                         (pardict['lt_novpar1'][i],
                          pardict['lt_novpar2'][i]),
                         pardict['sal_tc'][i],
                         pardict['off_tnov'][i],
                         pardict['prob_tc'][i],
                         pardict['prob_saccpar'][i],
                         pardict['sacc_grow'][i],
                         pardict['samebias'][i])
        sacs.append(sac)
    return sacs

def simulate_sacs_compare(pardict, novpres=0, fampres=10000, n=100, nplot=4,
                          ft_ref_dist=sf.observed_lendistrib, bins=100):
    sacs = get_saccaders(pardict)
    for sac in sacs:
        plot_inds = np.random.randint(0, n-1, nplot)
        ts, sals, looks, sac_ts, lookprob = sac.present_many([novpres, 
                                                              fampres],
                                                             n=n, 
                                                             prestime=4000, 
                                                             tstep=1)
        f = plt.figure(figsize=(7, 5.2))
        pref_ax = f.add_subplot(2, 1, 1)
        pts, prefs, xticks = s.plot_avg_preference(looks, ax=pref_ax, show=False)
        pref_ax.set_xticklabels(xticks, visible=False)
        pref_ax.set_xlabel('')
        pref_ax.set_yticks([.2, .4])
        sal_ax = f.add_subplot(2, 1, 2, sharex=pref_ax)
        sal_ax = s.plot_salience(ts, sals[plot_inds[0]], sal_ts=xticks, ax=sal_ax)
        for x in plot_inds:
            s.plot_presentation(ts, sals[x], looks[x], sac_ts[x], lookprob[x],
                                show=False)
        fixtimes = np.concatenate(map(np.diff, sac_ts), axis=0)
        # f2 = plt.figure()
        # ax1 = f2.add_subplot(1, 2, 1)
        # ax1.hist(ft_ref_dist, bins=bins, histtype='step', label='observed', 
        #          normed=True)
        # ax1.hist(fixtimes, bins=bins, histtype='step', label='model', normed=True)
        # ax2 = f2.add_subplot(1, 2, 2)
        # ax2.hist(map(len, sac_ts), bins=bins, histtype='step', 
        #          label='observed', normed=True)        
        lp = sf.compare_fixlens(fixtimes, ft_ref_dist)
        f.tight_layout()
        f.savefig('model_pref_sal.pdf', bbox_inches='tight')
    plt.show(block=False)
    return pts, prefs, lp

def get_good_params(mcmc_dict, ref_distrib=sf.observed_lendistrib,
                    include_thr=.1):
    sfixes = mcmc_dict['sample_eyetrace'][0]
    parr = np.zeros(len(sfixes))
    for i, sfix in enumerate(sfixes):
        sfix = sfix[np.logical_not(np.isnan(sfix))]
        ks, p = sts.ks_2samp(sfix, ref_distrib)
        parr[i] = p
    keep_inds = parr > include_thr
    newd = {}
    for k in mcmc_dict:
        if k[0] != '_':
            keep = mcmc_dict[k][0][keep_inds]
            newd[k] = {}
            newd[k][0] = keep
        else:
            newd[k] = mcmc_dict[k]
    return newd
    
def get_mode_values(mcmc_files=None, mcmc_dicts=None, burn_thr=0, thin=1, 
                    mcmc_params=full_params, bins=150, mode=True, 
                    plot_hists=False):
    if mcmc_files is not None:
        mcmc_dicts = [cp.load(open(m, 'rb')) for m in mcmc_files]
    if plot_hists:
        n = np.ceil(np.sqrt(len(mcmc_params)))
        f = plt.figure()
    pardict = {}
    for i, key in enumerate(mcmc_params):
        if plot_hists:
            ax = f.add_subplot(n, n, i+1)
        pardict[key] = {}
        for j, m in enumerate(mcmc_dicts):
            keepentries = m[key][0][burn_thr::thin]
            if mode:
                vals, edges = np.histogram(keepentries, bins=bins)
                mode = np.argmax(vals)
                m = np.mean(edges[mode:mode+1])
            else:
                m = np.median(keepentries)
            pardict[key][j] = m
            if plot_hists:
                ax.hist(keepentries, bins=bins, histtype='step', 
                        label='model {}'.format(j))
                ax.set_title(key)
        if i == 0 and plot_hists:
            ax.legend(frameon=False)
    if plot_hists:
        plt.show(block=False)
    return pardict
    
        
