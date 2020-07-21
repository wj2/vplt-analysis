
import numpy as np
import general.plotting_styles as gps
import general.plotting as gpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import scipy.stats as sts
import os
import general.neural_analysis as na
import general.utility as u
import pref_looking.plt_analysis as pl
import pref_looking.definitions as d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import configparser

config_path = 'pref_looking/figures_sp.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

def setup():
    gps.set_paper_style(colors)
    
def figure1(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure1']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('d', 'e', 'f', 'g', 'h')
    if data is None:
        data = {}

    fsize = (4.5, 4)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    sdms_schem_grid = gs[35:70, :70]
    brain_schem_grid = gs[:35, :25]
    plt_schem_grid = gs[:35, 25:70]
    
    plt_sacc_grid = gs[:25, 85:]
    sdms_behav_grid = gs[38:60, 85:]

    sacc_latency_grid = gs[70:, :26]
    sacc_velocity_grid = gs[70:, 36:63]
    fix_latency_grid = gs[70:, 72:]

    sdms_schem_ax = f.add_subplot(sdms_schem_grid)
    plt_schem_ax = f.add_subplot(plt_schem_grid)
    brain_schem_ax = f.add_subplot(brain_schem_grid)

    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    sdms_sc = params.getcolor('sdms_sacc_color')
    plt_sc = params.getcolor('plt_sacc_color')
    
    n_boots = params.getint('n_boots_bhv')    
    if 'd' not in data.keys() and 'd' in gen_panels:
        sdms_bhv = {}
        for m, mdata in exper_data.items():
            sdms_bhv[m] = pl.compute_sdms_performance(mdata[0],
                                                      d.cgroups[m],
                                                      n_boots=n_boots)
        data['d'] = sdms_bhv

    sdms_behav_ax = f.add_subplot(sdms_behav_grid)
    if 'd' in gen_panels:
        sdms_bhv = data['d']
        for i, (m, perf_dict) in enumerate(sdms_bhv.items()):
            offset = (i - len(sdms_bhv)/2)/10
            pl.plot_sdms_performance(perf_dict, offset, sdms_behav_ax,
                                     color=monkey_colors[m])
        gpl.add_hlines(.5, sdms_behav_ax)
        gpl.clean_plot_bottom(sdms_behav_ax, keeplabels=True)

    if 'e' not in data.keys() and 'e' in 'gen_panels':
        fsps = {}
        for m, mdata in exper_data.items():
            args = [mdata[0], d.reading_params[m]]
            ps = na.apply_function_on_runs(pl.get_first_saccade_prob, args)
            ps = np.array(ps)
            ps = ps[np.logical_not(np.isnan(ps))]
            fsps[m] = ps
        data['e'] = fsps

    plt_sacc_ax = f.add_subplot(plt_sacc_grid)
    if 'e' in gen_panels:
        fsps = data['e']
        for m, ps in fsps.items():
            plt_sacc_ax.hist(ps, histtype='step',
                             color=monkey_colors[m])
            gpl.add_vlines(np.nanmean(ps), plt_sacc_ax, color=monkey_colors[m])
        plt_sacc_ax.set_xlabel('first saccade probability')
        plt_sacc_ax.set_ylabel('sessions')
        gpl.add_vlines(.5, plt_sacc_ax)
        gpl.clean_plot(plt_sacc_ax, 0)

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccin_sdms)

    latt_group = (0, 1)
    if 'f' not in data.keys() and 'f' in gen_panels:
        data['f'] = {}
        for m, dat in exper_data.items():
            p_lat, s_lat = pl.compile_saccade_latencies(dat[0], dfunc_group[m],
                                                        latt_group)
            data['f'][m] = p_lat, s_lat
            
    latt_ax = f.add_subplot(sacc_latency_grid)
    if 'f' in gen_panels:
        for m, dat in exper_data.items():
            p_lat, s_lat = data['f'][m]
            latt_ax.hist(p_lat, label='PLT', density=True, histtype='step',
                         color=plt_sc)
            latt_ax.hist(s_lat, label='sDMST', density=True, histtype='step',
                         color=sdms_sc)
            gpl.add_vlines(np.mean(p_lat), latt_ax, color=plt_sc)
            gpl.add_vlines(np.mean(s_lat), latt_ax, color=sdms_sc)
            gpl.clean_plot(latt_ax, 0)
        latt_ax.set_xlabel('first saccade\nlatency (ms)')
        gpl.make_yaxis_scale_bar(latt_ax, anchor=0, magnitude=.01, double=False,
                                 label='density', text_buff=.4)

    vel_group = (0, 1)
    if 'g' not in data.keys() and 'g' in gen_panels:
        data['g'] = {}
        for m, dat in exper_data.items():
            p_vel, s_vel = pl.compile_saccade_velocities(dat[0], dfunc_group[m],
                                                         vel_group)
            data['g'][m] = p_vel, s_vel
        
    vel_ax = f.add_subplot(sacc_velocity_grid)
    if 'g' in gen_panels:
        for m, dat in exper_data.items():
            p_vel, s_vel = data['g'][m]
            vel_ax.hist(p_vel, label='PLT', density=True, histtype='step',
                         color=plt_sc)
            vel_ax.hist(s_vel, label='sDMST', density=True, histtype='step',
                         color=sdms_sc)
            gpl.add_vlines(np.mean(p_vel), vel_ax, color=plt_sc)
            gpl.add_vlines(np.mean(s_vel), vel_ax, color=sdms_sc)
            gpl.clean_plot(vel_ax, 0)
        vel_ax.set_xlabel('saccade velocity\n(deg/s)')
        gpl.make_yaxis_scale_bar(vel_ax, anchor=0, magnitude=.002, double=False)

    fix_group = (0, 1)
    if 'h' not in data.keys() and 'h' in gen_panels:
        data['h'] = {}
        for m, dat in exper_data.items():
            p_fix, s_fix = pl.compile_fixation_latencies(dat[0], dfunc_group[m],
                                                         fix_group)
            h_bins = np.logspace(0, 3, 10)
            data['h'][m] = p_fix, s_fix, h_bins

    fix_ax = f.add_subplot(fix_latency_grid)
    if 'h' in gen_panels:
        for m, dat in exper_data.items():
            p_fix, s_fix, h_bins = data['h'][m]
            fix_ax.hist(p_fix, bins=h_bins, label='PLT', density=True, histtype='step',
                         color=plt_sc)
            fix_ax.hist(s_fix, bins=h_bins, label='sDMST', density=True, histtype='step',
                         color=sdms_sc)
            gpl.add_vlines(np.median(p_fix), fix_ax, color=plt_sc)
            gpl.add_vlines(np.median(s_fix), fix_ax, color=sdms_sc)
            gpl.clean_plot(fix_ax, 0)
        fix_ax.set_xlabel('trial initiation\nlatency (ms)')
        fix_ax.set_xscale('log')
        gpl.make_yaxis_scale_bar(fix_ax, anchor=0, magnitude=.05, double=False)

    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig1-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data    

def figure2(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure2']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e')
    if data is None:
        data = {}

    fsize = (4.5, 4.25)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    eg1_grid = gs[:30, :60]
    eg2_grid = gs[35:65, :60]

    presacc_scatter_grid = gs[70:, :35]
    sacc_scatter_grid = gs[70:, 40:75]
    diff_scatter_grid = gs[85:, 80:]

    vel1_grid = gs[:25, 70:]
    vel2_grid = gs[30:55, 70:]
    vel_scatter_grid = gs[60:85, 70:]

    # colors
    sdms_sc = params.getcolor('sdms_sacc_color')
    plt_sc = params.getcolor('plt_sacc_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}

    # fig 2 timing params
    start = params.getint('start')
    end = params.getint('end')
    binsize = params.getint('binsize')
    binstep = params.getint('binstep')
    
    min_trials = params.getint('min_trials')
    min_spks = params.getint('min_spks')
    zscore = params.getboolean('zscore')
    causal_timing = params.getboolean('causal_timing')

    mf = d.first_sacc_func

    # fig 2 trial selection params
    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccin_sdms)

    if 'abc' not in data.keys():
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing,
                                           bhv_extract_func=pl.get_sacc_vel)
            neurs[m] = out
        data['abc'] = neurs
    else:
        neurs = data['abc']    
        
    scatter_pt = params.getint('check_pt')
    eg1_key = {'Rufus':(20, 'SPK07a')}
    eg2_key = {'Rufus':(23, 'CLUSTER9')}
    eg1_ax = f.add_subplot(eg1_grid)
    eg2_ax = f.add_subplot(eg2_grid, sharex=eg1_ax, sharey=eg1_ax)
    vel1_ax = f.add_subplot(vel1_grid)
    vel2_ax = f.add_subplot(vel2_grid, sharex=vel1_ax, sharey=vel1_ax)
    if 'b' in gen_panels:
        for m, (neur, xs, bhv) in neurs.items():
            pl.plot_single_unit_eg(neur, xs, eg1_key[m], labels=('PLT', 'sDMST'),
                                   ax=eg1_ax, colors=(plt_sc, sdms_sc))
            pl.plot_single_unit_eg(neur, xs, eg2_key[m], labels=('', ''),
                                   ax=eg2_ax, colors=(plt_sc, sdms_sc))
            pl.plot_single_unit_scatter(neur, bhv, xs, scatter_pt, eg1_key[m],
                                        labels=('', ''), ax=vel1_ax,
                                        colors=(plt_sc, sdms_sc))
            pl.plot_single_unit_scatter(neur, bhv, xs, scatter_pt, eg2_key[m],
                                        labels=('', ''), ax=vel2_ax,
                                        colors=(plt_sc, sdms_sc))
        gpl.make_xaxis_scale_bar(eg2_ax, magnitude=50, label='time (ms)')
        gpl.make_yaxis_scale_bar(eg2_ax, magnitude=10, anchor=5, double=False,
                                 label='spikes/s')
        gpl.make_yaxis_scale_bar(eg1_ax, magnitude=10, anchor=5, double=False,
                                 label='spikes/s')
        gpl.clean_plot_bottom(eg1_ax)

    vel_scatter_ax = f.add_subplot(vel_scatter_grid)
    if 'c' in gen_panels:
        neurs = data['abc']
        for m, (neur, xs, bhv) in neurs.items():
            vcorr = pl.compute_velocity_firing_correlation(neur, xs, bhv,
                                                           scatter_pt)
            vcorr_pairs = np.array(list((v, vcorr[1][k])
                                        for k, v in vcorr[0].items()))
            vel_scatter_ax.plot(vcorr_pairs[:, 0], vcorr_pairs[:, 1], 'o',
                                color=monkey_colors[m])
            
        
    boots = params.getint('boots')
    lims = None
    scatter_pt = params.getint('check_pt')
    scatter_ax = f.add_subplot(sacc_scatter_grid)
    if 'c' in gen_panels:
        for m, neur in neurs.items():
            pl.plot_neuron_scatters(neur[:2], scatter_pt, lims=lims, boots=boots,
                                    ax=scatter_ax, color=monkey_colors[m])
        scatter_ax.set_xlabel('PLT activity (spikes/s)')
        scatter_ax.set_ylabel('sDMST activity (spikes/s)')

    mf = d.fixation_acquired
    if 'd' not in data.keys() and 'd' in gen_panels:
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing)
            neurs[m] = out
        data['d'] = neurs
    
    scatter_pt = 0
    pre_scatter_ax = f.add_subplot(presacc_scatter_grid)
    if 'd' in gen_panels:
        neurs = data['d']
        for m, neur in neurs.items():
            pl.plot_neuron_scatters(neur[:2], scatter_pt, lims=lims, boots=boots,
                                    ax=pre_scatter_ax, color=monkey_colors[m])

    mf1 = d.fixation_acquired
    mf2 = d.first_sacc_func
    diff_zscore = params.getboolean('diff_scatter_zscore')
    if 'e' not in data.keys() and 'e' in gen_panels:
        data['e'] = {}
        for m, mdata in exper_data.items():
            mfs1 = (mf1,)*len(dfunc_group[m])
            out1 = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs1, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=diff_zscore,
                                           causal_timing=causal_timing)
            diffs1 = pl.compute_average_diff(out1[0], out1[1], scatter_pt)
            mfs2 = (mf2,)*len(dfunc_group[m])
            out2 = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs2, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=diff_zscore,
                                           causal_timing=causal_timing)
            diffs2 = pl.compute_average_diff(out2[0], out2[1], scatter_pt)
            data['e'][m] = diffs1, diffs2
        
    scatter_pt = 0
    diff_scatter_ax = f.add_subplot(diff_scatter_grid)
    if 'e' in gen_panels:
        for m, neur in neurs.items():
            diffs1, diffs2 = data['e'][m]
            pl.plot_neuron_diffs(diffs1, diffs2, boots=boots, ax=diff_scatter_ax,
                                 color=monkey_colors[m])
        gpl.clean_plot_bottom(diff_scatter_ax, keeplabels=True)
        gpl.make_yaxis_scale_bar(diff_scatter_ax, anchor=0, magnitude=.1,
                                 double=False, label='sDMS - PLT')
        diff_scatter_ax.set_xticks([0, 1])
        diff_scatter_ax.set_xticklabels(['pre', 'img'], rotation=90)
        
    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig2-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure3(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure3']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('ab', 'c', 'de')
    if data is None:
        data = {}

    fsize = (4.5, 3)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    sacc_dec_grid = gs[:48, :50]
    sacc_pt_grid = gs[:48, 50:55]
    
    sacc_scatter_grid = gs[:43, 70:]
    
    sacc_cross_grid = gs[52:, :50]
    sacc_cross_pt_grid = gs[52:, 50:55]
    
    sacc_ang_grid = gs[57:, 70:]

    # names
    sdms_name = params.get('sdms_name')
    plt_name = params.get('plt_name')
    
    # colors
    sdms_sc = params.getcolor('sdms_sacc_color')
    plt_sc = params.getcolor('plt_sacc_color')
    comp_c = params.getcolor('comparison_color')
    cross_color = params.getcolor('hyperplane_ang_color')
    wi_color = params.getcolor('within_ang_color')
    rand_color = params.getcolor('rand_ang_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    
    # fig 3 timing params
    start = params.getint('start')
    end = params.getint('end')
    binsize = params.getint('binsize')
    binstep = params.getint('binstep')
    
    min_trials = params.getint('min_trials')
    min_spks = params.getint('min_spks')
    zscore = params.getboolean('zscore')
    resample = params.getint('resample')
    leave_out = params.getfloat('leave_out')
    equal_fold = params.getboolean('equal_fold')
    with_replace = params.getboolean('with_replace')
    kernel = params.get('kernel')

    mf = d.first_sacc_func

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_sdms, d.saccout_sdms,
                            d.saccin_plt, d.saccout_plt)
    cond_labels = (sdms_name, plt_name)
    color_dict = {sdms_name:sdms_sc, plt_name:plt_sc}
    if 'ab' not in data.keys() and 'ab' in gen_panels:
        decs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample, zscore=zscore,
                                        leave_out=leave_out, kernel=kernel,
                                        cond_labels=cond_labels,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold,
                                        min_spks=min_spks)
            decs[m] = out
        data['ab'] = decs
        
    check_pt = params.getfloat('check_pt')
    sacc_dec_ax = f.add_subplot(sacc_dec_grid)
    sacc_pt_ax = f.add_subplot(sacc_pt_grid, sharey=sacc_dec_ax)
    if 'ab' in gen_panels:
        decs = data['ab']
        pl.plot_decoding_info(decs, check_pt, sacc_dec_ax, sacc_pt_ax,
                              colors=color_dict)
        gpl.make_yaxis_scale_bar(sacc_dec_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.clean_plot_bottom(sacc_dec_ax)
        gpl.clean_plot_bottom(sacc_pt_ax)
        
    pop = True
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')
    if 'c' not in data.keys() and 'c' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['c'] = decs_pop

    sacc_scatter_ax = f.add_subplot(sacc_scatter_grid)
    if 'c' in gen_panels:
        decs_pop = data['c']
        pl.plot_svm_session_scatter(decs_pop, check_pt, sacc_scatter_ax,
                                    colordict=monkey_colors)
        sacc_scatter_ax.set_xlabel('sDMST decoding')
        sacc_scatter_ax.set_ylabel('PLT decoding')

    dfunc_group['Rufus'] = (d.saccin_sdms, d.saccin_plt,
                            d.saccout_sdms, d.saccout_plt)
    kernel = params.get('kernel_cross')
    dfunc_pairs = (0, 0, 0, 0)
    cond_labels = (('sDMST', 'PLT'),)
    if 'de' not in data.keys() and 'de' in gen_panels:
        cdecs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        dfunc_pairs=dfunc_pairs,
                                        cond_labels=cond_labels,
                                        resample=resample, zscore=zscore,
                                        equal_fold=equal_fold, kernel=kernel,
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, cross_dec=True)
            cdecs[m] = out
        data['de'] = cdecs
        
    sacc_cross_ax = f.add_subplot(sacc_cross_grid, sharex=sacc_dec_ax,
                                  sharey=sacc_dec_ax)
    sacc_cross_pt_ax = f.add_subplot(sacc_cross_pt_grid,
                                     sharey=sacc_cross_ax)
    sacc_cross_ang_ax = f.add_subplot(sacc_ang_grid)
    check_pt = params.getfloat('check_pt')
    color_dict_cross = {' -> '.join((plt_name, sdms_name)):sdms_sc,
                        ' -> '.join((sdms_name, plt_name)):plt_sc}
    if 'de' in gen_panels:
        cdecs = data['de']
        pl.plot_decoding_info(cdecs, check_pt, sacc_cross_ax,
                              sacc_cross_pt_ax, colors=color_dict_cross)
        gpl.make_yaxis_scale_bar(sacc_cross_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.make_xaxis_scale_bar(sacc_cross_ax, magnitude=50, label='time (ms)')
        gpl.clean_plot_bottom(sacc_cross_pt_ax)

        pl.plot_svm_decoding_angs(cdecs, check_pt, sacc_cross_ang_ax,
                                  cross_color=cross_color, wi_color=wi_color,
                                  rand_color=rand_color)
        sacc_cross_ang_ax.set_xlabel('decoding plane angle (degrees)')
        sacc_cross_ang_ax.set_ylabel('density')
        
    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig3-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure4(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure4']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e')
    if data is None:
        data = {}

    sdms_sc = params.getcolor('sdms_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_sc = params.getcolor('plt_sacc_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    
    pro_s = params.get('pro_style')
    anti_s = params.get('anti_style')
    
    fsize = (5.5, 6.5)
    f = plt.figure(figsize=fsize) # , constrained_layout=True)
    gs = f.add_gridspec(100, 100)

    match_dec_grid = gs[:28, :50]
    match_pt_grid = gs[:28, 50:55]
    match_scatter_grid = gs[:28, 65:]
    
    time_sdms_grid = gs[35:53, :40]
    sacc_sdms_grid = gs[58:77, :40]
    match_sdms_grid = gs[81:, :40]
    
    time_plt_grid = gs[35:53, 40:80]
    sacc_plt_grid = gs[58:77, 40:80]
    fam_plt_grid = gs[81:, 40:80]
    
    full_ev_grid = gs[35:53, 88:]
    sacc_latency_grid = gs[64:80, 88:]
    match_latency_grid = gs[84:, 88:]

    
    # fig 4 timing params
    start = params.getint('start')
    end = params.getint('end')
    binsize = params.getint('binsize')
    binstep = params.getint('binstep')
    
    min_trials = params.getint('min_trials')
    min_spks = params.getint('min_spks')
    zscore = params.getboolean('zscore')
    resample = params.getint('resample')
    leave_out = params.getfloat('leave_out')
    equal_fold = params.getboolean('equal_fold')
    with_replace = params.getboolean('with_replace')
    kernel = params.get('kernel')

    mf = d.first_sacc_func
    
    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_match_sdms, d.saccout_match_sdms,
                            d.saccin_nonmatch_sdms, d.saccout_nonmatch_sdms,
                            d.novin_saccin, d.novin_saccout,
                            d.famin_saccin, d.famin_saccout)

    dfunc_pairs = (0, 0, 0, 0, 1, 1, 1, 1)
    cond_labels = ('match-nonmatch', 'novel-familiar')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    if 'a' not in data.keys() and 'a' in gen_panels:
        decs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start, end, 
                                        binsize, binstep, min_trials,
                                        resample=resample, cond_labels=cond_labels,
                                        zscore=zscore, dfunc_pairs=dfunc_pairs,
                                        leave_out=leave_out, kernel=kernel,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold,
                                        min_spks=min_spks)
            decs[m] = out
        data['a'] = decs

    check_pt = params.getfloat('check_pt')
    match_dec_ax = f.add_subplot(match_dec_grid)
    match_pt_ax = f.add_subplot(match_pt_grid, sharey=match_dec_ax)
    if 'a' in gen_panels:
        decs = data['a']
        pl.plot_decoding_info(decs, check_pt, match_dec_ax, match_pt_ax,
                              colors=color_dict)
        gpl.make_yaxis_scale_bar(match_dec_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.make_xaxis_scale_bar(match_dec_ax, magnitude=50,
                                 label='time (ms)')
        gpl.clean_plot_bottom(match_pt_ax)

    pop = True
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')

    if 'b' not in data.keys() and 'b' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        dfunc_pairs=dfunc_pairs, kernel=kernel,
                                        resample=resample, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out)
            decs_pop[m] = out
        data['b'] = decs_pop

    match_scatter_ax = f.add_subplot(match_scatter_grid)
    if 'b' in gen_panels:
        decs_pop = data['b']
        pl.plot_svm_session_scatter(decs_pop, check_pt, match_scatter_ax,
                                    colordict=monkey_colors)
        match_scatter_ax.set_xlabel('match-nonmatch decoding')
        match_scatter_ax.set_ylabel('novel-familiar decoding')

    start = params.getint('start_dpca')
    end = params.getint('end_dpca')
    binsize = params.getint('binsize_dpca')
    binstep = params.getint('binstep_dpca')
    
    min_trials = params.getint('min_trials_dpca')
    min_spks = params.getint('min_spks_dpca')
    resample = params.getint('resample_dpca')
    with_replace = params.getboolean('with_replace')
    use_max_trials = params.getboolean('use_max_trials')
    
    mf = d.first_sacc_func

    dfunc_group_sdms = {}
    dfunc_group_sdms['Rufus'] = (d.saccin_match_sdms, d.saccout_match_sdms,
                                 d.saccin_nonmatch_sdms,
                                 d.saccout_nonmatch_sdms)
    dfunc_pts = ((0, 0), (0, 1), (1, 0), (1, 1))
    cond_labels_sdms = ('mst')

    if 'c' not in data.keys() and ('c' in gen_panels or 'e' in gen_panels):
        dpca_outs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_dpca_transform(mdata[0], dfunc_group_sdms[m], mf,
                                             start, end, binsize, binstep,
                                             min_trials, cond_labels_sdms, 
                                             dfunc_pts, use_avail_trials=True,
                                             resample=resample,
                                             use_max_trials=use_max_trials,
                                             with_replace=with_replace,
                                             min_spks=min_spks)
            dpca_outs[m] = out
        data['c'] = dpca_outs
        
    signif_level = params.getfloat('signif_level_dpca')
    
    time_sdms_ax = f.add_subplot(time_sdms_grid)
    sacc_sdms_ax = f.add_subplot(sacc_sdms_grid, sharex=time_sdms_ax)
    match_sdms_ax = f.add_subplot(match_sdms_grid, sharex=time_sdms_ax)
    ax_keys = (('t', time_sdms_ax), ('st', sacc_sdms_ax), ('mt', match_sdms_ax))
    color_dict = {'t':sdms_sc, 'st':sdms_sc, 'mt':sdms_mc}
    style_dict = {'t':np.array(((pro_s, pro_s), (anti_s, anti_s))),
                  'st':np.array(((pro_s, pro_s), (anti_s, anti_s))),
                  'mt':np.array(((pro_s, anti_s), (pro_s, anti_s)))}
    if 'c' in gen_panels:
        for m, dpca_out in data['c'].items():
            org, dpcas, xs = dpca_out
            pl.plot_dpca_kernels(dpcas[1], xs, ax_keys, dim=0,
                                 signif_level=signif_level,
                                 color_dict=color_dict, style_dict=style_dict)

        gpl.make_yaxis_scale_bar(time_sdms_ax, magnitude=20)
        gpl.make_yaxis_scale_bar(sacc_sdms_ax, label='normalized firing rate',
                                 magnitude=10, text_buff=.19)
        gpl.make_yaxis_scale_bar(match_sdms_ax, magnitude=5)
        gpl.make_xaxis_scale_bar(match_sdms_ax, magnitude=50, label='time (ms)')
        gpl.clean_plot_bottom(time_sdms_ax)
        gpl.clean_plot_bottom(sacc_sdms_ax)
    dfunc_group_plt = {}
    dfunc_group_plt['Rufus'] = (d.novin_saccin, d.novin_saccout,
                                d.famin_saccin, d.famin_saccout)
    dfunc_pts = ((0, 0), (0, 1), (1, 0), (1, 1))
    cond_labels_plt = ('fst')
    if 'd' not in data.keys() and ('d' in gen_panels or 'e' in gen_panels):
        dpca_outs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_dpca_transform(mdata[0], dfunc_group_plt[m], mf,
                                             start, end, binsize, binstep,
                                             min_trials, cond_labels_plt, 
                                             dfunc_pts, use_avail_trials=True,
                                             use_max_trials=use_max_trials,
                                             resample=resample,
                                             with_replace=with_replace,
                                             min_spks=min_spks)
            dpca_outs[m] = out
        data['d'] = dpca_outs

    time_plt_ax = f.add_subplot(time_plt_grid, sharex=time_sdms_ax)
    sacc_plt_ax = f.add_subplot(sacc_plt_grid, sharey=sacc_sdms_ax,
                                sharex=time_plt_ax)
    fam_plt_ax = f.add_subplot(fam_plt_grid, sharey=match_sdms_ax,
                               sharex=time_plt_ax)
    ax_keys = (('t', time_plt_ax), ('st', sacc_plt_ax), ('ft', fam_plt_ax))
    color_dict = {'t':plt_sc, 'st':plt_sc, 'ft':plt_fc}
    style_dict = {'t':np.array(((pro_s, pro_s), (anti_s, anti_s))),
                  'st':np.array(((pro_s, pro_s), (anti_s, anti_s))),
                  'ft':np.array(((pro_s, anti_s), (pro_s, anti_s)))}
    if 'd' in gen_panels:
        for m, dpca_out in data['d'].items():
            org, dpcas, xs = dpca_out
            pl.plot_dpca_kernels(dpcas[1], xs, ax_keys, dim=0,
                                 signif_level=signif_level,
                                 color_dict=color_dict, style_dict=style_dict)
        gpl.make_xaxis_scale_bar(fam_plt_ax, magnitude=50, label='time (ms)')
        gpl.clean_plot_bottom(time_plt_ax)
        gpl.clean_plot_bottom(sacc_plt_ax)
        gpl.clean_plot(time_plt_ax, 1)
        gpl.clean_plot(sacc_plt_ax, 1)
        gpl.clean_plot(fam_plt_ax, 1)

    full_ev_ax = f.add_subplot(full_ev_grid)
    sacc_latency_ax = f.add_subplot(sacc_latency_grid)
    match_latency_ax = f.add_subplot(match_latency_grid, sharex=sacc_latency_ax)
    if 'e' in gen_panels:
        for m, dpca_out in data['c'].items():
            _, dpcas_sdms, xs = dpca_out
            ev_sdms = pl.compute_dpca_ev(dpcas_sdms)
            full_ev_ax.hist(ev_sdms, density=True, color=sdms_sc,
                            histtype='step')
            sdms_sacc_lat = pl.compute_dpca_latencies(dpcas_sdms, xs, 'st')
            sacc_latency_ax.hist(sdms_sacc_lat, density=True, color=sdms_sc,
                                 histtype='step')
            sacc_lat_mean = np.nanmean(sdms_sacc_lat)
            sacc_latency_ax.plot([sacc_lat_mean], .05, 'o', color=sdms_sc)
            sdms_match_lat = pl.compute_dpca_latencies(dpcas_sdms, xs, 'mt')
            match_latency_ax.hist(sdms_match_lat, density=True, color=sdms_mc,
                                  histtype='step')
            match_lat_mean = np.nanmean(sdms_match_lat)
            match_latency_ax.plot([match_lat_mean], .05, 'o', color=sdms_mc)
        for m, dpca_out in data['d'].items():
            _, dpcas_plt, xs = dpca_out
            ev_plt = pl.compute_dpca_ev(dpcas_plt)
            full_ev_ax.hist(ev_plt, density=True, color=plt_sc, histtype='step')
            plt_sacc_lat = pl.compute_dpca_latencies(dpcas_plt, xs, 'st')
            sacc_latency_ax.hist(plt_sacc_lat, density=True, color=plt_sc,
                                 histtype='step')
            sacc_lat_mean = np.nanmean(plt_sacc_lat)
            sacc_latency_ax.plot([sacc_lat_mean], .05, 'o', color=plt_sc)
            plt_fam_lat = pl.compute_dpca_latencies(dpcas_plt, xs, 'ft')
            match_latency_ax.hist(plt_fam_lat, density=True, color=plt_fc,
                                  histtype='step')
            fam_lat_mean = np.nanmean(plt_fam_lat)
            match_latency_ax.plot([fam_lat_mean], .05, 'o', color=plt_fc)
        gpl.clean_plot(full_ev_ax, 0)
        gpl.clean_plot(sacc_latency_ax, 0)
        gpl.clean_plot(match_latency_ax, 0)
        gpl.clean_plot_bottom(sacc_latency_ax)
        full_ev_ax.set_xlabel('explained variance')
        match_latency_ax.set_xlabel('time (ms)')
        
    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig4-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure5(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure5']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd', 'e')
    if data is None:
        data = {}

    sdms_sc = params.getcolor('sdms_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_sc = params.getcolor('plt_sacc_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}

    fsize = (5, 4)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)
    lum_schem_grid = gs[:30, :65]
    lum_sacc_grid = gs[:30, 75:]
    
    lum_eg1_grid = gs[40:70, 42:]
    lum_eg2_grid = gs[70:, 42:]

    lum_latency_grid = gs[40:62, :30]
    lum_scatter_grid = gs[78:, :30]

    lum_schem_ax = f.add_subplot(lum_schem_grid)

    start = params.getint('start_bhv')
    end = params.getint('end_bhv')
    binsize = params.getint('binsize_bhv')
    binstep = params.getint('binstep_bhv')
    n_boots = params.getint('n_boots_bhv')
    n_trials = params.getint('n_trials_bhv')
    if 'b' not in data.keys() and 'b' in gen_panels:
        fsps = {}
        for m, mdata in exper_data.items():
            args = [mdata[0], d.reading_params[m]]
            ps = na.apply_function_on_runs(pl.get_first_saccade_prob, args,
                                           conds_ref='lum_conds',
                                           min_trials=n_trials)
            ps = np.array(ps)
            ps = ps[np.logical_not(np.isnan(ps))]
            fsps[m] = ps
        data['b'] = fsps

    lum_sacc_ax = f.add_subplot(lum_sacc_grid)
    if 'b' in gen_panels:
        fsps = data['b']
        for m, ps in fsps.items():
            lum_sacc_ax.hist(ps, density=True, histtype='step',
                             color=monkey_colors[m])
            gpl.add_vlines(np.nanmean(ps), lum_sacc_ax,
                           color=monkey_colors[m])
        lum_sacc_ax.set_xlabel('first saccade probability')
        lum_sacc_ax.set_ylabel('session density')
        gpl.add_vlines(.5, lum_sacc_ax)
        gpl.clean_plot(lum_sacc_ax, 0)
    # fig 5 timing params
    start = params.getint('start')
    end = params.getint('end')
    binsize = params.getint('binsize')
    binstep = params.getint('binstep')
    
    min_trials = params.getint('min_trials')
    min_spks = params.getint('min_spks')
    zscore = params.getboolean('zscore')
    resample = params.getint('resample')
    leave_out = params.getfloat('leave_out')
    equal_fold = params.getboolean('equal_fold')
    with_replace = params.getboolean('with_replace')
    causal_timing = params.getboolean('causal_timing')
    
    mf = d.first_sacc_func

    # fig 5 trial selection params
    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_lum, d.saccin_sdms)

    if 'cde' not in data.keys() and ('c' in gen_panels or 'd' in gen_panels
                                     or 'e' in gen_panels):
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing)
            neurs[m] = out
        data['cde'] = neurs
    elif ('c' in gen_panels or 'd' in gen_panels
          or 'e' in gen_panels):
        neurs = data['cde']    

    latt_group = (0, 1)
    sacc_ax = f.add_subplot(lum_latency_grid)
    if 'c' in gen_panels:
        for m, dat in exper_data.items():
            p_lat, s_lat = pl.compile_saccade_latencies(dat[0], dfunc_group[m],
                                                        latt_group)
            sacc_ax.hist(p_lat, label='PLT', density=True, histtype='step',
                         color=plt_sc)
            sacc_ax.hist(s_lat, label='sDMST', density=True, histtype='step',
                         color=sdms_sc)
            gpl.add_vlines(np.mean(p_lat), sacc_ax, color=plt_sc)
            gpl.add_vlines(np.mean(s_lat), sacc_ax, color=sdms_sc)
            gpl.clean_plot(sacc_ax, 0)
        sacc_ax.set_xlabel('first saccade latency (ms)')
        sacc_ax.set_ylabel('density')
        
    eg1_key = {'Rufus':(37, 'CLUSTER36')}
    eg2_key = {'Rufus':(15, 'CLUSTER9')}
    eg1_ax = f.add_subplot(lum_eg1_grid)
    eg2_ax = f.add_subplot(lum_eg2_grid, sharex=eg1_ax, sharey=eg1_ax)
    if 'd' in gen_panels:
        for m, (neur, xs) in neurs.items():
            pl.plot_single_unit_eg(neur, xs, eg1_key[m],
                                   labels=('lumPLT', 'sDMST'),
                                   ax=eg1_ax, colors=(plt_sc, sdms_sc))
            pl.plot_single_unit_eg(neur, xs, eg2_key[m],
                                   labels=('', ''),
                                   ax=eg2_ax, colors=(plt_sc, sdms_sc))
        gpl.make_yaxis_scale_bar(eg1_ax, magnitude=10, anchor=5, label='spikes/s',
                                 double=False)
        gpl.make_yaxis_scale_bar(eg2_ax, magnitude=10, anchor=5, label='spikes/s',
                                 double=False)
        gpl.make_xaxis_scale_bar(eg2_ax, magnitude=50, label='time (ms)')
        gpl.clean_plot_bottom(eg1_ax)
            
    boots = params.getint('boots')
    lims = None
    scatter_pt = 0
    scatter_ax = f.add_subplot(lum_scatter_grid)
    if 'e' in gen_panels:
        for m, neur in neurs.items():
            pl.plot_neuron_scatters(neur, scatter_pt, lims=lims, boots=boots,
                                    ax=scatter_ax, color=monkey_colors[m])
        scatter_ax.set_xlabel('lumPLT activity (spikes/s)')
        scatter_ax.set_ylabel('sDMST activity (spikes/s)')
        
    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig5-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure6(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure6']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd')
    if data is None:
        data = {}

    # fig 6 arrangement
    fsize = (4, 3)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    sdms_sc = params.getcolor('sdms_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_sc = params.getcolor('plt_sacc_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}

    lum_sacc_grid = gs[:48, :50]
    lum_sacc_pt_grid = gs[:48, 50:55]
    lum_sacc_scatter_grid = gs[:40, 70:]
    
    lum_sal_grid = gs[52:, :50]
    lum_sal_pt_grid = gs[52:, 50:55]
    lum_sal_scatter_grid = gs[60:, 70:]

    # fig 6 params
    start = params.getint('start')
    end = params.getint('end')
    binsize = params.getint('binsize')
    binstep = params.getint('binstep')
    
    min_trials = params.getint('min_trials')
    min_spks = params.getint('min_spks')
    zscore = params.getboolean('zscore')
    resample = params.getint('resample')
    leave_out = params.getfloat('leave_out')
    equal_fold = params.getboolean('equal_fold')
    with_replace = params.getboolean('with_replace')
    kernel = params.get('kernel')

    mf = d.first_sacc_func

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_sdms, d.saccout_sdms,
                            d.saccin_lum, d.saccout_lum)
    cond_labels = ('sDMST', 'lumPLT')
    color_dict = {cond_labels[0]:sdms_sc, cond_labels[1]:plt_sc}

    if 'a' not in data.keys() and 'a' in gen_panels:
        decs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample, zscore=zscore,
                                        leave_out=leave_out,
                                        cond_labels=cond_labels,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold,
                                        min_spks=min_spks, kernel=kernel)
            decs[m] = out
        data['a'] = decs
        
    check_pt = params.getfloat('check_pt')
    lum_sacc_ax = f.add_subplot(lum_sacc_grid)
    lum_sacc_pt_ax = f.add_subplot(lum_sacc_pt_grid, sharey=lum_sacc_ax)
    if 'a' in gen_panels:
        decs = data['a']
        pl.plot_decoding_info(decs, check_pt, lum_sacc_ax, lum_sacc_pt_ax,
                              colors=color_dict)
        gpl.make_yaxis_scale_bar(lum_sacc_ax, magnitude=.5, anchor=.5, label='decoding',
                                 double=False)
        gpl.clean_plot_bottom(lum_sacc_ax)
        gpl.clean_plot_bottom(lum_sacc_pt_ax)

    pop = True
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')
    if 'b' not in data.keys() and 'b' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['b'] = decs_pop

    lum_sacc_scatter_ax = f.add_subplot(lum_sacc_scatter_grid)
    if 'b' in gen_panels:
        decs_pop = data['b']
        pl.plot_svm_session_scatter(decs_pop, check_pt, lum_sacc_scatter_ax,
                                    colordict=monkey_colors)
        lum_sacc_scatter_ax.set_xlabel('sDMST decoding')
        lum_sacc_scatter_ax.set_ylabel('lumPLT decoding')

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_match_sdms, d.saccout_match_sdms,
                            d.saccin_nonmatch_sdms, d.saccout_nonmatch_sdms,
                            d.saccin_high_lum, d.saccout_high_lum,
                            d.saccin_low_lum, d.saccout_low_lum)
    dfunc_pairs = (0, 0, 0, 0, 1, 1, 1, 1)
    cond_labels = ('sDMST', 'lumPLT')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    kernel = params.get('kernel')
    if 'c' not in data.keys() and 'c' in gen_panels:
        decs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample, zscore=zscore,
                                        leave_out=leave_out,
                                        dfunc_pairs=dfunc_pairs,
                                        cond_labels=cond_labels,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold,
                                        min_spks=min_spks, kernel=kernel)
            decs[m] = out
        data['c'] = decs
        
    check_pt = params.getfloat('check_pt')
    lum_sal_ax = f.add_subplot(lum_sal_grid)
    lum_sal_pt_ax = f.add_subplot(lum_sal_pt_grid, sharey=lum_sal_ax)
    if 'c' in gen_panels:
        decs = data['c']
        pl.plot_decoding_info(decs, check_pt, lum_sal_ax, lum_sal_pt_ax,
                              colors=color_dict)
        gpl.make_yaxis_scale_bar(lum_sal_ax, magnitude=.5, anchor=.5,
                                 label='decoding', double=False)
        gpl.make_xaxis_scale_bar(lum_sal_ax, magnitude=50, label='time (ms)')
        gpl.clean_plot_bottom(lum_sal_pt_ax)

    pop = True
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')
    if 'd' not in data.keys() and 'd' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        dfunc_pairs=dfunc_pairs,
                                        resample=resample, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['d'] = decs_pop

    lum_sal_scatter_ax = f.add_subplot(lum_sal_scatter_grid)
    if 'd' in gen_panels:
        decs_pop = data['d']
        pl.plot_svm_session_scatter(decs_pop, check_pt, lum_sal_scatter_ax,
                                    colordict=monkey_colors)
        lum_sal_scatter_ax.set_xlabel('sDMST decoding')
        lum_sal_scatter_ax.set_ylabel('lumPLT decoding')
        
    bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig6-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure1_old():
    start = params.getint('start_bhv')
    end = params.getint('end_bhv')
    binsize = params.getint('binsize_bhv')
    binstep = params.getint('binstep_bhv')
    if 'e' not in data.keys() and 'e' in gen_panels:
        plt_bhv = {}
        for m, mdata in exper_data.items():
            fsps = pl.get_first_saccade_prob_bs(mdata[0], d.reading_params[m],
                                                n_boots=n_boots)
            xs, btc = pl.get_bias_timecourse(mdata[0],
                                             d.reading_params[m],
                                             start, end, binsize, binstep)
            plt_bhv[m] = (fsps, xs, btc)
        data['e'] = plt_bhv

    plt_sacc_ax = f.add_subplot(plt_sacc_grid)
    plt_tc_ax = f.add_subplot(plt_tc_grid)
    if 'e' in gen_panels:
        plt_bhv = data['e']
        labels = []
        for i, (m, bhvs) in enumerate(plt_bhv.items()):
            fsps, xs, btc = bhvs
            gpl.plot_trace_werr(np.array([i]), np.expand_dims(fsps, 1),
                                ax=plt_sacc_ax, fill=False,
                                error_func=gpl.conf95_interval)
            gpl.plot_trace_werr(xs, btc, ax=plt_tc_ax)
            labels.append(m)
        plt_sacc_ax.set_xticks(range(len(plt_bhv)))
        plt_sacc_ax.set_xticklabels(labels, rotation=90)
        gpl.add_hlines(.5, plt_sacc_ax)
        gpl.add_hlines(0, plt_tc_ax)

def figure4_old():
    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_match_sdms, d.saccout_match_sdms,
                            d.saccin_nonmatch_sdms, d.saccout_nonmatch_sdms,
                            d.saccin_match_sdms, d.saccin_nonmatch_sdms,
                            d.novin_saccin, d.famin_saccin,
                            d.saccout_match_sdms, d.saccout_nonmatch_sdms,
                            d.novin_saccout, d.famin_saccout,
                            d.saccin_nov_sdms, d.saccout_nov_sdms,
                            d.novin_saccin, d.novin_saccout,
                            d.saccin_fam_sdms, d.saccout_fam_sdms,
                            d.famin_saccin, d.famin_saccout)
    dfunc_pairs = (0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 2, 2)
    sacc_lab_plt = 'PLT sacc in-sacc out'
    sacc_lab_sdms = 'sDMS sacc in-sacc out'
    match_lab_sdms = 'sDMS match-nonmatch'
    nov_lab_sdms = 'sDMS novel-familiar'
    nov_lab_plt = 'PLT novel-familiar'
    cond_labels = (sacc_lab_sdms, match_lab_sdms, nov_lab_sdms)
    ct_zscore = params.getboolean('collapse_time_zscore')
    collapse_time = params.getboolean('collapse_time')
    zscore_collapse = params.getboolean('zscore_collapse')
    min_trials_collapse = params.getint('min_trials_collapse')
    if 'c' not in data.keys() and 'c' in gen_panels:
        decs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep,
                                        min_trials_collapse,
                                        resample=resample,
                                        cond_labels=cond_labels,
                                        zscore=zscore_collapse,
                                        dfunc_pairs=dfunc_pairs,
                                        leave_out=leave_out,
                                        collapse_time=collapse_time,
                                        with_replace=with_replace,
                                        equal_fold=equal_fold,
                                        min_spks=min_spks,
                                        collapse_time_zscore=ct_zscore)
            decs[m] = out
        data['c'] = decs

    task_traj_ax = f.add_subplot(task_traj_grid, projection='3d')
    nov_ac_ax = f.add_subplot(nov_ac_grid)
    sacc_ac_ax = f.add_subplot(sacc_ac_grid, sharey=nov_ac_ax, sharex=nov_ac_ax)
    match_ac_ax = f.add_subplot(match_ac_grid, sharey=nov_ac_ax,
                                sharex=nov_ac_ax)

    plot_pairs = ((sacc_lab_sdms, sacc_lab_plt), (nov_lab_sdms, nov_lab_plt),
                  (match_lab_sdms,))
    plot_pairs = ((sacc_lab_sdms, ), (nov_lab_sdms, ),
                  (match_lab_sdms,))
    ax_list = (sacc_ac_ax, nov_ac_ax, match_ac_ax)
    ax_names = (sacc_lab_sdms, nov_lab_sdms, match_lab_sdms)
    if 'c' in gen_panels:
        decs = data['c']
        for m, d in decs.items():
            pl.plot_svm_pairs(d, plot_pairs, ax_list)
            trajs = pl.get_svm_trajectory(d, ax_names)
            for t in trajs:
                pl.plot_svm_trajectory(t, task_traj_ax)
        task_traj_ax.view_init(90, 135)
