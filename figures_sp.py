
import numpy as np
import general.plotting_styles as gps
import general.plotting as gpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.lines as lines
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
            monkey_paths=pl.monkey_paths, config_file=config_path,
            bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure1']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('d', 'e', 'f', 'g', 'h', 'other_quant')
    if data is None:
        data = {}

    fsize = (5, 4)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    sdms_schem_grid = gs[35:70, :70]
    brain_schem_grid = gs[:35, :25]
    plt_schem_grid = gs[:35, 25:70]
    
    plt_sacc_grid = gs[:25, 90:]
    sdms_behav_grid = gs[38:60, 90:]
    comb_behav_grid = gs[70:, 90:]

    sacc_latency_grid = gs[70:, :21]
    sacc_velocity_grid = gs[70:, 31:50]
    fix_latency_grid = gs[70:, 60:79]

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
            p = perf_dict['all']
            gpl.print_mean_conf95(p, m, 'sDMST performance', n_boots=n_boots,
                                  preboot=True)
            pd = {k:v for k, v in perf_dict.items() if k != 'all'}
            offset = (i - len(sdms_bhv)/2)/5
            pl.plot_sdms_performance(pd, offset, sdms_behav_ax,
                                     color=monkey_colors[m])
        gpl.add_hlines(.5, sdms_behav_ax)
        gpl.clean_plot_bottom(sdms_behav_ax, keeplabels=True)

    if 'e' not in data.keys() and 'e' in 'gen_panels':
        fsps = {}
        fs_total = {}
        bias_dicts = {}
        for m, mdata in exper_data.items():
            sdms_func = pl.compute_simple_sdms_performance
            sdms_args = [mdata[0], d.cgroups[m]]
            sd, sdms_inds = na.apply_function_on_runs(sdms_func, sdms_args,
                                                      ret_index=True,
                                                      n_boots=n_boots)
            sdms_dict = dict(zip(sdms_inds, sd))

            args = [mdata[0], d.reading_params[m]]
            fs_all = pl.get_first_saccade_prob_bs(mdata[0],
                                                  d.reading_params[m],
                                                  n_boots=n_boots)
            fs_total[m] = fs_all
            
            args = [mdata[0], d.reading_params[m]]
            out = na.apply_function_on_runs(pl.get_first_saccade_prob, args,
                                            ret_index=True)
            ps, run_inds = out
            ps = np.array(ps)
            bias_dict = dict(zip(run_inds, np.abs(ps - .5) + .5))
            ps = ps[np.logical_not(np.isnan(ps))]
            fsps[m] = ps
            bias_dicts[m] = bias_dict, sdms_dict
        data['e'] = fsps, fs_total, bias_dicts

    if 'other_quant' in gen_panels and 'other_quant' not in data.keys():
        fs_total = {}
        for m, mdata in exper_data.items():
            args = [mdata[0], d.reading_params[m]]
            img_prob = pl.get_image_saccade_prob(mdata[0],
                                                 d.reading_params[m])
            fs_total[m] = img_prob
        data['other_quant'] = fs_total

    if 'other_quant' in gen_panels:
        img_sacc_total = data['other_quant']
        for m, img_sacc in img_sacc_total.items():
            print('{} made {:.2f} first saccades to an image'.format(m, img_sacc))
        
    plt_sacc_ax = f.add_subplot(plt_sacc_grid)
    comb_behav_ax = f.add_subplot(comb_behav_grid)
    if 'e' in gen_panels:
        fsps, fs_total, bias_dicts = data['e']
        for i, (m, ps) in enumerate(fsps.items()):
            gpl.print_mean_conf95(fs_total[m], m, 'PLT FS bias', n_boots=n_boots,
                                  preboot=True)

            x_pos = np.array([i])
            ps_arr = np.expand_dims(ps, axis=1)
            p = plt_sacc_ax.violinplot(ps, positions=x_pos,
                                       showextrema=False)
            gpl.plot_trace_werr(x_pos, ps_arr, points=True,
                                color=monkey_colors[m], ax=plt_sacc_ax,
                                error_func=gpl.conf95_interval)
            gpl.set_violin_color(p, monkey_colors[m])
            b_p, b_s = bias_dicts[m]
            b_zip = np.array(list((p, b_s[k]) for k, p in b_p.items()))
            comb_behav_ax.plot(b_zip[:, 0], b_zip[:, 1], 'o',
                               color=monkey_colors[m])
        plt_sacc_ax.set_ylabel('P(first saccade to novel)')
        comb_behav_ax.set_xlabel('PLT bias')
        comb_behav_ax.set_ylabel('sDMST accuracy')
        comb_behav_ax.set_xticks([.5, .7])
        comb_behav_ax.set_yticks([.5, .8])
        plt_sacc_ax.set_xticks(range(len(fsps)))
        plt_sacc_ax.set_xticklabels(list(fsps.keys()), rotation=90)
        gpl.clean_plot_bottom(plt_sacc_ax, keeplabels=True)
        gpl.add_hlines(.5, plt_sacc_ax)
        gpl.clean_plot(plt_sacc_ax, 0)
        gpl.clean_plot(comb_behav_ax, 0)

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccin_sdms)
    dfunc_group['Neville'] = (d.saccin_plt_n, d.saccin_sdms_n)

    latt_group = (0, 1)
    if 'f' not in data.keys() and 'f' in gen_panels:
        data['f'] = {}
        for m, dat in exper_data.items():
            p_lat, s_lat = pl.compile_saccade_latencies(dat[0], dfunc_group[m],
                                                        latt_group)
            data['f'][m] = p_lat, s_lat
            
    latt_ax = f.add_subplot(sacc_latency_grid)
    if 'f' in gen_panels:
        p_lat_all = np.array([])
        s_lat_all = np.array([])
        for m, dat in exper_data.items():
            p_lat, s_lat = data['f'][m]
            gpl.print_mean_conf95(p_lat[0], m, 'saccade latency, PLT',
                                  n_boots=n_boots)
            gpl.print_mean_conf95(s_lat[0], m, 'saccade latency, sDMST',
                                  n_boots=n_boots)
            gpl.print_diff_conf95(p_lat[0], s_lat[0], m, 'latency diff',
                                  n_boots=n_boots)
            p_lat_all = np.concatenate((p_lat_all, p_lat[0]))
            s_lat_all = np.concatenate((s_lat_all, s_lat[0]))
        gpl.print_mean_conf95(p_lat_all, 'combined', 'saccade latency, PLT',
                              n_boots=n_boots)
        gpl.print_mean_conf95(s_lat_all, 'combined', 'saccade latency, sDMST',
                              n_boots=n_boots)
        gpl.print_diff_conf95(p_lat_all, s_lat_all, 'combined', 'latency diff',
                              n_boots=n_boots)
        latt_ax.hist(p_lat_all, label='PLT', density=True, histtype='step',
                     color=plt_sc)
        latt_ax.hist(s_lat_all, label='sDMST', density=True, histtype='step',
                     color=sdms_sc)
        gpl.clean_plot(latt_ax, 0)
        latt_ax.set_xlabel('first saccade\nlatency (ms)')
        plt_legend = lines.Line2D([], [], color=plt_sc,
                             label='PLT')
        sdms_legend = lines.Line2D([], [], color=sdms_sc,
                              label='sDMST')
        latt_ax.legend(handles=(plt_legend, sdms_legend), frameon=False)
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
        p_vel_all = np.array([])
        s_vel_all = np.array([])
        for m, dat in exper_data.items():
            p_vel, s_vel = data['g'][m]
            p_vel = np.array(p_vel)
            s_vel = np.array(s_vel)
            gpl.print_mean_conf95(p_vel, m, 'saccade velocity, PLT',
                              n_boots=n_boots)
            gpl.print_mean_conf95(s_vel, m, 'saccade velocity, sDMST',
                              n_boots=n_boots)
            gpl.print_diff_conf95(p_vel, s_vel, m, 'velocity diff',
                                  n_boots=n_boots)

            p_vel_all = np.concatenate((p_vel_all, p_vel))
            s_vel_all = np.concatenate((s_vel_all, s_vel))
            
        gpl.print_mean_conf95(p_vel_all, 'combined', 'saccade velocity, PLT',
                              n_boots=n_boots)
        gpl.print_mean_conf95(s_vel_all, 'combined', 'saccade velocity, sDMST',
                              n_boots=n_boots)
        gpl.print_diff_conf95(p_vel_all, s_vel_all, 'combined', 'velocity diff',
                              n_boots=n_boots)

        vel_ax.hist(p_vel_all, label='PLT', density=True, histtype='step',
                    color=plt_sc)
        vel_ax.hist(s_vel_all, label='sDMST', density=True, histtype='step',
                    color=sdms_sc)
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
        p_fix_all = np.array([])
        s_fix_all = np.array([])
        for m, dat in exper_data.items():
            p_fix, s_fix, h_bins = data['h'][m]
            gpl.print_mean_conf95(p_fix[0], m, 'fix latency, PLT',
                                  n_boots=n_boots, func=np.mean)
            gpl.print_mean_conf95(s_fix[0], m, 'fix latency, sDMST',
                              n_boots=n_boots, func=np.mean)
            gpl.print_diff_conf95(p_fix[0], s_fix[0], m, 'fix diff',
                                  n_boots=n_boots, func=np.mean)
            p_fix_all = np.concatenate((p_fix_all, p_fix[0]))
            s_fix_all = np.concatenate((s_fix_all, s_fix[0]))
        gpl.print_mean_conf95(p_fix_all, 'combined', 'fix latency, PLT',
                              n_boots=n_boots, func=np.mean)
        gpl.print_mean_conf95(s_fix_all, 'combined', 'fix latency, sDMST',
                              n_boots=n_boots, func=np.mean)
        gpl.print_diff_conf95(p_fix_all, s_fix_all, 'combined', 'fix diff',
                              n_boots=n_boots, func=np.mean)
        fix_ax.hist(p_fix_all, label='PLT', density=True,
                    histtype='step', color=plt_sc)
        fix_ax.hist(s_fix_all,  label='sDMST', density=True,
                    histtype='step', color=sdms_sc)
        gpl.clean_plot(fix_ax, 0)
        fix_ax.set_xlabel('trial initiation\nlatency (ms)')
        fix_ax.set_xlim([0, 500])
        gpl.make_yaxis_scale_bar(fix_ax, anchor=0, magnitude=.005, double=False)

    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig1-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data    

def figure2(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path,
            rand_eg=False, bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure2']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'cd', 'e', 'f')
    if data is None:
        data = {}

    fsize = (4.5, 4.25)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    eg1_sacc_grid = gs[:23, :60]
    eg2_sacc_grid = gs[27:48, :60]

    eg1_prior_grid = gs[52:73, :60]
    eg2_prior_grid = gs[77:, :60]

    presacc_scatter_grid = gs[30:47, 70:]
    sacc_scatter_grid = gs[55:72, 70:]
    diff_scatter_grid = gs[80:, 78:92]

    vel_scatter_grid = gs[:16, 76:94]

    # colors
    sdms_sc = params.getcolor('sdms_sacc_color')
    plt_sc = params.getcolor('plt_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_fc = params.getcolor('plt_fam_color')
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
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccout_plt,
                            d.saccin_sdms, d.saccout_sdms)
    dfunc_group['Neville'] = (d.saccin_plt_n, d.saccout_plt_n,
                              d.saccin_sdms_n, d.saccout_sdms_n)

    if 'a' not in data.keys():
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing,
                                           bhv_extract_func=pl.get_sacc_vel)
            neurs[m] = out
        data['a'] = neurs
    else:
        neurs = data['a']    
        
    scatter_pt = params.getint('check_pt')
    eg1_info = ('Rufus', (15, 'CLUSTER9'))
    key_r = list(neurs['Rufus'][0][0])
    key_n = list(neurs['Neville'][0][0])
    
    if rand_eg:
        eg1_ind = int(np.random.choice(len(key_r), 1))
        eg2_ind = int(np.random.choice(len(key_n), 1))
    else:
        eg1_ind = 1
        eg2_ind = 221        

    eg1_info = ('Rufus', key_r[eg1_ind])
    eg2_info = ('Neville', key_n[eg2_ind])

    eg1_ax = f.add_subplot(eg1_sacc_grid)
    eg2_ax = f.add_subplot(eg2_sacc_grid, sharex=eg1_ax, sharey=eg1_ax)
    if 'a' in gen_panels:
        m1, eg1_key = eg1_info
        m2, eg2_key = eg2_info
        ls = ('solid', 'dashed', 'solid', 'dashed')
        labels_1 = ('PLT', '', 'sDMST', '')
        labels_2 = ('', '', '', '')
        colors = (plt_sc, plt_sc, sdms_sc, sdms_sc)
        pl.plot_single_unit_eg(neurs[m1][0], neurs[m1][1], eg1_key,
                               labels=labels_1, linestyles=ls,
                               ax=eg1_ax, colors=colors)
        pl.plot_single_unit_eg(neurs[m2][0], neurs[m2][1], eg2_key,
                               labels=labels_2, ax=eg2_ax, linestyles=ls,
                               colors=colors)
        # gpl.make_xaxis_scale_bar(eg2_ax, magnitude=50, label='time (ms)')
        gpl.make_yaxis_scale_bar(eg2_ax, magnitude=35, anchor=5, double=False,
                                 label='spikes/s')
        gpl.make_yaxis_scale_bar(eg1_ax, magnitude=35, anchor=5, double=False,
                                 label='spikes/s')
        gpl.clean_plot_bottom(eg1_ax)
        gpl.clean_plot_bottom(eg2_ax)

        dfunc_group = {}
    dfunc_group['Rufus'] = (d.novin_saccin, d.famin_saccin,
                            d.saccin_match_sdms, d.saccin_nonmatch_sdms)
    dfunc_group['Neville'] = (d.novin_saccin_n, d.famin_saccin_n,
                              d.saccin_match_sdms_n, d.saccout_match_sdms_n)

    if 'b' not in data.keys():
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing,
                                           bhv_extract_func=pl.get_sacc_vel)
            neurs[m] = out
        data['b'] = neurs
    else:
        neurs = data['b']    
        
    scatter_pt = params.getint('check_pt')
    eg1_info = ('Rufus', (15, 'CLUSTER9'))
    key_r = list(neurs['Rufus'][0][0])
    key_n = list(neurs['Neville'][0][0])
    
    if rand_eg:
        eg1_ind = int(np.random.choice(len(key_r), 1))
        eg2_ind = int(np.random.choice(len(key_n), 1))
        print('Rufus eg index: {}'.format(eg1_ind))
        print('Neville eg index: {}'.format(eg2_ind))
    else:
        eg1_ind = 82
        eg2_ind = 75        

    eg1_info = ('Rufus', key_r[eg1_ind])
    eg2_info = ('Neville', key_n[eg2_ind])

    eg1_p_ax = f.add_subplot(eg1_prior_grid, sharex=eg1_ax)
    eg2_p_ax = f.add_subplot(eg2_prior_grid, sharex=eg1_p_ax, sharey=eg1_p_ax)
    if 'b' in gen_panels:
        m1, eg1_key = eg1_info
        m2, eg2_key = eg2_info
        ls = ('solid', 'dashed', 'solid', 'dashed')
        labels_1 = ('PLT', '', 'sDMST', '')
        labels_2 = ('', '', '', '')
        colors = (plt_fc, plt_fc, sdms_mc, sdms_mc)
        pl.plot_single_unit_eg(neurs[m1][0], neurs[m1][1], eg1_key,
                               labels=labels_1, linestyles=ls,
                               ax=eg1_p_ax, colors=colors)
        pl.plot_single_unit_eg(neurs[m2][0], neurs[m2][1], eg2_key,
                               labels=labels_2, ax=eg2_p_ax, linestyles=ls,
                               colors=colors)
        gpl.make_xaxis_scale_bar(eg2_p_ax, magnitude=50, label='time (ms)')
        gpl.make_yaxis_scale_bar(eg2_p_ax, magnitude=25, anchor=5, double=False,
                                 label='spikes/s')
        gpl.make_yaxis_scale_bar(eg1_p_ax, magnitude=25, anchor=5, double=False,
                                 label='spikes/s')
        gpl.clean_plot_bottom(eg1_p_ax)


    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccin_sdms)
    dfunc_group['Neville'] = (d.saccin_plt_n, d.saccin_sdms_n)

    if 'cd' not in data.keys():
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing,
                                           bhv_extract_func=pl.get_sacc_vel)
            neurs[m] = out
        data['cd'] = neurs
    else:
        neurs = data['cd']    
        
    vel_scatter_ax = f.add_subplot(vel_scatter_grid)
    if 'cd' in gen_panels:
        neurs = data['cd']
        for m, (neur, xs, bhv) in neurs.items():
            vcorr = pl.compute_velocity_firing_correlation(neur, xs, bhv,
                                                           scatter_pt)
            vcorr_pairs = np.array(list((v, vcorr[1][k])
                                        for k, v in vcorr[0].items()))
            gpl.print_mean_conf95(vcorr_pairs[:, 0], m, 'sDMST vcorr')
            gpl.print_mean_conf95(vcorr_pairs[:, 1], m, 'PLT vcorr')
            vel_scatter_ax.plot(vcorr_pairs[:, 0], vcorr_pairs[:, 1], 'o',
                                color=monkey_colors[m])
        gpl.clean_plot(vel_scatter_ax, 0)
        gpl.make_yaxis_scale_bar(vel_scatter_ax, magnitude=.3, anchor=0,
                                 label='PLT velocity\ncorrelation',
                                 text_buff=.65)
        gpl.make_xaxis_scale_bar(vel_scatter_ax, magnitude=.3, anchor=0,
                                 label='sDMST velocity\ncorrelation',
                                 text_buff=.4)
            
        
    boots = params.getint('boots')
    lims = None
    scatter_pt = params.getint('check_pt')
    scatter_ax = f.add_subplot(sacc_scatter_grid)
    if 'cd' in gen_panels:
        for m, neur in neurs.items():
            print('{}: {} neurons, task'.format(m, len(neur[0][0])))
            pl.plot_neuron_scatters(neur[:2], scatter_pt, lims=lims, boots=boots,
                                    ax=scatter_ax, color=monkey_colors[m])
        scatter_ax.set_xlabel('PLT activity (spikes/s)')

    mf = d.fixation_acquired
    if 'e' not in data.keys() and 'e' in gen_panels:
        neurs = {}
        for m, mdata in exper_data.items():
            mfs = (mf,)*len(dfunc_group[m])
            out = na.organize_spiking_data(mdata[0], dfunc_group[m], mfs, start,
                                           end, binsize, binstep=binstep, 
                                           min_trials=min_trials, zscore=zscore,
                                           causal_timing=causal_timing)
            neurs[m] = out
        data['e'] = neurs
    
    scatter_pt = 0
    pre_scatter_ax = f.add_subplot(presacc_scatter_grid, sharex=scatter_ax,
                                   sharey=scatter_ax)
    if 'e' in gen_panels:
        neurs = data['e']
        for m, neur in neurs.items():
            print('{}: {} neurons, pre'.format(m, len(neur[0][1])))
            pl.plot_neuron_scatters(neur[:2], scatter_pt, lims=lims, boots=boots,
                                    ax=pre_scatter_ax, color=monkey_colors[m])
        pre_scatter_ax.set_ylabel('sDMST activity\n(spikes/s)')

    mf1 = d.fixation_acquired
    mf2 = d.first_sacc_func
    diff_zscore = params.getboolean('diff_scatter_zscore')
    if 'f' not in data.keys() and 'f' in gen_panels:
        data['f'] = {}
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
            data['f'][m] = diffs1, diffs2
        
    scatter_pt = 0
    diff_scatter_ax = f.add_subplot(diff_scatter_grid)
    if 'f' in gen_panels:
        for i, (m, neur) in enumerate(neurs.items()):
            diffs1, diffs2 = data['f'][m]
            offset = (i - len(neurs)/2)/5
            pre_diffs = np.array(list(diffs1.values()))
            task_diffs = np.array(list(diffs2.values()))
            n_greater_pre = np.sum(pre_diffs > 0)
            n_greater_task = np.sum(task_diffs > 0)
            s_temp = '{}: {} neurons fire more in sDMST, {}'
            print(s_temp.format(m, n_greater_pre, 'pre'))
            print(s_temp.format(m, n_greater_task, 'task'))
            out = pl.plot_neuron_diffs(diffs1, diffs2, boots=boots,
                                       ax=diff_scatter_ax,
                                       color=monkey_colors[m], offset=offset)
            d1_boots, d2_boots = out
            gpl.print_mean_conf95(d1_boots, m, 'z-scored diff, pre',
                                  preboot=True)
            gpl.print_mean_conf95(d2_boots, m, 'z-scored diff, task',
                                  preboot=True)
            
        gpl.clean_plot_bottom(diff_scatter_ax, keeplabels=True)
        diff_scatter_ax.set_xticks([0, 1])
        diff_scatter_ax.set_xticklabels(['pre', 'img'], rotation=90)
        y_lab = 'sDMST - PLT\n(z-scored)'
        gpl.make_yaxis_scale_bar(diff_scatter_ax, anchor=0, magnitude=.1,
                                 double=False, label=y_lab,
                                 text_buff=.9)
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig2-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure2a(data=None, gen_panels=None, exper_data=None,
             monkey_paths=pl.monkey_paths, config_file=config_path,
             rand_eg=False, bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure2a']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'cd', 'e')
    if data is None:
        data = {}

    fsize = (4.5, 3)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    loss_gs = gs[:40, 50:62]
    sacc_gs = gs[:40, 71:83]
    prior_gs = gs[:40, 88:]

    sacc_task_gs = gs[70:, 50:70]
    prior_task_gs = gs[70:, 80:]

    glm_axs = (loss_gs, sacc_gs, prior_gs,
               sacc_task_gs, prior_task_gs)
    
    # colors
    sdms_sc = params.getcolor('sdms_sacc_color')
    plt_sc = params.getcolor('plt_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}

    # fig 2 timing params

    start_glm = params.getint('start_glm')
    end_glm = params.getint('end_glm')
    binsize_glm = params.getint('binsize_glm')
    binstep_glm = params.getint('binstep_glm')
    perms_glm = params.getint('perms_glm')
    min_trials_glm = params.getint('min_trials_glm')
    zscore_glm = params.getboolean('zscore_glm')
    t_glm = params.getint('time_glm')
    p_thr = params.getfloat('signif_level_glm')

    dfunc_group_both = {}
    dfunc_group_both['Rufus'] = (d.saccin_match_sdms, d.saccout_match_g_sdms,
                                 d.saccin_nonmatch_g_sdms,
                                 d.saccout_nonmatch_sdms,
                                 d.novin_saccin, d.novin_saccout,
                                 d.famin_saccin, d.famin_saccout)
    dfunc_group_both['Neville'] = (d.saccin_match_sdms_n,
                                   d.saccout_match_g_sdms_n,
                                   d.saccin_nonmatch_g_sdms_n,
                                   d.saccout_nonmatch_sdms_n,
                                   d.novin_saccin_n, d.novin_saccout_n,
                                   d.famin_saccin_n, d.famin_saccout_n)

    # swap priority
    glm_shape = ((0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
    # original
    # glm_shape = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    #              (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
    glm_labels_both = ('task', 'priority', 'sacc')

    time_cent = params.getfloat('time_cent')
    time_wid = params.getfloat('time_wid')
    av_only = params.getboolean('av_only')
    adapt_delta = params.getfloat('adapt_delta')
    req_trials = params.getint('min_trials_lm')
    mf = d.first_sacc_func

    max_fit = 2
    if 'cd' not in data.keys():
        glm_dat = {}
        for m, mdata in exper_data.items():
            out_both = pl.compare_models_single(mdata[0], time_cent, time_wid, mf,
                                             dfunc_group_both[m], glm_shape,
                                             glm_labels_both, av_only=av_only,
                                             adapt_delta=adapt_delta,
                                             min_trials=req_trials,
                                             max_fit=max_fit)
            glm_dat[m] = out_both
        data['cd'] = glm_dat

    # find good way to plot model comparison result
    monkey_offsets = {'Neville':.1,
                      'Rufus':-.1}
    if 'cd' in gen_panels:
        neurs = data['cd']
        loss_ax = f.add_subplot(loss_gs)
        sacc_ax = f.add_subplot(sacc_gs)
        prior_ax = f.add_subplot(prior_gs, sharey=sacc_ax, sharex=sacc_ax)
        sacc_task_ax = f.add_subplot(sacc_task_gs, aspect='equal')
        prior_task_ax = f.add_subplot(prior_task_gs, aspect='equal',
                                      sharey=sacc_task_ax,
                                      sharex=sacc_task_ax)
        axs = (loss_ax, sacc_ax, prior_ax,
               sacc_task_ax, prior_task_ax)
        make_scale_bars = True
        for m, d_both in neurs.items():
            fits_m, comps_m, labs_m = d_both
            out = pl.summarize_model_comparison(fits_m, comps_m, labs_m,
                                                monkey_name=m)
            m_loss, _, _, m_metrics = out
            pl.plot_model_comparison(m_loss, m_metrics, axs=axs,
                                     color=monkey_colors[m],
                                     monkey_offset=monkey_offsets[m],
                                     make_scale_bars=make_scale_bars,
                                     monkey_name=m)
            make_scale_bars = False
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig2a-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data    

def figure3(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path, bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure3']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('ab', 'c', 'de', 'e', 'f')
    if data is None:
        data = {}

    fsize = (4.5, 4.5)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    sacc_dec_grid = gs[:32, :50]
    sacc_pt_grid = gs[:32, 50:55]

    sacc_scatter_grid = gs[:28, 70:]
    
    sacc_cross_grid = gs[34:65, :50]
    sacc_cross_pt_grid = gs[34:65, 50:55]
    
    dec_bhv_plt_grid = gs[38:60, 70:84]
    dec_bhv_sdmst_grid = gs[38:60, 86:]

    match_dec_grid = gs[67:, :50]
    match_pt_grid = gs[67:, 50:55]

    match_scatter_grid = gs[72:, 70:]
    
    # names
    sdms_name = params.get('sdms_name')
    plt_name = params.get('plt_name')
    
    # colors
    sdms_sc = params.getcolor('sdms_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color')
    plt_sc = params.getcolor('plt_sacc_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    cross_color = params.getcolor('hyperplane_ang_color')
    wi_color = params.getcolor('within_ang_color')
    rand_color = params.getcolor('rand_ang_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    cp_color = params.getcolor('cp_bar_color')
    
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
    combine_monkeys = params.getboolean('combine_monkeys')
    cumulative = params.getboolean('cumulative')
    mf = d.first_sacc_func

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_sdms, d.saccout_sdms,
                            d.saccin_plt, d.saccout_plt)
    dfunc_group['Neville'] = (d.saccin_sdms_n, d.saccout_sdms_n,
                              d.saccin_plt_n, d.saccout_plt_n)
    cond_labels = (sdms_name, plt_name)
    color_dict = {sdms_name:sdms_sc, plt_name:plt_sc}
    if 'ab' not in data.keys() and 'ab' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               cumulative=cumulative)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['ab'] = decs

    check_pt = params.getfloat('check_pt')
    sacc_dec_ax = f.add_subplot(sacc_dec_grid)
    sacc_pt_ax = f.add_subplot(sacc_pt_grid, sharey=sacc_dec_ax)
    if 'ab' in gen_panels:
        decs = data['ab']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, sacc_dec_ax, sacc_pt_ax,
                                    colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'saccade decoding, {}', cond_labels)
        gpl.make_yaxis_scale_bar(sacc_dec_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.clean_plot_bottom(sacc_dec_ax)
        gpl.clean_plot_bottom(sacc_pt_ax)

    pop = True
    resample_pop = params.getint('resample_pop')
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')
    if 'c' not in data.keys() and 'c' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample_pop, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['c'] = decs_pop
    if 'c-' not in data.keys() and 'c' in gen_panels:
        fsps = {}
        for m, mdata in exper_data.items():
            args = [mdata[0], d.reading_params[m]]
            sb, sb_inds = na.apply_function_on_runs(pl.get_side_bias,
                                                    args, ret_index=True)
            
            args = [mdata[0], d.reading_params[m]]
            ps, run_inds = na.apply_function_on_runs(pl.get_first_saccade_prob,
                                                     args, ret_index=True)

            sdms_func = pl.compute_simple_sdms_performance
            sdms_args = [mdata[0], d.cgroups[m]]
            sd, sdms_inds = na.apply_function_on_runs(sdms_func, sdms_args,
                                                      ret_index=True, n_boots=1)
            bias_dict = dict(zip(run_inds, ps))
            side_dict = dict(zip(sb_inds, sb))
            sdms_dict = dict(zip(sdms_inds, sd))
            fsps[m] = (bias_dict, sdms_dict, side_dict)
        data['c-'] = fsps

    sacc_scatter_ax = f.add_subplot(sacc_scatter_grid)
    bias_plt_ax = f.add_subplot(dec_bhv_plt_grid)
    bias_sdmst_ax = f.add_subplot(dec_bhv_sdmst_grid, sharey=bias_plt_ax)
    if 'c' in gen_panels:
        decs_pop = data['c']
        bias_dicts = data['c-']
        _, pts = pl.plot_svm_session_scatter(decs_pop, check_pt, sacc_scatter_ax,
                                             colordict=monkey_colors)
        pl.print_svm_scatter(pts, 'saccade pop')
        pl.plot_svm_bhv_scatter(decs_pop, bias_dicts, check_pt,
                                bias_plt_ax, bias_sdmst_ax,
                                colordict=monkey_colors, print_=True)
        gpl.clean_plot(bias_plt_ax, 0)
        gpl.clean_plot(bias_sdmst_ax, 1)
        bias_plt_ax.set_xticks([.5, .6])
        bias_plt_ax.set_xlabel('PLT bias')
        bias_plt_ax.set_ylabel('decoding')
        bias_sdmst_ax.set_xlabel('sDMST\naccuracy')
        bias_sdmst_ax.set_xticks([.75, .85])
        sacc_scatter_ax.set_xlabel('sDMST decoding')
        sacc_scatter_ax.set_ylabel('PLT decoding')

    dfunc_group['Rufus'] = (d.saccin_sdms, d.saccin_plt,
                            d.saccout_sdms, d.saccout_plt)
    dfunc_group['Neville'] = (d.saccin_sdms_n, d.saccin_plt_n,
                              d.saccout_sdms_n, d.saccout_plt_n)
    kernel = params.get('kernel_cross')
    zscore = params.getboolean('zscore')
    equal_fold = params.getboolean('equal_fold')

    dfunc_pairs = (0, 0, 0, 0)
    cond_labels = (('sDMST', 'PLT'),)
    if 'de' not in data.keys() and 'de' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               cross_dec=True,
                                               dfunc_pairs=dfunc_pairs)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel, cross_dec=True,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['de'] = decs
        
    sacc_cross_ax = f.add_subplot(sacc_cross_grid, sharex=sacc_dec_ax,
                                  sharey=sacc_dec_ax)
    sacc_cross_pt_ax = f.add_subplot(sacc_cross_pt_grid,
                                     sharey=sacc_cross_ax)
    # sacc_cross_ang_ax = f.add_subplot(sacc_ang_grid)
    check_pt = params.getfloat('check_pt')
    color_dict_cross = {' -> '.join((plt_name, sdms_name)):sdms_sc,
                        ' -> '.join((sdms_name, plt_name)):plt_sc}
    if 'de' in gen_panels:
        cdecs = data['de']
        pl.print_decoding_info(cdecs)
        pts = pl.plot_decoding_info(cdecs, check_pt, sacc_cross_ax,
                              sacc_cross_pt_ax, colors=color_dict_cross)
        pl.print_svm_decoding_diff(pts, 'saccade cross, {}', cond_labels[0])
        gpl.make_yaxis_scale_bar(sacc_cross_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.clean_plot_bottom(sacc_cross_ax)
        gpl.clean_plot_bottom(sacc_cross_pt_ax)

    mf = d.first_sacc_func
    
    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_match_sdms, d.saccout_match_g_sdms,
                            d.saccin_nonmatch_g_sdms, d.saccout_nonmatch_sdms,
                            d.novin_saccin, d.novin_saccout,
                            d.famin_saccin, d.famin_saccout)
    dfunc_group['Neville'] = (d.saccin_match_sdms_n, d.saccout_match_g_sdms_n,
                              d.saccin_nonmatch_g_sdms_n,
                              d.saccout_nonmatch_sdms_n,
                              d.novin_saccin_n, d.novin_saccout_n,
                              d.famin_saccin_n, d.famin_saccout_n)

    dfunc_pairs = (0, 0, 0, 0, 1, 1, 1, 1)
    cond_labels = ('match-nonmatch', 'novel-familiar')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    if 'e' not in data.keys() and 'e' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               dfunc_pairs=dfunc_pairs)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['e'] = decs

    check_pt = params.getfloat('check_pt_img')
    match_dec_ax = f.add_subplot(match_dec_grid)
    match_pt_ax = f.add_subplot(match_pt_grid, sharey=match_dec_ax)
    if 'e' in gen_panels:
        decs = data['e']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, match_dec_ax, match_pt_ax,
                              colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'match decoding, {}', cond_labels)
        gpl.make_yaxis_scale_bar(match_dec_ax, anchor=.5, magnitude=.5,
                                 double=False, label='decoding')
        gpl.make_xaxis_scale_bar(match_dec_ax, magnitude=20,
                                 label='time (ms)')
        gpl.clean_plot_bottom(match_pt_ax)

    pop = True
    resample_pop = params.getint('resample_pop')
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')

    if 'f' not in data.keys() and 'f' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        dfunc_pairs=dfunc_pairs, kernel=kernel,
                                        resample=resample_pop, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out)
            decs_pop[m] = out
        data['f'] = decs_pop

    match_scatter_ax = f.add_subplot(match_scatter_grid)
    if 'f' in gen_panels:
        decs_pop = data['f']
        _, pts = pl.plot_svm_session_scatter(decs_pop, check_pt,
                                             match_scatter_ax,
                                             colordict=monkey_colors)
        pl.print_svm_scatter(pts, 'match pop', combine=True)
        match_scatter_ax.set_xlabel('match-nonmatch decoding')
        match_scatter_ax.set_ylabel('novel-familiar decoding')
        
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig3-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure4(data=None, gen_panels=None, exper_data=None, plot_monkey='Rufus',
            monkey_paths=pl.monkey_paths, config_file=config_path, bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure4']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('c', 'd', 'e')
    if data is None:
        data = {}

    sdms_tc = params.getcolor('sdms_gen_color')
    sdms_sc = params.getcolor('sdms_sacc_color')
    sdms_mc = params.getcolor('sdms_match_color') 
    plt_tc = params.getcolor('plt_gen_color')
    plt_sc = params.getcolor('plt_sacc_color')
    plt_fc = params.getcolor('plt_fam_color')
    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    
    pro_s = params.get('pro_style')
    anti_s = params.get('anti_style')
    
    fsize = (5, 4.5)
    f = plt.figure(figsize=fsize) # , constrained_layout=True)
    gs = f.add_gridspec(100, 100)
    
    time_comb_grid = gs[:30, :25]
    sacc_comb_grid = gs[35:63, :25]
    match_comb_grid = gs[68:, :25]
    
    time_sdms_grid = gs[:30, 30:55]
    sacc_sdms_grid = gs[35:63, 30:55]
    fam_sdms_grid = gs[68:, 30:55]

    time_plt_grid = gs[:30, 60:85]
    sacc_plt_grid = gs[35:63, 60:85]
    fam_plt_grid = gs[68:, 60:85]
    
    full_ev_grid = gs[:24, 92:]
    sacc_latency_grid = gs[40:64, 92:]
    match_latency_grid = gs[76:, 92:]

    
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
    combine_monkeys = params.getboolean('combine_monkeys')

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

    dfunc_group_comb = {}
    dfunc_group_comb['Rufus'] = (d.saccin_match_sdms, d.saccout_match_g_sdms,
                                 d.saccin_nonmatch_g_sdms,
                                 d.saccout_nonmatch_sdms,
                                 d.novin_saccin, d.novin_saccout,
                                 d.famin_saccin, d.famin_saccout)
    dfunc_group_comb['Neville'] = (d.saccin_match_sdms_n, d.saccout_match_g_sdms_n,
                                   d.saccin_nonmatch_g_sdms_n,
                                   d.saccout_nonmatch_sdms_n,
                                   d.novin_saccin_n, d.novin_saccout_n,
                                   d.famin_saccin_n, d.famin_saccout_n)
    dfunc_pts = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
    cond_labels_comb = ('bpst')

    if 'cd' not in data.keys() and ('c' in gen_panels or 'd' in gen_panels
                                    or 'e' in gen_panels):
        dpca_outs = {}
        for m, mdata in exper_data.items():
            out = pl.organize_dpca_transform(mdata[0], dfunc_group_comb[m], mf,
                                             start, end, binsize, binstep,
                                             min_trials, cond_labels_comb, 
                                             dfunc_pts, use_avail_trials=True,
                                             resample=resample,
                                             use_max_trials=use_max_trials,
                                             with_replace=with_replace,
                                             min_spks=min_spks)
            dpca_outs[m] = out
        data['cd'] = dpca_outs

    signif_level = params.getfloat('signif_level_dpca')
    
    time_comb_ax = f.add_subplot(time_comb_grid)
    sacc_comb_ax = f.add_subplot(sacc_comb_grid)
    match_comb_ax = f.add_subplot(match_comb_grid)
    ax_keys = (('t', time_comb_ax), ('st', sacc_comb_ax), ('pt', match_comb_ax))
    color_dict = {'t':sdms_tc, 'st':sdms_sc, 'pt':sdms_mc}
    style_dict = {'t':np.array((((pro_s, pro_s), (anti_s, anti_s)),
                                ((pro_s, pro_s), (anti_s, anti_s)))),
                  'st':np.array((((pro_s, anti_s), (pro_s, anti_s)),
                                 ((pro_s, anti_s), (pro_s, anti_s)))),
                  'pt':np.array((((pro_s, pro_s), (anti_s, anti_s)),
                                 ((pro_s, pro_s), (anti_s, anti_s))))}
    dim_dict = {'t':0, 'st':1, 'pt':1}
    signif_h_dict = {'t':0, 'st':-4.5, 'pt':-2}
    if 'c' in gen_panels:
        for m, dpca_out in data['cd'].items():
            if m == plot_monkey:
                org, dpcas, xs = dpca_out
                pl.plot_dpca_kernels(dpcas[0], xs, ax_keys, dim_dict=dim_dict,
                                     signif_level=signif_level,
                                     color_dict=color_dict, style_dict=style_dict,
                                     signif_heights=signif_h_dict)

        comb_time_yscale = 10
        comb_sacc_yscale = 10
        comb_match_yscale = 2
        gpl.make_yaxis_scale_bar(time_comb_ax, magnitude=comb_time_yscale)
        gpl.make_yaxis_scale_bar(sacc_comb_ax, label='normalized firing rate',
                                 magnitude=comb_sacc_yscale, text_buff=.19)
        gpl.make_yaxis_scale_bar(match_comb_ax, magnitude=comb_match_yscale)
                
        comb_time_xscale = 20
        comb_sacc_xscale = 5
        comb_match_xscale = 2
        gpl.make_xaxis_scale_bar(time_comb_ax, magnitude=comb_time_xscale, label='')
        gpl.make_xaxis_scale_bar(sacc_comb_ax, magnitude=comb_sacc_xscale, label='')
        gpl.make_xaxis_scale_bar(match_comb_ax, magnitude=comb_match_xscale,
                                 label='normalized firing rate')

        # gpl.clean_plot_bottom(time_comb_ax)
        time_comb_ax.set_title('sDMST', loc='left')
        time_comb_ax.text(0, .98, 'condition-independent',
                          transform=time_comb_ax.transAxes)
        sacc_comb_ax.text(0, .98, 'saccade-dependent',
                          transform=sacc_comb_ax.transAxes)
        match_comb_ax.text(0, .98, 'image-dependent',
                          transform=match_comb_ax.transAxes)
        # gpl.clean_plot_bottom(sacc_sdms_ax)

    style_dict = {'bt':np.array(((pro_s, pro_s), (anti_s, anti_s))),
                  'bst':np.array(((pro_s, anti_s), (pro_s, anti_s))),
                  'bpt':np.array(((pro_s, pro_s), (anti_s, anti_s)))}
    dim_dict = {'bt':0, 'bst':1, 'bpt':1}
    signif_h_dict = {'bt':0, 'bst':-4.6, 'bpt':-2}
    
    time_sdms_ax = f.add_subplot(time_sdms_grid)
    sacc_sdms_ax = f.add_subplot(sacc_sdms_grid)
    fam_sdms_ax = f.add_subplot(fam_sdms_grid)
    color_dict = {'bt':sdms_tc, 'bst':sdms_sc, 'bpt':sdms_mc}
    ax_keys = (('bt', time_sdms_ax), ('bst', sacc_sdms_ax), ('bpt', fam_sdms_ax))
    task_ind_sdms = 0                    
    if 'd' in gen_panels:
        for m, dpca_out in data['cd'].items():
            if m == plot_monkey:
                org, dpcas, xs = dpca_out
                pl.plot_dpca_kernels(dpcas[0], xs, ax_keys, dim_dict=dim_dict,
                                     signif_level=signif_level,
                                     color_dict=color_dict, style_dict=style_dict,
                                     signif_heights=signif_h_dict,
                                     task_ind=task_ind_sdms)

        task_time_yscale = 5
        task_sacc_yscale = 2
        task_match_yscale = 1
        gpl.make_yaxis_scale_bar(time_sdms_ax, magnitude=task_time_yscale)
        gpl.make_yaxis_scale_bar(sacc_sdms_ax, label='normalized firing rate',
                                 magnitude=task_sacc_yscale, text_buff=.19)
        gpl.make_yaxis_scale_bar(fam_sdms_ax, magnitude=task_match_yscale)

        task_time_xscale = 5
        task_sacc_xscale = 2
        task_match_xscale = 2
        gpl.make_xaxis_scale_bar(time_sdms_ax, magnitude=task_time_xscale,
                                 label='')
        gpl.make_xaxis_scale_bar(sacc_sdms_ax, magnitude=task_sacc_xscale,
                                 label='')
        gpl.make_xaxis_scale_bar(fam_sdms_ax, magnitude=task_match_xscale,
                                 label='normalized firing rate')

        time_sdms_ax.set_title('sDMST')
        gpl.clean_plot(time_sdms_ax, 1)
        gpl.clean_plot(sacc_sdms_ax, 1)
        gpl.clean_plot(fam_sdms_ax, 1)
        pro_sleg = lines.Line2D([], [], color=plt_tc, linestyle=pro_s,
                             label='saccade in')
        anti_sleg = lines.Line2D([], [], color=plt_tc, linestyle=anti_s,
                              label='saccade out')
        sacc_sdms_ax.legend(handles=(pro_sleg, anti_sleg), frameon=False,
                           loc='upper left')
        pro_mleg = lines.Line2D([], [], color=plt_tc, linestyle=pro_s,
                             label='nov/match in')
        anti_mleg = lines.Line2D([], [], color=plt_tc, linestyle=anti_s,
                              label='fam/nonmatch in')
        fam_sdms_ax.legend(handles=(pro_mleg, anti_mleg), frameon=False,
                           loc='upper left')

    time_plt_ax = f.add_subplot(time_plt_grid, sharex=time_sdms_ax,
                                sharey=time_sdms_ax)
    sacc_plt_ax = f.add_subplot(sacc_plt_grid, sharey=sacc_sdms_ax,
                                sharex=sacc_sdms_ax)
    fam_plt_ax = f.add_subplot(fam_plt_grid, sharey=fam_sdms_ax,
                               sharex=fam_sdms_ax)
    color_dict = {'bt':plt_tc, 'bst':plt_sc, 'bpt':plt_fc}
    ax_keys = (('bt', time_plt_ax), ('bst', sacc_plt_ax), ('bpt', fam_plt_ax))       
    task_ind_plt = 1
    if 'd' in gen_panels:
        for m, dpca_out in data['cd'].items():
            if m == plot_monkey:
                org, dpcas, xs = dpca_out
                pl.plot_dpca_kernels(dpcas[0], xs, ax_keys, dim_dict=dim_dict,
                                     signif_level=signif_level,
                                     color_dict=color_dict, style_dict=style_dict,
                                     signif_heights=signif_h_dict,
                                     task_ind=task_ind_plt)
                
        gpl.make_xaxis_scale_bar(time_plt_ax, magnitude=task_time_xscale,
                                 label='')
        gpl.make_xaxis_scale_bar(sacc_plt_ax, magnitude=task_sacc_xscale,
                                 label='')
        gpl.make_xaxis_scale_bar(fam_plt_ax, magnitude=task_match_xscale,
                                 label='normalized firing rate')

        # gpl.clean_plot_bottom(time_plt_ax)
        time_plt_ax.set_title('PLT')
        # gpl.clean_plot_bottom(sacc_plt_ax)
        gpl.clean_plot(time_plt_ax, 1)
        gpl.clean_plot(sacc_plt_ax, 1)
        gpl.clean_plot(fam_plt_ax, 1)
        pro_sleg = lines.Line2D([], [], color=plt_tc, linestyle=pro_s,
                             label='saccade in')
        anti_sleg = lines.Line2D([], [], color=plt_tc, linestyle=anti_s,
                              label='saccade out')
        sacc_plt_ax.legend(handles=(pro_sleg, anti_sleg), frameon=False,
                           loc='upper left')
        pro_mleg = lines.Line2D([], [], color=plt_tc, linestyle=pro_s,
                             label='nov/match in')
        anti_mleg = lines.Line2D([], [], color=plt_tc, linestyle=anti_s,
                              label='fam/nonmatch in')
        fam_plt_ax.legend(handles=(pro_mleg, anti_mleg), frameon=False,
                           loc='upper left')

    offset = .05
    full_ev_ax = f.add_subplot(full_ev_grid)
    sacc_latency_ax = f.add_subplot(sacc_latency_grid, sharex=full_ev_ax)
    match_latency_ax = f.add_subplot(match_latency_grid, sharex=sacc_latency_ax)
    sdms_cent = 0
    plt_cent = .5
    if 'e' in gen_panels:
        p_temp = '{}: {:.2f} ms {} encoding latency, {}'
        for i, (m, dpca_out) in enumerate(data['c'].items()):
            _, dpcas_sdms, xs = dpca_out
            ev_sdms = pl.compute_dpca_ev(dpcas_sdms)*100
            
            print('{}: {:.2f}% variance explained, {}'.format(m, ev_sdms[0],
                                                              'sDMST'))
            x_val = [sdms_cent - offset + 2*offset*i]
            gpl.plot_trace_werr(x_val, ev_sdms,
                                points=True, color=monkey_colors[m], ax=full_ev_ax)
            sdms_sacc_lat = pl.compute_dpca_latencies(dpcas_sdms, xs, 'st')
            if not np.all(np.isnan(sdms_sacc_lat)):
                # sacc_latency_ax.hist(sdms_sacc_lat, density=True, color=sdms_sc,
                #                      histtype='step')
                # sacc_latency_ax.plot([sacc_lat_mean], .05, 'o', color=sdms_sc
                sacc_lat_mean = np.nanmean(sdms_sacc_lat)
                gpl.plot_trace_werr(x_val, sdms_sacc_lat, points=True,
                                    ax=sacc_latency_ax,
                                    color=monkey_colors[m])
                
                print(p_temp.format(m, sacc_lat_mean, 'sacc', 'sDMST'))
            sdms_match_lat = pl.compute_dpca_latencies(dpcas_sdms, xs, 'mt')
            if not np.all(np.isnan(sdms_match_lat)):
                # match_latency_ax.hist(sdms_match_lat, density=True, color=sdms_mc,
                # histtype='step')
                # match_latency_ax.plot([match_lat_mean], .05, 'o', color=sdms_mc)
                match_lat_mean = np.nanmean(sdms_match_lat)
                gpl.plot_trace_werr(x_val, sdms_match_lat, points=True,
                                    ax=match_latency_ax, color=monkey_colors[m])
                print(p_temp.format(m, match_lat_mean, 'match', 'sDMST'))
                
        for i, (m, dpca_out) in enumerate(data['d'].items()):
            _, dpcas_plt, xs = dpca_out
            ev_plt = pl.compute_dpca_ev(dpcas_plt)*100
            print('{}: {:.2f}% variance explained, {}'.format(m, ev_plt[0]*100,
                                                              'PLT'))
            x_val = [plt_cent - offset + 2*offset*i]
            gpl.plot_trace_werr(x_val, ev_plt, points=True,
                                color=monkey_colors[m], ax=full_ev_ax)

            plt_sacc_lat = pl.compute_dpca_latencies(dpcas_plt, xs, 'st')
            if not np.all(np.isnan(plt_sacc_lat)):
                # sacc_latency_ax.hist(plt_sacc_lat, density=True, color=plt_sc,
                #                      histtype='step')
                # sacc_latency_ax.plot([sacc_lat_mean], .05, 'o', color=plt_sc)
                gpl.plot_trace_werr(x_val, plt_sacc_lat, ax=sacc_latency_ax,
                                    points=True, color=monkey_colors[m])
                sacc_lat_mean = np.nanmean(plt_sacc_lat)
                print(p_temp.format(m, sacc_lat_mean, 'sacc', 'PLT'))
            plt_fam_lat = pl.compute_dpca_latencies(dpcas_plt, xs, 'ft')
            if not np.all(np.isnan(plt_fam_lat)):
                # match_latency_ax.hist(plt_fam_lat, density=True, color=plt_fc,
                #                       histtype='step')
                # match_latency_ax.plot([fam_lat_mean], .05, 'o', color=plt_fc)
                gpl.plot_trace_werr(x_val, plt_fam_lat, ax=match_latency_ax,
                                    points=True, color=monkey_colors[m])
                fam_lat_mean = np.nanmean(plt_fam_lat)
                print(p_temp.format(m, fam_lat_mean, 'fam', 'PLT'))
        gpl.clean_plot(full_ev_ax, 0)
        gpl.clean_plot_bottom(full_ev_ax, keeplabels=True)

        full_ev_ax.set_xticks([sdms_cent, plt_cent])
        full_ev_ax.set_xticklabels(['sDMST', 'PLT'], rotation=90)
        full_ev_ax.set_xlim([-offset*4 + sdms_cent, plt_cent + offset*4])

        
        gpl.clean_plot(sacc_latency_ax, 0)
        gpl.clean_plot(match_latency_ax, 0)
        gpl.clean_plot_bottom(sacc_latency_ax)
        gpl.clean_plot_bottom(match_latency_ax, keeplabels=True)
        match_latency_ax.set_xticks([sdms_cent, plt_cent])
        match_latency_ax.set_xticklabels(['sDMST', 'PLT'], rotation=90)
        match_latency_ax.set_xlim([-offset*4 + sdms_cent, plt_cent + offset*4])        
        
        gpl.make_yaxis_scale_bar(full_ev_ax, anchor=0, magnitude=5,
                                 label='percent explained\nvariance',
                                 double=False, text_buff=1)

        tb = .95
        gpl.make_yaxis_scale_bar(sacc_latency_ax, anchor=0, magnitude=10,
                                 label='latency (ms)', text_buff=tb)
        gpl.make_yaxis_scale_bar(match_latency_ax, anchor=0, magnitude=50,
                                 label='latency (ms)', double=False, text_buff=tb)

        
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig4-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure5(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path, bf=None):
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
            ps = na.apply_function_on_runs(pl.get_first_saccade_prob_bs, args,
                                           conds_ref='lum_conds',
                                           min_trials=n_trials, n_boots=100)
            ps = np.array(ps)
            intervs = np.array(list(gpl.conf95_interval(p) for p in ps))
            upper = intervs[:, 0] < .5
            lower = intervs[:, 1] > .5
            print('{} sig sessions'.format(m),
                  np.sum(np.logical_or(upper, lower)))
            ps = np.nanmean(ps, axis=1)
            ps = ps[np.logical_not(np.isnan(ps))]
            fsps[m] = ps
        data['b'] = fsps

    lum_sacc_ax = f.add_subplot(lum_sacc_grid)
    if 'b' in gen_panels:
        fsps = data['b']
        for i, (m, ps) in enumerate(fsps.items()):
            x_pos = np.array([i])
            ps_arr = np.expand_dims(ps, axis=1)
            p = lum_sacc_ax.violinplot(ps_arr[:, 0], positions=x_pos,
                                       showextrema=False)
            gpl.plot_trace_werr(x_pos, ps_arr, points=True,
                                color=monkey_colors[m], ax=lum_sacc_ax,
                                error_func=gpl.conf95_interval)
            gpl.set_violin_color(p, monkey_colors[m])
        lum_sacc_ax.set_ylabel('luminance bias')
        lum_sacc_ax.set_xticks(range(len(fsps)))
        lum_sacc_ax.set_xticklabels(list(fsps.keys()), rotation=90)
        gpl.clean_plot_bottom(lum_sacc_ax, keeplabels=True)
        gpl.add_hlines(.5, lum_sacc_ax)
        gpl.clean_plot(lum_sacc_ax, 0)
        lum_sacc_ax.set_yticks([.4, .5, .6])
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
    dfunc_group['Neville'] = (d.saccin_lum_n, d.saccin_sdms_n)

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
    latt_ax = f.add_subplot(lum_latency_grid)
    if 'c' in gen_panels:
        p_lat_all = np.array([])
        s_lat_all = np.array([])
        for m, dat in exper_data.items():
            p_lat, s_lat = pl.compile_saccade_latencies(dat[0], dfunc_group[m],
                                                        latt_group)
            p_lat = np.array(p_lat[0]) 
            s_lat = np.array(s_lat[0])
            gpl.print_mean_conf95(p_lat, m, 'saccade latency, lumPLT',
                              n_boots=n_boots)
            gpl.print_mean_conf95(s_lat, m, 'saccade latency, sDMST',
                              n_boots=n_boots)
            gpl.print_diff_conf95(p_lat, s_lat, m, 'latency diff',
                                  n_boots=n_boots)
            p_lat_all = np.concatenate((p_lat_all, p_lat))
            s_lat_all = np.concatenate((s_lat_all, s_lat))
        gpl.print_mean_conf95(p_lat_all, 'combined', 'saccade latency, lumPLT',
                              n_boots=n_boots)
        gpl.print_mean_conf95(s_lat_all, 'combined', 'saccade latency, sDMST',
                              n_boots=n_boots)
        gpl.print_diff_conf95(p_lat_all, s_lat_all, 'combined', 'latency diff',
                              n_boots=n_boots)
        latt_ax.hist(p_lat_all, label='PLT', density=True, histtype='step',
                     color=plt_sc)
        latt_ax.hist(s_lat_all, label='sDMST', density=True, histtype='step',
                     color=sdms_sc)
        gpl.clean_plot(latt_ax, 0)
        latt_ax.set_xlabel('first saccade\nlatency (ms)')
        latt_ax.set_xlim([0, 600])
        gpl.make_yaxis_scale_bar(latt_ax, anchor=0, magnitude=.01, double=False,
                                 label='density', text_buff=.4)
        
    eg1_info = ('Rufus', (15, 'CLUSTER9'))
    eg2_info = ('Neville', (6, 'CLUSTER28'))
    eg1_ax = f.add_subplot(lum_eg1_grid)
    eg2_ax = f.add_subplot(lum_eg2_grid, sharex=eg1_ax, sharey=eg1_ax)
    if 'd' in gen_panels:
        m1, eg1_key = eg1_info
        m2, eg2_key = eg2_info
        print('{}: {} neurons, task'.format(m1, len(neurs[m1][0][0])))
        print('{}: {} neurons, task'.format(m2, len(neurs[m2][0][0])))
        pl.plot_single_unit_eg(neurs[m1][0], neurs[m1][1], eg1_key,
                               labels=('lumPLT', 'sDMST'),
                               ax=eg1_ax, colors=(plt_sc, sdms_sc))
        pl.plot_single_unit_eg(neurs[m2][0], neurs[m2][1], eg2_key,
                               labels=('', ''), ax=eg2_ax,
                               colors=(plt_sc, sdms_sc))
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
            _, pts = pl.plot_neuron_scatters(neur, scatter_pt, lims=lims,
                                             boots=boots, ax=scatter_ax,
                                             color=monkey_colors[m])
            task_diffs = np.diff(np.mean(pts[..., 0], axis=2), axis=1)
            n_greater_task = np.sum(task_diffs > 0)
            s_temp = '{}: {} neurons fire more in sDMST, {}'
            print(s_temp.format(m, n_greater_task, 'task'))


        scatter_ax.set_xlabel('lumPLT activity (spikes/s)')
        scatter_ax.set_ylabel('sDMST activity (spikes/s)')
        
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig5-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure6(data=None, gen_panels=None, exper_data=None,
            monkey_paths=pl.monkey_paths, config_file=config_path, bf=None):
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
    combine_monkeys = params.getboolean('combine_monkeys')

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
    dfunc_group['Neville'] = (d.saccin_sdms_n, d.saccout_sdms_n,
                            d.saccin_lum_n, d.saccout_lum_n)
    cond_labels = ('sDMST', 'lumPLT')
    color_dict = {cond_labels[0]:sdms_sc, cond_labels[1]:plt_sc}
    if 'a' not in data.keys() and 'a' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['a'] = decs
        
    check_pt = params.getfloat('check_pt')
    lum_sacc_ax = f.add_subplot(lum_sacc_grid)
    lum_sacc_pt_ax = f.add_subplot(lum_sacc_pt_grid, sharey=lum_sacc_ax)
    if 'a' in gen_panels:
        decs = data['a']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, lum_sacc_ax, lum_sacc_pt_ax,
                              colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'lum saccade decoding, {}', cond_labels)
        gpl.make_yaxis_scale_bar(lum_sacc_ax, magnitude=.5, anchor=.5,
                                 label='decoding', double=False)
        gpl.clean_plot_bottom(lum_sacc_ax)
        gpl.clean_plot_bottom(lum_sacc_pt_ax)
    pop = True
    resample_pop = params.getint('resample_pop')
    min_population = params.getint('min_population')
    zscore = params.getboolean('zscore_pop')
    equal_fold = params.getboolean('equal_fold_pop')
    kernel = params.get('kernel_pop')
    if 'b' not in data.keys() and 'b' in gen_panels:
        decs_pop = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs(mdata[0], dfunc_group[m], mf, start,
                                        end, binsize, binstep, min_trials,
                                        resample=resample_pop, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['b'] = decs_pop

    lum_sacc_scatter_ax = f.add_subplot(lum_sacc_scatter_grid)
    if 'b' in gen_panels:
        decs_pop = data['b']
        _, pts = pl.plot_svm_session_scatter(decs_pop, check_pt,
                                             lum_sacc_scatter_ax,
                                             colordict=monkey_colors)
        pl.print_svm_scatter(pts, 'sacc pop', combine=True)
        lum_sacc_scatter_ax.set_xlabel('sDMST decoding')
        lum_sacc_scatter_ax.set_ylabel('lumPLT decoding')

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.saccin_match_sdms, d.saccout_match_sdms,
                            d.saccin_nonmatch_sdms, d.saccout_nonmatch_sdms,
                            d.saccin_high_lum, d.saccout_high_lum,
                            d.saccin_low_lum, d.saccout_low_lum)
    dfunc_group['Neville'] = (d.saccin_match_sdms_n, d.saccout_match_sdms_n,
                              d.saccin_nonmatch_sdms_n, d.saccout_nonmatch_sdms_n,
                              d.saccin_high_lum_n, d.saccout_high_lum_n,
                              d.saccin_low_lum_n, d.saccout_low_lum_n)
    dfunc_pairs = (0, 0, 0, 0, 1, 1, 1, 1)
    cond_labels = ('sDMST', 'lumPLT')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    kernel = params.get('kernel')
    zscore = params.getboolean('zscore')
    equal_fold = params.getboolean('equal_fold')
    if 'c' not in data.keys() and 'c' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               dfunc_pairs=dfunc_pairs)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['c'] = decs
        
    check_pt = params.getfloat('check_pt_img')
    lum_sal_ax = f.add_subplot(lum_sal_grid)
    lum_sal_pt_ax = f.add_subplot(lum_sal_pt_grid, sharey=lum_sal_ax)
    if 'c' in gen_panels:
        decs = data['c']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, lum_sal_ax, lum_sal_pt_ax,
                              colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'lum decoding, {}', cond_labels)
        gpl.make_yaxis_scale_bar(lum_sal_ax, magnitude=.5, anchor=.5,
                                 label='decoding', double=False)
        gpl.make_xaxis_scale_bar(lum_sal_ax, magnitude=20, label='time (ms)')
        gpl.clean_plot_bottom(lum_sal_pt_ax)

    pop = True
    resample_pop = params.getint('resample_pop')
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
                                        resample=resample_pop, zscore=zscore,
                                        pop=pop, equal_fold=equal_fold, 
                                        with_replace=with_replace,
                                        min_population=min_population,
                                        leave_out=leave_out, kernel=kernel)
            decs_pop[m] = out
        data['d'] = decs_pop

    lum_sal_scatter_ax = f.add_subplot(lum_sal_scatter_grid)
    if 'd' in gen_panels:
        decs_pop = data['d']
        _, pts = pl.plot_svm_session_scatter(decs_pop, check_pt,
                                             lum_sal_scatter_ax,
                                             colordict=monkey_colors)
        pl.print_svm_scatter(pts, 'lum pop', combine=True)
        lum_sal_scatter_ax.set_xlabel('sDMST decoding')
        lum_sal_scatter_ax.set_ylabel('lumPLT decoding')
        
    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'fig6-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure_si_saccdec(data=None, gen_panels=None, exper_data=None,
                      monkey_paths=pl.monkey_paths, config_file=config_path,
                      bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure_saccdec']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a',)
    if data is None:
        data = {}

    # fig SI-BHV arrangement
    fsize = (4, 3)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    dfuncs = {'Rufus':(d.saccin_plt, d.saccin_sdms),
              'Neville':(d.saccin_plt_n, d.saccin_sdms_n)}
    mf = d.first_sacc_func
    pretime = params.getint('pretime')
    posttime = params.getint('posttime')
    if 'a' not in data.keys() and 'a' in gen_panels:
        data['a'] = {}
        for m, (mdata, _) in exper_data.items():
            mask1 = dfuncs[m][0](mdata)
            mask2 = dfuncs[m][1](mdata)
            mdata_masked1 = mdata[mask1]
            mdata_masked2 = mdata[mask2]
            ts1 = mf(mdata_masked1)
            ts2 = mf(mdata_masked2)
            out = pl.decode_task_from_eyepos(mdata_masked1['eyepos'], ts1,
                                             mdata_masked2['eyepos'], ts2,
                                             pretime, posttime,
                                             class_weight='balanced')
            data['a'][m] = out
            print(out)
            print(out.shape)
    if 'a' in gen_panels:
        pass
    return data

def figure_si_bhv(data=None, gen_panels=None, exper_data=None,
                  monkey_paths=pl.monkey_paths, config_file=config_path,
                  bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure_bhv']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('abc',)
    if data is None:
        data = {}

    # fig SI-BHV arrangement
    fsize = (4, 3)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}
    
    combine_monkeys = params.getboolean('combine_monkeys')

    fs_bias_grid = gs[:40, :30]
    lt_bias_grid = gs[:40, 36:63]
    mt_corr_grid = gs[:40, 69:]
    
    fs_time_scatter_grid = gs[55:, :43]
    fs_sal_scatter_grid = gs[55:, 57:]

    if 'abc' not in data.keys() and 'abc' in gen_panels:
        sbps = {}
        fsps = {}
        tbps = {}
        ltime_totals = {}
        bias_dicts = {}
        for m, mdata in exper_data.items():
            
            bias_func = pl.get_side_bias
            bias_args = [mdata[0], d.reading_params[m]]
            sb, sb_inds = na.apply_function_on_runs(bias_func, bias_args,
                                                    ret_index=True)
            sb = np.array(sb)
            sb_dict = dict(zip(sb_inds, np.abs(sb - .5) + .5))
            sb = sb[np.logical_not(np.isnan(sb))]
            sbps[m] = sb
            
            args = [mdata[0], d.reading_params[m]]
            out = na.apply_function_on_runs(pl.get_first_saccade_prob, args,
                                            ret_index=True)
            ps, run_inds = out
            ps = np.array(ps)
            fs_bias_dict = dict(zip(run_inds, np.abs(ps - .5) + .5))
            ps = ps[np.logical_not(np.isnan(ps))]
            fsps[m] = ps

            ltime_total = pl.get_time_bias(mdata[0], d.reading_params[m])
            ltime_totals[m] = ltime_total
            
            args = [mdata[0], d.reading_params[m]]
            out = na.apply_function_on_runs(pl.get_time_bias, args,
                                            ret_index=True)
            ps, run_inds = out
            ps = np.array(ps)
            tb_bias_dict = dict(zip(run_inds, np.abs(ps)))
            ps = ps[np.logical_not(np.isnan(ps))]
            tbps[m] = ps

            bias_dicts[m] = fs_bias_dict, tb_bias_dict, sb_dict
        data['abc'] = fsps, tbps, sbps, bias_dicts, ltime_totals

    fs_bias_ax = f.add_subplot(fs_bias_grid)
    lt_bias_ax = f.add_subplot(lt_bias_grid, sharey=fs_bias_ax)
    mt_corr_ax = f.add_subplot(mt_corr_grid, sharey=fs_bias_ax)
    
    fs_lt_scatter_ax = f.add_subplot(fs_time_scatter_grid)
    fs_sal_scatter_ax = f.add_subplot(fs_sal_scatter_grid)
    
    if 'abc' in gen_panels:
        fsps, tbps, sbps, bias_dicts, ltime_totals = data['abc']
        for m, mdata in exper_data.items():
            fs_bias_ax.hist(fsps[m], histtype='step',
                            color=monkey_colors[m])
            lt_bias_ax.hist(tbps[m], histtype='step',
                            color=monkey_colors[m])
            mt_corr_ax.hist(sbps[m], histtype='step',
                            color=monkey_colors[m])

            gpl.print_mean_conf95(tbps[m], m, 'looking time')
            gpl.print_mean_conf95(sbps[m], m, 'side bias')
            gpl.print_corr_conf95(tbps[m], fsps[m], m, 'time-FS corr')
            gpl.print_corr_conf95(sbps[m], fsps[m], m, 'side-FS corr')

            b_p, b_t, b_s = bias_dicts[m]
            b_zip = np.array(list((p, b_t[k], b_s[k])
                                  for k, p in b_p.items()))
            fs_lt_scatter_ax.plot(b_zip[:, 0], b_zip[:, 1], 'o',
                                  color=monkey_colors[m])
            fs_sal_scatter_ax.plot(b_zip[:, 0], b_zip[:, 2], 'o',
                                  color=monkey_colors[m])
        gpl.make_yaxis_scale_bar(fs_bias_ax, magnitude=5, anchor=0,
                                 label='sessions', double=False,
                                 text_buff=.25)
        fs_bias_ax.set_xlabel('first saccade bias')
        fs_bias_ax.set_xticks([.4, .5, .6])
        lt_bias_ax.set_xlabel('looking time bias')
        lt_bias_ax.set_xticks([-.5, 0, .5])
        mt_corr_ax.set_xlabel('side bias')
        mt_corr_ax.set_xticks([.3, .5, .7])
        gpl.make_xaxis_scale_bar(fs_lt_scatter_ax, magnitude=.1, anchor=.5,
                                 label='first saccade bias', double=False,
                                 text_buff=.25)
        gpl.make_yaxis_scale_bar(fs_lt_scatter_ax, magnitude=.3, anchor=0,
                                 label='looking time bias', double=False,
                                 text_buff=.25)
        gpl.make_xaxis_scale_bar(fs_sal_scatter_ax, magnitude=.1, anchor=.5,
                                 label='first saccade bias', double=False,
                                 text_buff=.25)
        gpl.make_yaxis_scale_bar(fs_sal_scatter_ax, magnitude=.1, anchor=.5,
                                 label='side bias', double=False,
                                 text_buff=.25)
        gpl.clean_plot(fs_bias_ax, 0)
        gpl.clean_plot(lt_bias_ax, 1)
        gpl.clean_plot(mt_corr_ax, 1)
        gpl.clean_plot(fs_lt_scatter_ax, 0)
        gpl.clean_plot(fs_sal_scatter_ax, 0)

    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'si-bhv-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure_si_sal(data=None, gen_panels=None, exper_data=None,
                  monkey_paths=pl.monkey_paths, config_file=config_path,
                  bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure_sal']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('ab',)
    if data is None:
        data = {}

    # fig SI-SAL arrangement
    fsize = (4.25, 2)
    f = plt.figure(figsize=fsize)
    gs = f.add_gridspec(100, 100)

    comp_c = params.getcolor('comparison_color')
    r_color = params.getcolor('rufus_color')
    n_color = params.getcolor('neville_color')
    monkey_colors = {'Rufus':r_color, 'Neville':n_color}

    r_folder = params.get('rufus_familiar_folder')
    n_folder = params.get('neville_familiar_folder')
    monkey_folders = {'Rufus':r_folder, 'Neville':n_folder}
    n_boots = params.getint('n_boots_bhv')    
    
    combine_monkeys = params.getboolean('combine_monkeys')

    time_sal_scatter_grid = gs[:, :45]
    fs_sal_scatter_grid = gs[:, 55:]

    if 'ab' not in data.keys() and 'ab' in gen_panels:
        bias_dicts = {}
        for m, mdata in exper_data.items():
                        
            args = [mdata[0], d.reading_params[m]]
            out = na.apply_function_on_runs(pl.get_first_saccade_prob, args,
                                            ret_index=True)
            ps, run_inds = out
            ps = np.array(ps)
            fs_bias_dict = dict(zip(run_inds, ps))

            args = [mdata[0], d.reading_params[m]]
            out = na.apply_function_on_runs(pl.get_time_bias, args,
                                            ret_index=True)
            ps, run_inds = out
            ps = np.array(ps)
            tb_bias_dict = dict(zip(run_inds, ps))

            args = [mdata[0], d.reading_params[m], monkey_folders[m]]
            out = na.apply_function_on_runs(pl.get_sal_bias, args,
                                            ret_index=True)
            sal, run_inds = out
            sal = np.array(sal)
            sal_bias_dict = dict(zip(run_inds, sal))

            args = [mdata[0], d.reading_params[m], monkey_folders[m]]
            out = na.apply_function_on_runs(pl.get_sal_bias, args,
                                            ret_index=True, time=True)
            sal_time, run_inds = out
            sal = np.array(sal)
            sal_time_bias_dict = dict(zip(run_inds, sal_time))
            
            bias_dicts[m] = (fs_bias_dict, tb_bias_dict, sal_bias_dict,
                             sal_time_bias_dict)
        data['ab'] = bias_dicts

    time_sal_scatter_ax = f.add_subplot(time_sal_scatter_grid, aspect='equal')
    fs_sal_scatter_ax = f.add_subplot(fs_sal_scatter_grid, aspect='equal')
    
    if 'ab' in gen_panels:
        bias_dicts = data['ab']
        for m, mdata in exper_data.items():
            b_fs, b_tb, b_sal, b_saltime = bias_dicts[m]
            b_zip = np.array(list((p, b_tb[k], b_sal[k], b_saltime[k])
                                  for k, p in b_fs.items()))
            fs_sal_scatter_ax.plot(b_zip[:, 0], b_zip[:, 2], 'o',
                                  color=monkey_colors[m])
            time_sal_scatter_ax.plot(b_zip[:, 1], b_zip[:, 3], 'o',
                                     color=monkey_colors[m])
            gpl.print_mean_conf95(b_zip[:, 2], m, 'sal FS bias', n_boots=n_boots)
            gpl.print_mean_conf95(b_zip[:, 3], m, 'sal time bias', n_boots=n_boots)
        fs_sal_scatter_ax.set_title('first saccade')
        gpl.make_xaxis_scale_bar(fs_sal_scatter_ax, magnitude=.1, anchor=.5,
                                 label='familiarity', double=False,
                                 text_buff=.15)
        gpl.make_yaxis_scale_bar(fs_sal_scatter_ax, magnitude=.1, anchor=.5,
                                 label='salience', double=True,
                                 text_buff=.38)

        time_sal_scatter_ax.set_title('looking time')
        gpl.make_xaxis_scale_bar(time_sal_scatter_ax, magnitude=.3, anchor=0,
                                 label='familiarity', double=False,
                                 text_buff=.30)
        gpl.make_yaxis_scale_bar(time_sal_scatter_ax, magnitude=.1, anchor=0,
                                 label='salience', double=True,
                                 text_buff=.25)

        
        gpl.clean_plot(time_sal_scatter_ax, 0)
        gpl.clean_plot(fs_sal_scatter_ax, 0)

    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'si-sal-py.svg')
    f.savefig(fname, bbox_inches='tight', transparent=True)
    return data

def figure_si_spatial(data=None, gen_panels=None, exper_data=None,
                      monkey_paths=pl.monkey_paths, config_file=config_path,
                      bf=None):
    setup()
    cf = u.ConfigParserColor()
    cf.read(config_file)
    params = cf['figure_spatial']
    if exper_data is None:
        exper_data = load_all_data(monkey_paths)
    if gen_panels is None:
        gen_panels = ('a', 'b', 'c', 'd')
    if data is None:
        data = {}

    # fig SI-SPATIAL arrangement
    fsize = (5, 3)
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
    combine_monkeys = params.getboolean('combine_monkeys')

    img_sacc_grid = gs[:48, :40]
    img_sacc_pt_grid = gs[:48, 40:45]
    lum_sacc_grid = gs[:48, 50:95]
    lum_sacc_pt_grid = gs[:48, 95:]

    img_sal_grid = gs[52:, :40]
    img_sal_pt_grid = gs[52:, 40:45]
    lum_sal_grid = gs[52:, 50:95]
    lum_sal_pt_grid = gs[52:, 95:]

    # fig SI-SPATIAL params
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
    dfunc_group['Rufus'] = (d.saccin_plt, d.saccout_plt,
                            d.saccin_plt_close, d.saccout_plt_close)
    dfunc_group['Neville'] = (d.saccin_plt_n, d.saccout_plt_n,
                              d.saccin_plt_close_n, d.saccout_plt_close_n)
    cond_labels = ('PLT', 'close PLT')
    color_dict = {cond_labels[0]:sdms_sc, cond_labels[1]:plt_sc}
    if 'a' not in data.keys() and 'a' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['a'] = decs
        
    check_pt = params.getfloat('check_pt')
    img_sacc_ax = f.add_subplot(img_sacc_grid)
    img_sacc_pt_ax = f.add_subplot(img_sacc_pt_grid, sharey=img_sacc_ax)
    if 'a' in gen_panels:
        decs = data['a']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, img_sacc_ax, img_sacc_pt_ax,
                                    colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'saccade decoding, {}', cond_labels)

        gpl.make_yaxis_scale_bar(img_sacc_ax, magnitude=.5, anchor=.5,
                                 label='decoding', double=False)
        gpl.clean_plot_bottom(img_sacc_ax)
        gpl.clean_plot_bottom(img_sacc_pt_ax)

    dfunc_group['Rufus'] = (d.saccin_lum, d.saccout_lum,
                            d.saccin_lum_close, d.saccout_lum_close)
    dfunc_group['Neville'] = (d.saccin_lum_n, d.saccout_lum_n,
                              d.saccin_lum_close_n, d.saccout_lum_close_n)
    cond_labels = ('lumPLT', 'close lumPLT')
    color_dict = {cond_labels[0]:sdms_sc, cond_labels[1]:plt_sc}
    if 'b' not in data.keys() and 'b' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['b'] = decs

    check_pt = params.getfloat('check_pt')
    lum_sacc_ax = f.add_subplot(lum_sacc_grid, sharey=img_sacc_ax)
    lum_sacc_pt_ax = f.add_subplot(lum_sacc_pt_grid, sharey=lum_sacc_ax)
    if 'b' in gen_panels:
        decs = data['b']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, lum_sacc_ax, lum_sacc_pt_ax,
                                    colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'lum saccade decoding, {}', cond_labels)

        gpl.clean_plot(lum_sacc_ax, 1)
        gpl.clean_plot_bottom(lum_sacc_ax)
        gpl.clean_plot_bottom(lum_sacc_pt_ax)

    dfunc_group = {}
    dfunc_group['Rufus'] = (d.novin_saccin, d.novin_saccout,
                            d.famin_saccin, d.famin_saccout,
                            d.novin_saccin_close, d.novin_saccout_close,
                            d.famin_saccin_close, d.famin_saccout_close)
    dfunc_group['Neville'] = (d.novin_saccin_n, d.novin_saccout_n,
                              d.famin_saccin_n, d.famin_saccout_n,
                              d.novin_saccin_close_n, d.novin_saccout_close_n,
                              d.famin_saccin_close_n, d.famin_saccout_close_n)

    dfunc_pairs = (0, 0, 0, 0, 1, 1, 1, 1)
    cond_labels = ('PLT', 'close PLT')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    kernel = params.get('kernel')
    zscore = params.getboolean('zscore')
    equal_fold = params.getboolean('equal_fold')
    if 'c' not in data.keys() and 'c' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               dfunc_pairs=dfunc_pairs)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['c'] = decs
        
    check_pt = params.getfloat('check_pt_img')
    img_sal_ax = f.add_subplot(img_sal_grid)
    img_sal_pt_ax = f.add_subplot(img_sal_pt_grid, sharey=img_sal_ax)
    if 'c' in gen_panels:
        decs = data['c']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, img_sal_ax, img_sal_pt_ax,
                                    colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'sal decoding, {}', cond_labels)

        gpl.make_yaxis_scale_bar(img_sal_ax, magnitude=.5, anchor=.5,
                                 label='decoding', double=False)
        gpl.make_xaxis_scale_bar(img_sal_ax, magnitude=20, label='time (ms)')
        gpl.clean_plot_bottom(img_sal_pt_ax)

    dfunc_group['Rufus'] = (d.saccin_high_lum, d.saccout_high_lum,
                            d.saccin_low_lum, d.saccout_low_lum,
                            d.saccin_high_lum_close, d.saccout_high_lum_close,
                            d.saccin_low_lum_close, d.saccout_low_lum_close)
    dfunc_group['Neville'] = (d.saccin_high_lum_n,
                              d.saccout_high_lum_n,
                              d.saccin_low_lum_n,
                              d.saccout_low_lum_n,
                              d.saccin_high_lum_close_n,
                              d.saccout_high_lum_close_n,
                              d.saccin_low_lum_close_n,
                              d.saccout_low_lum_close_n)
    cond_labels = ('lumPLT', 'close lumPLT')
    color_dict = {cond_labels[0]:sdms_mc, cond_labels[1]:plt_fc}
    kernel = params.get('kernel')
    zscore = params.getboolean('zscore')
    equal_fold = params.getboolean('equal_fold')
    if 'd' not in data.keys() and 'd' in gen_panels:
        org_data = {}
        for m, mdata in exper_data.items():
            out = pl.organize_svm_pairs_prelim(mdata[0], dfunc_group[m], mf,
                                               start, end, binsize, binstep,
                                               min_trials, zscore=zscore,
                                               cond_labels=cond_labels,
                                               min_spks=min_spks,
                                               dfunc_pairs=dfunc_pairs)
            org_data[m] = out
        if combine_monkeys:
            org_data = pl.combine_svm_format(org_data)
        decs = {}
        if zscore:
            norm = False
        else:
            norm = True
        for m, od in org_data.items():
            (dat, xs), dfunc_pairs, cond_labels = od
            dec, org = pl.organized_decoding(dat, dfunc_pairs, cond_labels,
                                             require_trials=min_trials,
                                             norm=norm, resample=resample,
                                             leave_out=leave_out,
                                             kernel=kernel,
                                             equal_fold=equal_fold)
            decs[m] = (org, dec, xs)
        data['d'] = decs
        
    check_pt = params.getfloat('check_pt_img')
    lum_sal_ax = f.add_subplot(lum_sal_grid, sharey=img_sal_ax)
    lum_sal_pt_ax = f.add_subplot(lum_sal_pt_grid, sharey=lum_sal_ax)
    if 'd' in gen_panels:
        decs = data['d']
        pl.print_decoding_info(decs)
        pts = pl.plot_decoding_info(decs, check_pt, lum_sal_ax, lum_sal_pt_ax,
                                    colors=color_dict)
        pl.print_svm_decoding_diff(pts, 'lum decoding, {}', cond_labels)

        gpl.make_xaxis_scale_bar(lum_sal_ax, magnitude=20, label='time (ms)')
        gpl.clean_plot(lum_sal_ax, 1)
        gpl.clean_plot_bottom(lum_sal_pt_ax)

    if bf is None:
        bf = params.get('basefolder')
    fname = os.path.join(bf, 'si-spatial-py.svg')
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


    # dfunc_group_plt = {}
    # dfunc_group_sdms = {}
    # dfunc_group_sdms['Rufus'] = (d.saccin_match_sdms, d.saccout_match_g_sdms,
    #                             d.saccin_nonmatch_g_sdms,
    #                             d.saccout_nonmatch_sdms)
    # dfunc_group_plt['Rufus'] = (d.novin_saccin, d.novin_saccout,
    #                             d.famin_saccin, d.famin_saccout)
    # dfunc_group_sdms['Neville'] = (d.saccin_match_sdms_n,
    #                                d.saccout_match_g_sdms_n,
    #                                d.saccin_nonmatch_g_sdms_n,
    #                                d.saccout_nonmatch_sdms_n)
    # dfunc_group_plt['Neville'] = (d.novin_saccin_n, d.novin_saccout_n,
    #                               d.famin_saccin_n, d.famin_saccout_n)

    # glm_shape = ((0, 0), (0, 1), (1, 0), (1, 1))
    # glm_labels_plt = ('fam', 'sacc')
    # glm_labels_sdms = ('match', 'sacc')
    
    # if 'cd' not in data.keys():
    #     glm_dat = {}
    #     for m, mdata in exper_data.items():
    #         out_plt = na.glm_fit_full(mdata[0], glm_shape, dfunc_group_plt[m],
    #                                   mf, start_glm, end_glm, binsize_glm,
    #                                   cond_labels=glm_labels_plt,
    #                                   binstep=binstep_glm, perms=perms_glm,
    #                                   min_trials=min_trials_glm,
    #                                   zscore=zscore_glm,
    #                                   causal_timing=causal_timing)
    #         out_sdms = na.glm_fit_full(mdata[0], glm_shape, dfunc_group_sdms[m],
    #                                    mf, start_glm, end_glm, binsize_glm,
    #                                    cond_labels=glm_labels_sdms,
    #                                    binstep=binstep_glm, perms=perms_glm,
    #                                    min_trials=min_trials_glm,
    #                                    zscore=zscore_glm,
    #                                    causal_timing=causal_timing)
    #         glm_dat[m] = (out_plt, out_sdms)
    #     data['cd'] = glm_dat

    # glm_subgroups = ((0, 1, 2, 3),
    #                  (4, 5, 6, 7))
    # labels = (('M', 'NM', 'IN', 'OUT'),
    #           ('M-IN', 'M-OUT', 'NM-IN', 'NM-OUT'))
    # if 'cd' in gen_panels:
    #     neurs = data['cd']
    #     for m, glm_dat in neurs.items():
    #         d_plt, d_sdms = glm_dat
    #         axs_gs = glm_axs[m]
    #         axs = list(f.add_subplot(ax) for ax in axs_gs)
    #         (coeffs_plt, ps_plt, _), xs = d_plt
    #         t_ind = np.argmax(np.abs(t_glm - xs))
    #         gpl.plot_glm_indiv_selectivity(coeffs_plt[:, t_ind],
    #                                        ps_plt[:, t_ind],
    #                                        group_term_labels=labels,
    #                                        subgroups=glm_subgroups,
    #                                        p_thr=p_thr, axs=axs[:2])
    #         coeffs_sdms, ps_sdms, _ = d_sdms[0]
    #         gpl.plot_glm_indiv_selectivity(coeffs_sdms[:, t_ind],
    #                                        ps_sdms[:, t_ind],
    #                                        group_term_labels=labels,
    #                                        subgroups=glm_subgroups,
    #                                        p_thr=p_thr, axs=axs[2:])

    
    # plt_r1 = gs[:30, 35:50]
    # plt_r2 = gs[:30, 50:65]

    # plt_n1 = gs[:30, 70:85]
    # plt_n2 = gs[:30, 85:]

    # sdms_r1 = gs[35:, 35:50]
    # sdms_r2 = gs[35:, 50:65]

    # sdms_n1 = gs[35:, 70:85]
    # sdms_n2 = gs[35:, 85:]

    # glm_axs = {'Neville':(plt_n1, plt_n2, sdms_n1, sdms_n2),
    #            'Rufus':(plt_r1, plt_r2, sdms_r1, sdms_r2)}
