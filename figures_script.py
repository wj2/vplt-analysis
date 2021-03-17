
import argparse
import pickle
import os
import multiprocessing
multiprocessing.set_start_method("fork")

import pref_looking.plt_analysis as pl
import general.plotting_styles as gps
import pref_looking.figures_sp as fsp

fig_data_default = 'pref_looking/context_analysis_save-combined.pkl'
monkey_paths_default = ('Neville', 'pref_looking/data-neville/',
                        'Rufus', 'pref_looking/data-rufus/')
default_config = 'pref_looking/figures_sp.conf'

def create_parser():
    parser = argparse.ArgumentParser(description='generate figures for pref '
                                     'looking paper')
    parser.add_argument('--monkey_data', type=str, default=monkey_paths_default,
                        nargs='+', help='monkey name and path pairs')
    parser.add_argument('--use_fig_data', default=False,
                        action='store_true', help='use pre-computed figure '
                        'data if available')
    parser.add_argument('--fig_data', type=str, default=fig_data_default,
                        help='pre-computed figure data pkl')
    parser.add_argument('-o', '--output_folder', default='pl_results', type=str,
                        help='folder to save the output in')
    parser.add_argument('--generate_figures', default='', type=str,
                        nargs='+', help='figures to generate')
    parser.add_argument('--config_path', default=default_config, type=str,
                        help='path to config file')
    return parser

def make_monkey_paths(mps):
    num_monkeys = int(len(mps)/2)
    mdict = {}
    for i in range(num_monkeys):
        mdict[mps[2*i]] = mps[2*i + 1]
    return mdict

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    gps.set_paper_style()

    if len(args.monkey_data) % 2 > 0:
        raise IOError('monkey data needs to be even in length')

    monkey_paths = make_monkey_paths(args.monkey_data)
    data = pl.load_all_data(monkey_paths)

    if args.use_fig_data and os.path.isfile(args.fig_data):
        fig_data = pickle.load(open(args.fig_data, 'rb'))
    else:
        fig_data = {}

    fig_funcs = {'1':fsp.figure1, '2':fsp.figure2, '2a':fsp.figure2a,
                 '3':fsp.figure3, '4':fsp.figure4, '5':fsp.figure5,
                 '6':fsp.figure6, 'si-spatial':fsp.figure_si_spatial,
                 'si-bhv':fsp.figure_si_bhv, 'si-sal':fsp.figure_si_sal}

    if args.generate_figures == '':
        gen_figs = fig_funcs.keys()
    else:
        gen_figs = args.generate_figures

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
    for gf in gen_figs:
        fig_data[gf] = fig_funcs[gf](exper_data=data, data=fig_data.get(gf),
                                     config_file=args.config_path,
                                     bf=args.output_folder)

    out_datafile = os.path.join(args.output_folder, 'fig_data.pkl')
    pickle.dump(fig_data, open(out_datafile, 'wb'))
