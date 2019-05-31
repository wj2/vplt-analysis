#!/usr/bin/env python

import argparse
import numpy as np
import pickle as p
import datetime
import os

def create_parser():
    parser = argparse.ArgumentParser(description='fit Stan models to '
                                     'behavioral data')
    parser.add_argument('data_dir', type=str, help='directory of the merged '
                        'data files, with image logs')
    parser.add_argument('data_pattern', type=str, help='regex pattern for '
                        'data files')
    parser.add_argument('monkey_key', type=str, help='key for monkey identity '
                        'used to reference definitions file')
    parser.add_argument('-n', '--nov_bias_var', default=10, type=float,
                        help='variance for novelty bias prior')
    parser.add_argument('-b', '--side_bias_var', default=5, type=float,
                        help='variance for side bias prior')
    parser.add_argument('-s', '--salience_var', default=5, type=float,
                        help='variance for image salience prior')
    parser.add_argument('-m', '--model_path', type=str,
                        default=None, help='path to stan '
                        'model to fit')
    parser.add_argument('--parallel', default=True, action='store_false',
                        help='run in parallel')
    parser.add_argument('--runfolder', default='./', type=str,
                        help='path to run the script from')
    parser.add_argument('--output_pattern', default='stanfits_{}-{}.pkl',
                        type=str, help='pattern for output filename')
    parser.add_argument('--outfolder', default='../data/plt_fits/', type=str,
                        help='folder to save output in')
    parser.add_argument('--chains', type=int, default=4, help='number of '
                        'Monte Carlo chains to use')
    parser.add_argument('--length', type=int, default=2000, help='length of '
                        'each chain')
    parser.add_argument('--adapt_delta', type=float, default=.8,
                        help='adapt_delta value to use')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    os.chdir(args.runfolder)
    import pref_looking.definitions as d
    import general.utility as u
    monkey_key = args.monkey_key
    
    pattern = args.data_pattern
    datadir = args.data_dir
    max_trials = None
    forget_imglog = True

    pdict = d.reading_params[monkey_key]
    imglog_path = os.path.join(datadir, 'imglogs/')
    d_r = u.load_collection_bhvmats(datadir, pdict, pattern,
                                    make_log_name=False, max_trials=max_trials,
                                    dates=d.lip_dates,
                                    forget_imglog=forget_imglog,
                                    repl_logpath=imglog_path)
    data = d_r
    data_we = d_r[d_r['TrialError'] == 0]
    
    import pref_looking.image_selection as select
    if args.model_path is None:
        args.model_path = select.model_path_notau
    
    selection_cfs = d.rufus_bhv_model

    out = select.generate_stan_datasets(data_we, selection_cfs, pdict)
    run_dict, analysis_dict = out
    
    pev = args.nov_bias_var
    pbv = args.side_bias_var
    psv = args.salience_var

    model_path = args.model_path

    prior_dict = {'prior_eps_var':pev, 'prior_bias_var':pbv,
                  'prior_salience_var':psv}

    control_dict = {'adapt_delta':args.adapt_delta}
    stan_param_dict = {'chains':args.chains, 'iter':args.length,
                       'control':control_dict}
    
    fit_models = select.fit_run_models(run_dict, prior_dict=prior_dict, 
                                       model_path=model_path, parallel=True,
                                       stan_params=stan_param_dict)

    dt = str(datetime.datetime.now()).replace(' ', '-')
    fname = args.output_pattern.format(monkey_key, dt)
    fname = os.path.join(args.outfolder, fname)
    out_dict = {'analysis':analysis_dict, 'models':fit_models}
    p.dump(out_dict, open(fname, 'wb'))
