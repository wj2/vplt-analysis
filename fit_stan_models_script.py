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
    parser.add_argument('-p', '--nov_bias_mean_mean', default=0, type=float,
                        help='mean for mean novelty bias prior')
    parser.add_argument('-n', '--nov_bias_mean_var', default=10, type=float,
                        help='variance for mean novelty bias prior')
    parser.add_argument('--nov_bias_var_mean', default=5, type=float,
                        help='mean for novelty bias variance prior')
    parser.add_argument('--nov_bias_var_var', default=10, type=float,
                        help='variance for novelty bias variance prior')
    parser.add_argument('--side_bias_var_mean', default=0, type=float,
                        help='mean for side bias variance prior')
    parser.add_argument('-b', '--side_bias_var_var', default=15, type=float,
                        help='variance for side bias variance prior')
    parser.add_argument('--side_bias_mean_mean', default=5, type=float,
                        help='mean for side bias mean prior')
    parser.add_argument('--side_bias_mean_var', default=15, type=float,
                        help='variance for side bias mean prior')
    parser.add_argument('-s', '--salience_var_var', default=15, type=float,
                        help='variance of variance for image salience prior')
    parser.add_argument('--salience_var_mean', default=0, type=float,
                        help='mean of variance for image salience prior')
    parser.add_argument('--salience_mean_var', default=15, type=float,
                        help='variance of mean for image salience prior')
    parser.add_argument('--salience_mean_mean', default=0, type=float,
                        help='mean of mean for image salience prior')
    parser.add_argument('--bias_var_var', default=15, type=float,
                        help='variance of variance for looking bias prior')
    parser.add_argument('--bias_var_mean', default=0, type=float,
                        help='mean of variance for looking bias prior')
    parser.add_argument('-m', '--model_path', type=str,
                        default=None, help='path to stan '
                        'model to fit')
    parser.add_argument('--outcome', default='saccade', help='outcome type to '
                        'use (default is first saccade)')
    parser.add_argument('--start_count', default=100, type=int,
                        help='start counting image fixation for bias outcome')
    parser.add_argument('--count_len', type=int, default=400,
                        help='how long to count for for bias outcome')
    parser.add_argument('--not_parallel', default=False, action='store_true',
                        help='do not run in parallel (done by default)')
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
    parser.add_argument('--max_treedepth', type=int, default=10,
                        help='maximum tree depth to use')
    parser.add_argument('--full_data', default=False, action='store_true',
                        help='collapse data across days and fit it all at '
                        'once -- does not do this by default')
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
    import general.stan_utility as su
    if args.model_path is None and args.outcome == 'saccade':
        args.model_path = select.model_path_notau
    elif args.model_path is None and args.outcome == 'bias':
        args.model_path = select.model_path_timebias

    if args.outcome == 'saccade':
        mult = 1
    elif args.outcome == 'bias':
        mult = 0
    
    selection_cfs = d.rufus_bhv_model

    collapse = args.full_data
    out = select.generate_stan_datasets(data_we, selection_cfs, pdict,
                                        collapse=collapse,
                                        outcome_type=args.outcome,
                                        start_count=args.start_count,
                                        count_len=args.count_len)
    run_dict, analysis_dict = out

    pbvv = args.side_bias_var_var
    pbvm = args.side_bias_var_mean
    pbmv = args.side_bias_mean_var
    pbmm = args.side_bias_mean_mean*mult

    pevv = args.nov_bias_var_var
    pevm = args.nov_bias_var_mean
    pemv = args.nov_bias_mean_var
    pemm = args.nov_bias_mean_mean*mult
    
    psmv = args.salience_mean_var
    psmm = args.salience_mean_mean
    psvv = args.salience_var_var
    psvm = args.salience_var_mean*mult

    plvm = args.bias_var_mean
    plvv = args.bias_var_var

    model_path = args.model_path

    prior_dict = {'prior_bias_var_var':pbvv, 'prior_bias_var_mean':pbvm,
                  'prior_bias_mean_var':pbmv, 'prior_bias_mean_mean':pbmm,
                  'prior_salience_var_var':psvv, 'prior_salience_var_mean':psvm,
                  'prior_eps_mean_mean':pemm, 'prior_eps_mean_var':pemv, 
                  'prior_eps_var_mean':pevm, 'prior_eps_var_var':pevv,
                  'prior_look_var_mean':plvm, 'prior_look_var_var':plvv,
                  'prior_salience_mean_mean':psmm,
                  'prior_salience_mean_var':psmv}

    control_dict = {'adapt_delta':args.adapt_delta,
                    'max_treedepth':args.max_treedepth}
    stan_param_dict = {'chains':args.chains, 'iter':args.length,
                       'control':control_dict}
    parallel = not args.not_parallel
    out = select.fit_run_models(run_dict, prior_dict=prior_dict, 
                                model_path=model_path, parallel=parallel,
                                stan_params=stan_param_dict)
    model, fit_models = out
    fit_models = su.store_models(fit_models)
    dt = str(datetime.datetime.now()).replace(' ', '-')
    fname = args.output_pattern.format(monkey_key, dt)
    fname = os.path.join(args.outfolder, fname)
    out_dict = {'analysis':analysis_dict, 'models':fit_models, 'info':run_dict}
    p.dump(out_dict, open(fname, 'wb'))
