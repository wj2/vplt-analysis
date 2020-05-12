import general.utility as u
import numpy as np

nov_color = (.7, .4, .4)
fam_color = (.4, .4, .7)

# lip dates
lip_dates = ['09272017', '10162017', '10242017', '10302017', '10312017',
             '11012017', '11022017', '11032017', '11092017', '11302017',
             '12182017', '12192017', '12212017', '12222017', '12282017',
             '12292017', '01032018', '01042018', '01302018', '01312018',
             '02012018', '03282018', '03292018', '03302018', '04052018',
             '04062018', '05212018', '05222018', '05232018']

# Conditions
fn_plt_r = 19
ff_plt_r = 20
nn_plt_r = 21
nf_plt_r = 22
rr_plt_r = 23
rn_plt_r = 24
nr_plt_r = 25
rf_plt_r = 26
fr_plt_r = 27

lh_plt_r = 28
ll_plt_r = 29
hh_plt_r = 30
hl_plt_r = 31

plt_cond_dict = {'cond_ff':ff_plt_r, 'cond_nn':nn_plt_r, 'cond_fn':fn_plt_r, 
                 'cond_nf':nf_plt_r}
plt_rw_f_cond_dict = {'cond_ff':ff_plt_r, 'cond_nn':rr_plt_r, 'cond_fn':fr_plt_r,
                      'cond_nf':rf_plt_r}
plt_rw_n_cond_dict = {'cond_ff':nn_plt_r, 'cond_nn':rr_plt_r, 'cond_fn':nr_plt_r,
                      'cond_nf':rn_plt_r}

fn_plt_b = 7
nf_plt_b = 10
ff_plt_b = 8
nn_plt_b = 9
ii_plt_b = 11
in_plt_b = 12
ni_plt_b = 13
fi_plt_b = 14
if_plt_b = 15
bootsy_plt_conds_norw = (fn_plt_b, ff_plt_b, nn_plt_b, nf_plt_b, ii_plt_b,
                         in_plt_b, ni_plt_b, fi_plt_b, if_plt_b)
bootsy_plt_conds_norw_noi = (fn_plt_b, ff_plt_b, nn_plt_b, nf_plt_b)
bootsy_sdms_conds_norw = ()

fn_plt_s = 10
nf_plt_s = 7
ff_plt_s = 8
nn_plt_s = 9
ii_plt_s = 11
in_plt_s = 12
ni_plt_s = 13
fi_plt_s = 14
if_plt_s = 15
stan_plt_conds_norw = (fn_plt_s, ff_plt_s, nn_plt_s, nf_plt_s) # , ii_plt_s,
                       # in_plt_s, ni_plt_s, fi_plt_s, if_plt_s)
stan_sdms_conds_norw = ()

d1_xy_bootsy = (-3.5, 0)
d2_xy_bootsy = (3.5, 0)
d1_xy_stan = (9, 0)
d2_xy_stan = (-9, 0)
d1_xy_rufus = (-9, 0)
d2_xy_rufus = (9, 0)
img_wid = 3.5
img_hei = 3.5
img_wid_big = 5
img_hei_big = 5

fn_sdms1_r = 1 # fam sample, fam in rf
fn_sdms2_r = 2 # fam sample, nov in rf
ff_sdms1_r = 3
ff_sdms2_r = 4
nn_sdms1_r = 5
nn_sdms2_r = 6
nf_sdms1_r = 7 # nov sample, nov in rf
nf_sdms2_r = 8 # nov sample, fam in rf
rr_sdms1_r = 9
rr_sdms2_r = 10
rn_sdms1_r = 11
rn_sdms2_r = 12
nr_sdms1_r = 13
nr_sdms2_r = 14
rf_sdms1_r = 15
rf_sdms2_r = 16
fr_sdms1_r = 17
fr_sdms2_r = 18

rufus_plt_conds_norw = (fn_plt_r, ff_plt_r, nn_plt_r, nf_plt_r, rr_plt_r,
                        rn_plt_r, nr_plt_r, rf_plt_r, fr_plt_r,)
                        # hl_plt_r, ll_plt_r, hh_plt_r, lh_plt_r)
rufus_sdms_conds_norw = (fn_sdms1_r, fn_sdms2_r, ff_sdms1_r, ff_sdms2_r,
                         nn_sdms1_r, nn_sdms2_r, nf_sdms1_r, nf_sdms2_r,
                         rr_sdms1_r, rr_sdms2_r, rn_sdms1_r, rn_sdms2_r,
                         nr_sdms1_r, nr_sdms2_r, rf_sdms1_r, rf_sdms2_r,
                         fr_sdms1_r, fr_sdms2_r)

# Constraint functions
## behavior model
rufus_bhv_model = u.make_trial_constraint_func(('trial_type', 'TrialError',
                                                'angular_separation'),
                                               (rufus_plt_conds_norw, 0, 180), 
                                               (np.isin, np.equal, np.equal),
                                               combfunc=np.logical_and)
stan_bhv_model = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                               (stan_plt_conds_norw, 0), 
                                               (np.isin, np.equal),
                                               combfunc=np.logical_and)
bootsy_bhv_model = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                               (bootsy_plt_conds_norw_noi, 0), 
                                               (np.isin, np.equal),
                                               combfunc=np.logical_and)
rufus_plt = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                         (rufus_plt_conds_norw, 0), 
                                         (np.isin, np.equal),
                                         combfunc=np.logical_and)
rufus_sdms = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                          (rufus_sdms_conds_norw, 0), 
                                          (np.isin, np.equal),
                                          combfunc=np.logical_and)

bhv_models = {'Rufus':rufus_bhv_model, 'Stan':stan_bhv_model,
              'Bootsy':bootsy_bhv_model}



## sdmst
fam_targ_in = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                           (fn_sdms1_r, 0), 
                                           (np.equal, np.equal),
                                           combfunc=np.logical_and)
nov_targ_in = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                           (nf_sdms1_r, 0), 
                                           (np.equal, np.equal),
                                           combfunc=np.logical_and)
fam_dist_in = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                           (nf_sdms2_r, 0), 
                                           (np.equal, np.equal),
                                           combfunc=np.logical_and)
nov_dist_in = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                           (fn_sdms2_r, 0), 
                                           (np.equal, np.equal),
                                           combfunc=np.logical_and)

## plt
plt_far = u.make_trial_constraint_func(('trial_type',
                                        'angular_separation',
                                        'TrialError'),
                                       (rufus_plt_conds_norw, 180, 0),
                                       (np.isin, np.equal, np.equal))

plt_sacc_in = u.make_trial_constraint_func(('trial_type', 'left_first',
                                            'angular_separation',
                                            'TrialError'),
                                           (rufus_plt_conds_norw, True, 180, 0),
                                           (np.isin, np.equal, np.equal,
                                            np.equal))
plt_sacc_out = u.make_trial_constraint_func(('trial_type', 'right_first',
                                             'angular_separation',
                                             'TrialError'),
                                            (rufus_plt_conds_norw, True, 180, 0),
                                            (np.isin, np.equal, np.equal,
                                             np.equal))

nov_in = u.make_trial_constraint_func(('trial_type', 'left_first',
                                       'angular_separation',
                                       'TrialError'),
                                      (nf_plt_r, True, 180, 0),
                                      (np.equal, np.equal, np.equal, np.equal))
nov_out = u.make_trial_constraint_func(('trial_type', 'right_first',
                                        'angular_separation',
                                        'TrialError'),
                                       (nf_plt_r, True, 180, 0),
                                       (np.equal, np.equal, np.equal, np.equal))
nov_in_close = u.make_trial_constraint_func(('trial_type', 'left_first', 
                                             'angular_separation',
                                             'TrialError'),
                                            (nf_plt_r, True, 45, 0),
                                            (np.equal, np.equal, np.equal,
                                             np.equal))
fam_in = u.make_trial_constraint_func(('trial_type', 'left_first', 
                                       'angular_separation',
                                       'TrialError'),
                                      (fn_plt_r, True, 180, 0),
                                      (np.equal, np.equal, np.equal, np.equal))
fam_out = u.make_trial_constraint_func(('trial_type', 'right_first', 
                                        'angular_separation',
                                        'TrialError'),
                                       (fn_plt_r, True, 180, 0),
                                       (np.equal, np.equal, np.equal, np.equal))
fam_in_close = u.make_trial_constraint_func(('trial_type', 'left_first', 
                                             'angular_separation',
                                             'TrialError'),
                                            (fn_plt_r, True, 45, 0),
                                            (np.equal, np.equal, np.equal,
                                             np.equal))

hlumin_saccin = u.make_trial_constraint_func(('trial_type', 'left_first',
                                              'angular_separation',
                                              'TrialError'),
                                             ((hl_plt_r, hh_plt_r), True, 180,
                                              0),
                                             (np.isin, np.equal, np.equal,
                                              np.equal))
llumin_saccin = u.make_trial_constraint_func(('trial_type', 'left_first',
                                              'angular_separation',
                                              'TrialError'),
                                             ((lh_plt_r, ll_plt_r), True, 180,
                                              0),
                                             (np.isin, np.equal, np.equal,
                                              np.equal))


novin_saccin = u.make_trial_constraint_func(('trial_type', 'left_first',
                                             'angular_separation',
                                             'TrialError'),
                                            ((nf_plt_r, nn_plt_r), True, 180,
                                             0),
                                            (np.isin, np.equal, np.equal,
                                             np.equal))
novin_saccout = u.make_trial_constraint_func(('trial_type', 'right_first',
                                              'angular_separation',
                                              'TrialError'),
                                             ((nf_plt_r, nn_plt_r), True, 180,
                                              0),
                                             (np.isin, np.equal, np.equal,
                                              np.equal))
famin_saccin = u.make_trial_constraint_func(('trial_type', 'left_first',
                                             'angular_separation',
                                             'TrialError'),
                                            ((fn_plt_r, ff_plt_r), True, 180,
                                             0),
                                            (np.isin, np.equal, np.equal,
                                             np.equal))
famin_saccout = u.make_trial_constraint_func(('trial_type', 'right_first',
                                              'angular_separation',
                                              'TrialError'),
                                             ((fn_plt_r, ff_plt_r), True, 180,
                                              0),
                                             (np.isin, np.equal, np.equal,
                                              np.equal))
novout_saccout = u.make_trial_constraint_func(('trial_type', 'right_first',
                                               'angular_separation',
                                               'TrialError'),
                                              ((fn_plt_r, nn_plt_r), True, 180,
                                               0),
                                              (np.isin, np.equal, np.equal,
                                               np.equal))
famout_saccout = u.make_trial_constraint_func(('trial_type', 'right_first',
                                               'angular_separation',
                                               'TrialError'),
                                              ((nf_plt_r, ff_plt_r), True, 180,
                                               0),
                                              (np.isin, np.equal, np.equal,
                                               np.equal))

novin_lip = u.make_trial_constraint_func(('trial_type', 'left_first',
                                          'angular_separation',
                                          'TrialError'),
                                         ((nf_plt_r, nn_plt_r), True, 180, 0),
                                         (np.isin, np.equal, np.equal, np.equal))
famin_lip = u.make_trial_constraint_func(('trial_type', 'left_first',
                                          'angular_separation',
                                          'TrialError'),
                                         ((fn_plt_r, ff_plt_r), True, 180, 0),
                                         (np.isin, np.equal, np.equal, np.equal))
novout_lip = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_r, nn_plt_r), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
famout_lip = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((nf_plt_r, ff_plt_r), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))

saccin_lip = u.make_trial_constraint_func(('trial_type', 'left_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_r, nn_plt_r, ff_plt_r,
                                        nf_plt_r),
                                       True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
saccout_lip = u.make_trial_constraint_func(('trial_type', 'right_first',
                                        'angular_separation',
                                        'TrialError'),
                                       ((fn_plt_r, nn_plt_r, ff_plt_r,
                                         nf_plt_r),
                                        True, 180, 0),
                                       (np.isin, np.equal, np.equal, np.equal))

novin_lip_sdms = u.make_trial_constraint_func(('trial_type',
                                      'angular_separation',
                                      'TrialError'),
                                     ((nf_sdms1_r, nn_sdms1_r), 180, 0),
                                     (np.isin, np.equal, np.equal))
famin_lip_sdms = u.make_trial_constraint_func(('trial_type', 
                                      'angular_separation',
                                      'TrialError'),
                                     ((fn_sdms1_r, ff_sdms1_r), 180, 0),
                                     (np.isin, np.equal, np.equal))
novout_lip_sdms = u.make_trial_constraint_func(('trial_type',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_sdms2_r, nn_sdms2_r), 180, 0),
                                      (np.isin, np.equal, np.equal))
famout_lip_sdms = u.make_trial_constraint_func(('trial_type', 
                                       'angular_separation',
                                       'TrialError'),
                                      ((nf_sdms2_r, ff_sdms2_r), 180, 0),
                                      (np.isin, np.equal, np.equal))

saccin_lip_sdms = u.make_trial_constraint_func(('trial_type',
                                       'TrialError'),
                                      ((fn_sdms1_r, nn_sdms1_r, ff_sdms1_r,
                                        nf_sdms1_r), 0),
                                      (np.isin, np.equal))
saccout_lip_sdms = u.make_trial_constraint_func(('trial_type', 'TrialError'),
                                                ((fn_sdms2_r, nn_sdms2_r,
                                                  ff_sdms2_r, nf_sdms2_r), 0),
                                       (np.isin, np.equal))


novin_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'left_first',
                                      'angular_separation',
                                      'TrialError'),
                                     ((nf_plt_b, nn_plt_b), True, 180, 0),
                                     (np.isin, np.equal, np.equal, np.equal))
famin_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'left_first',
                                      'angular_separation',
                                      'TrialError'),
                                     ((fn_plt_b, ff_plt_b), True, 180, 0),
                                     (np.isin, np.equal, np.equal, np.equal))
novout_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_b, nn_plt_b), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
famout_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((nf_plt_b, ff_plt_b), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))

saccin_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'left_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_b, nn_plt_b, ff_plt_b,
                                        nf_plt_b),
                                       True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
saccout_itc_bootsy = u.make_trial_constraint_func(('trial_type', 'right_first',
                                        'angular_separation',
                                        'TrialError'),
                                       ((fn_plt_b, nn_plt_b, ff_plt_b,
                                         nf_plt_b),
                                        True, 180, 0),
                                       (np.isin, np.equal, np.equal, np.equal))


novin_itc_stan = u.make_trial_constraint_func(('trial_type', 'left_first',
                                      'angular_separation',
                                      'TrialError'),
                                     ((nf_plt_s, nn_plt_s), True, 180, 0),
                                     (np.isin, np.equal, np.equal, np.equal))
famin_itc_stan = u.make_trial_constraint_func(('trial_type', 'left_first',
                                      'angular_separation',
                                      'TrialError'),
                                     ((fn_plt_s, ff_plt_s), True, 180, 0),
                                     (np.isin, np.equal, np.equal, np.equal))
novout_itc_stan = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_s, nn_plt_s), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
famout_itc_stan = u.make_trial_constraint_func(('trial_type', 'right_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((nf_plt_s, ff_plt_s), True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))

saccin_itc_stan = u.make_trial_constraint_func(('trial_type', 'left_first',
                                       'angular_separation',
                                       'TrialError'),
                                      ((fn_plt_s, nn_plt_s, ff_plt_s,
                                        nf_plt_s),
                                       True, 180, 0),
                                      (np.isin, np.equal, np.equal, np.equal))
saccout_itc_stan = u.make_trial_constraint_func(('trial_type', 'right_first',
                                        'angular_separation',
                                        'TrialError'),
                                       ((fn_plt_s, nn_plt_s, ff_plt_s,
                                         nf_plt_s),
                                        True, 180, 0),
                                       (np.isin, np.equal, np.equal, np.equal))


# Timing functions
fix_off_func = u.make_time_field_func('fixation_off')
first_sacc_func = u.make_time_field_func('first_sacc_time')


# Parameter collections
eye_params = {}
eye_params['Rufus'] = {'skips':1, 'stdthr':None, 'filtwin':40, 'thr':.07,
                       'fixthr':10, 'vthr':None}
eye_params['Bootsy'] = {'skips':1, 'stdthr':None, 'filtwin':40, 'thr':.07,
                        'fixthr':10, 'vthr':None}
eye_params['Stan'] = {'skips':1, 'stdthr':None, 'filtwin':40, 'thr':.07,
                      'fixthr':10, 'vthr':None}

reading_params = {}
reading_params['Stan'] = {'plt_conds': stan_plt_conds_norw,
                          'sdms_conds': stan_sdms_conds_norw,
                          'noerr': False, 'ephys': True,
                          'default_img1_xy': d1_xy_stan, 
                          'default_img2_xy':d2_xy_stan,
                          'default_wid':img_wid_big, 
                          'default_hei':img_hei_big,
                          'eye_params':eye_params['Stan']}
reading_params['Bootsy'] = {'plt_conds': bootsy_plt_conds_norw,
                            'sdms_conds': bootsy_sdms_conds_norw,
                            'noerr': False, 'ephys': True,
                            'default_img1_xy': d1_xy_bootsy, 
                            'default_img2_xy':d2_xy_bootsy,
                            'default_wid':img_wid, 
                            'default_hei':img_hei,
                            'eye_params':eye_params['Bootsy']}
reading_params['Rufus'] = {'plt_conds': rufus_plt_conds_norw,
                           'sdms_conds': rufus_sdms_conds_norw,
                           'noerr': False, 'ephys': True,
                           'default_img1_xy': d1_xy_rufus, 
                           'default_img2_xy':d2_xy_rufus,
                           'eye_params':eye_params['Rufus']}
