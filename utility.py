
import numpy as np
import os
import re
import shutil
import scipy.io as sio
import general.utility as u

monthdict = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', 
             '06':'Jun', '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct',
             '11':'Nov', '12':'Dec'}

def load_data(path, varname='superx'):
    data = sio.loadmat(path, mat_dtype=True)
    return data[varname]

def load_separate(paths, pattern=None, varname='x'):
    files = []
    pattern = '.*' + pattern
    for p in paths:
        if os.path.isdir(p):
            f = os.listdir(p)[1:]
            f = [os.path.join(p, x) for x in f]
            files = files + f
        else:
            files.append(p)
    if pattern is not None:
        files = filter(lambda x: re.match(pattern, x) is not None, files)
    for i, f in enumerate(files):
        d = load_data(f, varname=varname)
        if i == 0:
            alldata = d
        else:
            alldata = np.concatenate((alldata, d), axis=0)
    return alldata    


# def load_bhvmat_imglog(path_bhv, path_log=None, noerr=True, prevlog_dict=None,
#                        trial_field='TrialNumber', fixation_on=35, 
#                        fixation_off=36, start_trial=9, datanum=0, 
#                        lever_release=4, lever_hold=7, reward_given=48,
#                        end_trial=18, fixation_acq=8, left_img_on=191,
#                        left_img_off=192, right_img_on=195, right_img_off=196,
#                        default_wid=6, default_hei=6, default_img1_xy=(-5, 0),
#                        default_img2_xy=(5, 0), 
#                        plt_conds=(7,8,9,10,11,12,13,14,15,16)):
#     if prevlog_dict is None:
#         log_dict = {}
#     else:
#         log_dict = prevlog_dict
#     bhv = sio.loadmat(path_bhv)['bhv']
#     if path_log is not None:
#         log = open(path_log, 'rb').readlines()
#     else:
#         log = None
#     trls = bhv[trial_field][0,0]
#     fields = ['trial_type', 'TrialError', 'block_number', 'code_numbers',
#               'code_times', 'trial_starts', 'datafile', 'datanum',
#               'centimgon','centimgoff','trialnum','image_nos','leftimg',
#               'rightimg', 'leftviews', 'rightviews','fixation_on','fixation_off',
#               'lever_release','lever_hold','reward_time','ISI_start',
#               'ISI_end','fixation_acquired','left_img_on','left_img_off',
#               'right_img_on','right_img_off','eyepos','spike_times','LFP',
#               'task_cond_num', 'img1_xy', 'img2_xy', 'img_wid', 'img_hei']
#     dt = {'names':fields, 'formats':['O']*len(fields)}
#     x = np.zeros(len(trls), dtype=dt)
#     for i, t in enumerate(trls):
#         x[i]['trial_type'] = bhv['ConditionNumber'][0,0][i, 0]
#         x[i]['task_cond_num'] = bhv['ConditionNumber'][0,0][i, 0]
#         x[i]['TrialError'] = bhv['TrialError'][0,0][i, 0]
#         x[i]['block_number'] = bhv['BlockNumber'][0,0][i, 0]
#         x[i]['code_numbers'] = bhv['CodeNumbers'][0,0][0, i]
#         x[i]['code_times'] = bhv['CodeTimes'][0,0][0, i]
#         x[i]['trial_starts'] = get_bhvcode_time(start_trial, x[i]['code_numbers'],
#                                                 x[i]['code_times'], first=True)
#         x[i]['ISI_end'] = get_bhvcode_time(start_trial, x[i]['code_numbers'],
#                                            x[i]['code_times'], first=True)
#         x[i]['ISI_start'] = get_bhvcode_time(end_trial, x[i]['code_numbers'],
#                                              x[i]['code_times'], first=True)
#         x[i]['datafile'] = bhv['DataFileName']
#         x[i]['datanum'] = datanum
#         x[i]['centimgon'] = []
#         x[i]['centimgoff'] = []
#         x[i]['trialnum'] = bhv['TrialNumber'][0,0][i, 0]
#         x[i]['image_nos'] = []
#         if x[i]['trial_type'] in plt_conds and log is not None:
#             entry1 = log.pop(0)
#             entry1 = entry1.strip('\r\n').split('\t')
#             tn1, s1, _, cond1, vs1, cat1, img1 = entry1
#             entry2 = log.pop(0)
#             entry2 = entry2.strip('\r\n').split('\t')
#             tn2, s2, _, cond2, vs2, cat2, img2 = entry2
#             assert tn1 == tn2
#             assert int(tn1) == int(x[i]['trialnum'])
#             img1_n, ext1 = os.path.splitext(img1)
#             img2_n, ext2 = os.path.splitext(img2)
#             log_dict[img1_n] = log_dict.get(img1_n, 0) + 1
#             log_dict[img2_n] = log_dict.get(img2_n, 0) + 1
#             x[i]['leftimg'] = img1_n
#             x[i]['rightimg'] = img2_n
#             x[i]['leftviews'] = log_dict[img1_n]
#             x[i]['rightviews'] = log_dict[img2_n]
#         else:
#             x[i]['leftimg'] = ''
#             x[i]['rightimg'] = ''
#             x[i]['leftviews'] = 0
#             x[i]['rightviews'] = 0
#         x[i]['fixation_on'] = get_bhvcode_time(fixation_on, 
#                                                x[i]['code_numbers'],
#                                                x[i]['code_times'], first=True)
#         x[i]['fixation_off'] = get_bhvcode_time(fixation_off, 
#                                                 x[i]['code_numbers'],
#                                                 x[i]['code_times'], first=True)
#         x[i]['lever_release'] = get_bhvcode_time(lever_release,
#                                                  x[i]['code_numbers'],
#                                                  x[i]['code_times'], first=True)
#         x[i]['lever_hold'] = get_bhvcode_time(lever_hold,
#                                               x[i]['code_numbers'],
#                                               x[i]['code_times'], first=True)
#         x[i]['reward_time'] = get_bhvcode_time(reward_given,
#                                                x[i]['code_numbers'],
#                                                x[i]['code_times'], first=True)
#         x[i]['fixation_acquired'] = get_bhvcode_time(fixation_acq,
#                                                      x[i]['code_numbers'],
#                                                      x[i]['code_times'], 
#                                                      first=True)
#         x[i]['left_img_on'] = get_bhvcode_time(left_img_on, 
#                                                x[i]['code_numbers'],
#                                                x[i]['code_times'], first=True)
#         x[i]['left_img_off'] = get_bhvcode_time(left_img_off,
#                                                 x[i]['code_numbers'],
#                                                 x[i]['code_times'], first=True)
#         x[i]['right_img_on'] = get_bhvcode_time(right_img_on, 
#                                                 x[i]['code_numbers'],
#                                                 x[i]['code_times'], first=True)
#         x[i]['right_img_off'] = get_bhvcode_time(right_img_off,
#                                                  x[i]['code_numbers'],
#                                                  x[i]['code_times'], first=True)
#         x[i]['eyepos'] = bhv['AnalogData'][0,0][0, i]['EyeSignal'][0,0]
#         if ('UserVars' in bhv.dtype.names 
#             and 'img1_xy' in bhv['UserVars'][0,0].dtype.names
#             and bhv['UserVars'][0,0]['img1_xy'].shape[1] > i):
#             x[i]['img1_xy'] = bhv['UserVars'][0,0]['img1_xy'][0, i]
#             x[i]['img2_xy'] = bhv['UserVars'][0,0]['img2_xy'][0, i]
#         else:
#             x[i]['img1_xy'] = default_img1_xy
#             x[i]['img2_xy'] = default_img2_xy
#         if ('UserVars' in bhv.dtype.names 
#             and 'img_wid' in bhv['UserVars'][0,0].dtype.names
#             and bhv['UserVars'][0,0]['img_wid'].shape[1] > i):
#             x[i]['img_wid'] = bhv['UserVars'][0,0]['img_wid'][0, i]
#             x[i]['img_hei'] = bhv['UserVars'][0,0]['img_hei'][0, i]
#         else:
#             x[i]['img_wid'] = default_wid
#             x[i]['img_hei'] = default_hei
#     if noerr:
#         x = x[x['TrialError'] == 0]
#     return x, log_dict

def get_bhvcode_time(codenum, trial_codenums, trial_codetimes, first=True):
    i = np.where(codenum == trial_codenums)[0]
    if len(i) > 0:
        if first:
            i = i[0]
        else:
            i = i[-1]
        ret = trial_codetimes[i][0]
    else:
        ret = None
    return ret

def get_only_vplt(data, condrange=(7, 20), condfield='trial_type'):
    mask =np.logical_and(data[condfield] >= condrange[0], 
                         data[condfield] <= condrange[1])
    return data[mask]

def distribute_imglogs(il_path, out_path):
    il_list = os.listdir(il_path)
    nomatch = []
    for il in il_list:
        m = re.findall('-([0-9][0-9])(?=-)', il)
        if len(m) == 2:
            mo = monthdict[m[0]]
            da = m[1]
            if da[0] == '0':
                da = da[1]
            foldname = mo+da
            fpath = os.path.join(out_path, foldname)
            if foldname not in os.listdir(out_path):
                os.mkdir(fpath)
            shutil.copy(os.path.join(il_path, il),
                        os.path.join(out_path, foldname))
        else:
            nomatch.append(il)
    return nomatch

def get_data_run_nums(data, drunfield):
    return np.unique(np.concatenate(data[drunfield], axis=0))

def bootstrap_list(l, func, n=1000):
    stats = np.ones(n)
    for i in xrange(n):
        samp = np.random.choice(l, len(l))
        stats[i] = func(samp)
    return stats

def collapse_list_dict(ld):
    for i, k in enumerate(ld.keys()):
        if i == 0:
            l = ld[k]
        else:
            l = np.concatenate((l, ld[k]))
    return l

def gen_img_list(famfolder=None, fam_n=None, novfolder=None, nov_n=None, 
                 intfamfolder=None, intfam_n=None):
    if famfolder is not None:
        fams = os.listdir(famfolder)[1:]
    else:
        fams = ['F {}'.format(i+1) for i in np.arange(fam_n)]
    if novfolder is not None:
        novs = os.listdir(novfolder)[1:]
    else:
        novs = ['N {}'.format(i+1) for i in np.arange(nov_n)]
    if intfamfolder is not None:
        intfams = os.listdir(intfamfolder)[1:]
    else:
        intfams = ['IF {}'.format(i+1) for i in np.arange(intfam_n)]
    return fams, novs, intfams
        

def get_img_names(codes, famfolder='/Users/wjj/Dropbox/research/uc/freedman/'
                  'pref_looking/famimgs', if_ns=25, n_ns=50):
    f, n, i = gen_img_list(famfolder=famfolder, nov_n=n_ns, intfam_n=if_ns)
    all_imnames = np.array(f + n + i)
    cs = (codes - 1).astype(np.int)
    return all_imnames[cs]

def get_cent_codes(tcodes, imgcodebeg=56, imgcodeend=180):
    return tcodes[(tcodes >= imgcodebeg)*(tcodes <= imgcodeend)]

def get_code_member(code, fambeg=56, famend=105, intbeg=106, intend=130, 
                    novbeg=131, novend=180):
    if novbeg <= code <= novend:
        member = np.array([[1, 0, 0]])
    elif intbeg <= code <= intend:
        member = np.array([[0, 1, 0]])
    elif fambeg <= code <= famend:
        member = np.array([[0, 0, 1]])
    return member

def get_code_views(code, fambeg=56, famend=105, intbeg=106, intend=130, 
                   novbeg=131, novend=180, novviews=25., intviews=450., 
                   famviews=10000.):
    memb = get_code_member(code, novbeg=novbeg, novend=novend, intbeg=intbeg, 
                           intend=intend, fambeg=fambeg, famend=famend)
    return np.sum(np.array([novviews, intviews, famviews])*memb)

def views_to_member(views, novbeg=0, novend=200, intbeg=201, intend=1000, 
                    fambeg=1001, famend=100000):
    return get_code_member(views, novbeg=novbeg, novend=novend, intbeg=intbeg,
                           intend=intend, fambeg=fambeg, famend=famend)
    
def bins_tr(spks, beg, end, binsize, column=False, usetype=np.float):
    bs = np.arange(beg, end + binsize, binsize)
    spk_bs, _ = np.histogram(spks, bins=bs)
    if column:
        spk_bs = np.reshape(spk_bs, (-1, 1))
    return spk_bs.astype(usetype)

def get_neuron_spks_list(data, zerotimecode=8, drunfield='datanum', 
                         spktimes='spike_times'):
    undruns = get_data_run_nums(data, drunfield)
    neurons = []
    for i, dr in enumerate(undruns):
        d = data[data[drunfield] == dr]
        drneurs = []
        for j, tr in enumerate(d):
            t = get_code_time(tr, zerotimecode)
            for k, spks in enumerate(tr[spktimes][0, :]):
                if len(spks) > 0:
                    tspks = spks - t
                    if j == 0:
                        drneurs.append([tspks])
                    else:
                        drneurs[k].append(tspks)
                else:
                    if j == 0:
                        drneurs.append([])
        neurons = neurons + drneurs
    return neurons

def euler_integrate(func, beg, end, step):
    accumulate = 0
    for t in np.arange(beg, end, step):
        accumulate = accumulate + step*func(t)
    return accumulate

def evoked_st_cumdist(spkts, t, lam):
    out = 1 - (1 - empirical_fs_cumdist(spkts, t))*np.exp(lam*t)
    return out

def empirical_fs_cumdist(spkts, t):
    spk_before = lambda x: np.any(x < t) or (x.size == 0)
    successes = np.sum(map(spk_before, spkts))
    x = map(spk_before, spkts)
    print(successes / float(len(spkts)))
    return successes / float(len(spkts))

def get_spks_window(dat, begwin, endwin):
    makecut = lambda x: (np.sum(np.logical_and(begwin <= x, x <= endwin)) 
                         / (endwin - begwin))
    stuff = map(makecut, dat)
    return stuff

def get_code_time(trl, code, codenumfield='code_numbers', 
                  codetimefield='code_times'):
    ct = trl[codetimefield][trl[codenumfield] == code]
    return ct

def estimate_latency(neurspks, backgrd_window, latenwindow, integstep=.5):
    bckgrd_spks = get_spks_window(neurspks, backgrd_window[0], 
                                  backgrd_window[1])
    bgd_est = np.mean(bckgrd_spks)
    expect_func = lambda x: 1 - evoked_st_cumdist(neurspks, x, bgd_est)
    est_lat = euler_integrate(expect_func, latenwindow[0], latenwindow[1], 
                              integstep)
    sm_func = lambda x: 2*x*(1 - evoked_st_cumdist(neurspks, x, bgd_est))
    sm_latency = euler_integrate(sm_func, latenwindow[0], latenwindow[1], 
                                 integstep)
    est_std = np.sqrt(sm_latency - est_lat**2)
    return est_lat, est_std

def get_trls_with_neurnum(dat, neurnum, neurfield='spike_times', 
                          drunfield='datanum'):
    druns = get_data_run_nums(dat, drunfield)
    count_neurs = 0
    for i, dr in enumerate(druns):
        drdat = dat[dat[drunfield] == dr]
        new_neurs = drdat[0][neurfield].shape[1]
        if count_neurs <= neurnum and new_neurs + count_neurs > neurnum:
            neur_i = neurnum - count_neurs
            for trls in drdat:
                trls[neurfield] = trls[neurfield][:, [neur_i]]
            keeptrls = drdat
            return keeptrls
        else:
            count_neurs = count_neurs + new_neurs
    raise Exception('only {} neurons in this dataset, which is less '
                    'than {}'.format(count_neurs, neurnum))
            
def get_condnum_trls(dat, condnums, condnumfield='trial_type'):
    keeps = np.zeros((dat.shape[0], len(condnums)))
    for i, c in enumerate(condnums):
        keeps[:, i] = dat[condnumfield] == c
    flatkeep = np.sum(keeps, 1)
    return dat[flatkeep > 0]
