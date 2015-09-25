
import numpy as np
import os
import re
import shutil
import scipy.io as sio

monthdict = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', 
             '06':'Jun', '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct',
             '11':'Nov', '12':'Dec'}

def load_data(path, varname='superx'):
    data = sio.loadmat(path, mat_dtype=True)
    return data[varname]

def load_separate(paths, pattern=None, varname='x'):
    files = []
    for p in paths:
        if os.path.isdir(p):
            f = os.listdir(p)[1:]
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
