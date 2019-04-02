
import numpy as np
import itertools as it
import scipy.special as ss
import scipy.stats as sts
import scipy.integrate as sin
import scipy.optimize as sio

import general.utility as u
import general.rf_models as rfm
import mixedselectivity_theory.nms_continuous as nmc

def simulate_estimates(p, s, pop_ests):
    pts = np.random.uniform(0, p, (s, 1))
    est_ns = np.random.normal(0, np.sqrt(pop_ests), (s, len(pop_ests)))
    ests = pts + est_ns
    return ests

def calculate_distance(ests):
    cs = it.combinations(range(ests.shape[1]), 2)
    dist = 0
    for c in cs:
        sub_est = np.abs(ests[:, c[0]] - ests[:, c[1]])
        dist = dist + np.sum(sub_est)
    return dist

def assign_estimates_2pop(ests):
    s, pops = ests.shape
    assert pops == 2
    baseline = calculate_distance(ests)
    perturbs = list(it.combinations(range(s), 2))
    assign_dists = np.zeros(len(perturbs))
    for i, p in enumerate(perturbs):
        ests[p[1], 0], ests[p[0], 0] = ests[p[0], 0], ests[p[1], 0]
        assign_dists[i] = calculate_distance(ests)
        ests[p[1], 0], ests[p[0], 0] = ests[p[0], 0], ests[p[1], 0]
    return assign_dists, baseline

def estimate_assignment_error(p, s, pop_ests, n=500):
    err = np.zeros(n)
    for i in range(n):
        est = simulate_estimates(p, s, pop_ests)
        ad, base = assign_estimates_2pop(est)
        err[i] = np.any(ad < base)
    return err

def estimate_ae_sr_range(s, srs, n_pops=2, n=500, p=100, boot=True):
    errs = np.zeros((n, len(srs)))
    for i, sr in enumerate(srs):
        pop_est = (p/sr)**2
        pop_ests = (pop_est,)*n_pops
        errs[:, i] = estimate_assignment_error(p, s, pop_ests, n=n)
    if boot:
        errs = u.bootstrap_on_axis(errs, u.mean_axis0, axis=0, n=n)
    return errs

def estimate_ae_sr_s_ranges(esses, srs, n_pops=2, n=500, p=100, boot=True):
    errs = np.zeros((len(esses), n, len(srs)))
    for i, s in enumerate(esses):
        errs[i] = estimate_ae_sr_range(s, srs, n_pops=n_pops, n=n, p=p,
                                       boot=boot)
    return errs

def error_approx_sr_s_ranges(esses, srs, n_pops=2):
    errs = np.zeros((len(esses), len(srs)))
    for i, s in enumerate(esses):
        errs[i] = error_approx_sr_range(s, srs, n_pops=n_pops)
    return errs

def error_approx_sr_range(s, srs, n_pops=2, p=100, pop_est=2):
    errs = np.zeros(len(srs))
    for i, sr in enumerate(srs):
        pop_est = (p/sr)**2
        pop_ests = (pop_est,)*n_pops
        errs[i] = error_approx(p, s, pop_ests)
    return errs

def error_approx(p, s, pop_ests, integ_step=.0001, eps=0):
    integ_step = integ_step*p
    eps = eps*p
    factor = ss.comb(s, 2)
    var = np.sum(pop_ests)
    int_func = lambda x: 2*((p - x)/(p**2))*sts.norm(x, np.sqrt(var)).cdf(0)
    pe = u.euler_integrate(int_func, eps, p, integ_step)
    return factor*pe

def error_approx_further(esses, srs, p=100):
    errs = np.zeros((len(esses), len(srs)))
    pop_ests = (p/srs)**2
    for i, s in enumerate(esses):
        first_term = 2*np.sqrt(pop_ests)/(p*np.sqrt(np.pi))
        second_term = -2*pop_ests/(2*(p**2))
        errs[i] = ss.comb(s, 2)*(first_term + second_term)
    errs[errs > 1] = 1
    return errs

def distortion_error_approx(esses, d1, d2, p=100):
    errs = np.zeros((len(esses), len(d1)))
    first_term = np.sqrt(2*(d1 + d2))/(p*np.sqrt(np.pi))
    second_term = -(d1 + d2)/(2*(p**2))
    for i, s in enumerate(esses):
        errs[i] = ss.comb(s, 2)*(first_term + second_term)
    errs[errs > 1] = 1
    return errs

def feature_info_consumption(esses, distortion, d=1, p=100):
    info = np.log(p/np.sqrt(2*np.pi*distortion))
    infos = np.ones((len(esses), len(distortion)))
    for i, s in enumerate(esses):
        infos[i] = d*s*info
    return info

def feature_redundant_info(esses, d1, d2, overlapping_d=1, p=100):
    redundancy = np.log(p/np.sqrt(2*np.pi*(d1 + d2)))
    redunds = np.ones((len(esses), len(d1)))
    for i, s in enumerate(esses):
        redunds[i] = overlapping_d*s*redundancy
    return redunds

def feature_wasted_info(esses, d1, d2, distortion, overlapping_d=1, p=100):
    waste = np.log(p*np.sqrt(d1 - distortion)/(np.sqrt(2*np.pi)*d1))
    wastes = np.zeros((len(esses), len(d1)))
    for i, s in enumerate(esses):
        wastes[i] = overlapping_d*s*waste
    return wastes
                   

def constrained_distortion_info(esses, distortion, p=100, diff_ds=1000,
                                eps=.1, d_mult=10):
    d1 = np.linspace(distortion + eps, d_mult*distortion, diff_ds)
    d2 = (d1*distortion)/(d1 - distortion)
    redundancy = feature_redundant_info(esses, d1, d2, p=p)
    info1 = feature_info_consumption(esses, d1, p=p)
    info2 = feature_info_consumption(esses, d2, p=p)
    pe = distortion_error_approx(esses, d1, d2, p=p)
    return (info1, d1), (info2, d2), redundancy, pe
                        
def minimal_error_redund(esses, distortion, p=100, diff_ds=1000,
                         eps=.1, d_mult=10, lam_start=0, lam_end=.125,
                         lam_n=1000):
    t1, t2, redund, pe = constrained_distortion_info(esses, distortion, p=p,
                                                     diff_ds=diff_ds, eps=eps,
                                                     d_mult=d_mult)
    i1, d1 = t1
    i2, d2 = t2
    lams = np.linspace(lam_start, lam_end, lam_n)
    opt_d1 = np.zeros((len(esses), lam_n))
    for i, s in enumerate(esses):
        for j, l in enumerate(lams):
            lagr = pe[i] + l*redund[i]
            opt_d1[i, j] = d1[np.argmin(lagr)]
    return opt_d1, lams

def line_picking_clt(x, overlapping_d):
    mu = np.sqrt(overlapping_d*1/6)
    var = 7/120.
    p = sts.norm(mu, np.sqrt(var)).pdf(x)
    return p

def line_picking_cube(x):
    """
    Taken from: http://mathworld.wolfram.com/CubeLinePicking.html
    """
    def _arcsec(y):
        v = np.arccos(1/y)
        return v
    
    def _arccsc(y):
        v = np.arcsin(1/y)
        return v

    l1 = lambda x: -(x**2)*((x - 8)*(x**2) + np.pi*(6*x - 4))
    l2 = lambda x:  2*x*(((x**2) - 8*np.sqrt((x**2) - 1) + 3)*(x**2)
                         - 4*np.sqrt((x**2) - 1) + 12*(x**2)*_arcsec(x)
                         + np.pi*(3 - 4*x) - .5)
    l3 = lambda x: x*((1 + (x**2))*(6*np.pi + 8*np.sqrt((x**2) - 2)
                                    - 5 - (x**2))
                      - 16*x*_arccsc(np.sqrt(2 - 2*(x**-2)))
                      + 16*x*np.arctan(x*np.sqrt((x**2) - 2))
                      - 24*((x**2) + 1)*np.arctan(np.sqrt((x**2) - 2)))
    conds = (x <= 1, np.logical_and(x > 1, x <= np.sqrt(2)),
             x > np.sqrt(2))
    funcs = (l1, l2, l3)
    p = np.piecewise(x, conds, funcs)
    return p

def line_picking_square(x):
    """
    Taken from: http://mathworld.wolfram.com/SquareLinePicking.html
    """
    l1 = lambda x: 2*x*((x**2) - 4*x + np.pi)
    l2 = lambda x: 2*x*(4*np.sqrt((x**2) - 1) - ((x**2) + 2 - np.pi)
                        - 4*np.arctan(np.sqrt((x**2) - 1)))
    conds = (x <= 1, x > 1)
    funcs = (l1, l2)
    p = np.piecewise(x, conds, funcs)
    return p

def line_picking_line(x):
    p = 2*(1 - x)
    return p
    
def integrate_assignment_error(esses, d1, d2, overlapping_d, p=100):
    if d1 is None and d2 is not None:
        d1 = d2
    elif d2 is None and d1 is not None:
        d2 = d1        
    d1, d2 = d1/(p**2), d2/(p**2)
    p = 1
    integ_end = np.sqrt(overlapping_d)
    if overlapping_d == 1:
        dist_pdf = line_picking_line
    elif overlapping_d == 2:
        dist_pdf = line_picking_square
    elif overlapping_d == 3:
        dist_pdf = line_picking_cube
    else:
        if overlapping_d < 10:
            print('Using CLT approximation for a small overlap, probably'
                  ' will not be accurate. {} < 10'.format(overlapping_d))
        dist_pdf = lambda x: line_picking_clt(x, overlapping_d)
    ae = ae_integ(esses, d1, d2, p=p, integ_start=0, integ_end=integ_end,
                  dist_pdf=dist_pdf)
    return ae
    
def ae_integ(esses, d1, d2, p=1, integ_start=0, integ_end=None, dist_pdf=None,
             err_thr=.01):
    pes = np.zeros_like(d1)
    for i, d1_i in enumerate(d1):
        d2_i = d2[i]
        def _f(x):
            v1 = dist_pdf(x)
            v2 = sts.norm(x, np.sqrt(d1_i + d2_i)).cdf(0)
            v = v1*v2
            return v
           
        pes[i], err = sin.quad(_f, integ_start, integ_end)
        assert err < err_thr
    errs = np.zeros((len(esses), len(d1)))
    for i, s in enumerate(esses):
        errs[i] = ss.comb(s, 2)*pes
    return errs

def distortion_func(bits, features, objs, overlaps, d1=None, p=100,
                    d_cand=None):
    if d1 is not None:
        d = _distortion_func_numerical(bits, features, objs, overlaps, d1, p=p,
                                       d_cand=d_cand)
    else:
        d = _distortion_func_analytical(bits, features, objs, overlaps, p=p)
    return d

def _distortion_func_analytical(bits, features, objs, overlaps, p=100):
    t1 = (p**2)/(2*np.pi)
    t2 = (1/2)**(2*overlaps/(features + overlaps))
    t3 = np.exp(-2*bits/(objs*(features + overlaps)))
    d = t1*t2*t3
    return d

def _distortion_func_numerical(bits, features, objs, overlaps, d1, p=100,
                               d_cand=None):
    if d_cand is None:
        d_cand = _distortion_func_analytical(bits, features, objs, overlaps,
                                             p=p)
    if np.array(d1).shape[0] > 1:
        d_cand = np.ones(d1.shape)*d_cand
        
    def _targ_func(d):
        t1 = (p/np.sqrt(2*np.pi))**(overlaps + features)
        t2 = (1/d)**(features/2)
        t3 = ((d1 - d)/(d1**2))**(overlaps/2)
        t4 = np.exp(bits/objs)
        out = t1*t2*t3 - t4
        return out
    
    lower = np.zeros_like(d_cand)
    upper = np.ones_like(d_cand)*np.inf
    res = sio.least_squares(_targ_func, d_cand, bounds=(lower, upper))
    d = res.x
    return d

def d_func(d1, overall_d):
    if d1 is None:
        d2 = [overall_d*2]
        d1 = [overall_d*2]
    else:
        d1 = overall_d*2 + np.array(d1)
        d2 = d1*overall_d/(d1 - overall_d)
    return np.array(d2), np.array(d1)

def weighted_errors_lambda(bits, features, objs, overlaps_list, d1_list=None,
                           p=100, lam_range=None, lam_beg=0, lam_end_mult=100,
                           lam_n=1000):
    if d1_list is None:
        d1_size = 1
    else:
        d1_size = len(d1_list)
    if lam_range is None:
        lam_range = np.linspace(lam_beg, lam_end_mult*p*features*objs, lam_n)
    totals = np.zeros((len(overlaps_list), d1_size, len(lam_range)))
    local_d = np.zeros((len(overlaps_list), d1_size))
    assignment_err = np.zeros_like(local_d)
    d1s = np.zeros_like(local_d)
    trans_matrix = np.ones((len(lam_range), d1_size))
    lam_range = lam_range.reshape((-1, 1))*trans_matrix
    for i, overlaps in enumerate(overlaps_list):
        obj_list = (objs,)
        initial_d = distortion_func(bits, features, objs, overlaps, p=p)
        d2_i, d1_i = d_func(d1_list, initial_d)
        overall_d = distortion_func(bits, features, objs, overlaps, d1=d1_i,
                                    p=p, d_cand=initial_d)
        assignment_d = integrate_assignment_error(obj_list, d1_i, d2_i,
                                                  overlaps, p=p)
        local_d[i] = overall_d
        assignment_err[i] = assignment_d
        w_sum = overall_d.reshape((1, -1)) + lam_range*assignment_d
        totals[i] = w_sum.T
        d1s[i] = d1_i
    return totals, local_d, assignment_err, d1s

def minimum_weighted_error_lambda(bits, features, objs, p=100, lam_range=None,
                                  lam_beg=0, lam_end_mult=3, lam_n=5000,
                                  d1_n=5000, d1_mult=100):
    d1_list = np.linspace(0, d1_mult*p, d1_n)
    overlaps = np.arange(1, features + 1, 1)
    if lam_range is None:
        lam_range = np.linspace(lam_beg, lam_end_mult*p*features*objs, lam_n)
    out = weighted_errors_lambda(bits, features, objs, overlaps,
                                 d1_list=d1_list, lam_range=lam_range, p=p)
    total, local_d, ae, d1s = out
    flat_shape = (len(overlaps)*len(d1_list), len(lam_range))
    o_i, d1_i = np.unravel_index(np.argmin(total.reshape(flat_shape), axis=0),
                                 (len(overlaps), len(d1_list)))
    opt_d1 = d1_list[d1_i]
    opt_over = overlaps[o_i]
    min_total = total[o_i, d1_i, range(len(lam_range))]
    return opt_d1, opt_over, min_total, lam_range, local_d, ae, total

def generate_1d_rfs(space_size, rf_spacing, rf_size, mult=1):
    rf_cents = np.arange(0, space_size + rf_spacing, rf_spacing)
    func = rfm.make_gaussian_vector_rf(rf_cents, rf_size, 1, 0)
    trs = lambda stims: mult*func(stims)
    return trs, rf_cents

def get_local_manifold_rfs(trs, pt, rf_mult=1, delt=5):
    diff = trs(pt) - trs(pt + delt)
    norm_diff = diff/delt # /np.sqrt(np.sum(diff**2))
    return norm_diff/rf_mult

def estimate_error(local, cov):
    est_var = np.dot(local, np.dot(cov, local))
    return est_var

def estimate_est_cov(local1, local2, cov):
    est_cov = np.dot(local1, np.dot(cov, local2))
    return est_cov

def error_theory(trs, n, dists, rf_mult, noise_cov):
    pairs = _make_pairs(dists, n)
    var = np.zeros(len(dists))
    cov = np.zeros_like(var)
    for i, p in enumerate(pairs):
        p1, p2 = p
        lm1 = get_local_manifold_rfs(trs, p1, rf_mult=rf_mult)[0]
        lm2 = get_local_manifold_rfs(trs, p2, rf_mult=rf_mult)[0]
        var[i] = estimate_error(lm1, noise_cov)
        cov[i] = estimate_est_cov(lm1, lm2, noise_cov)
    return var, cov

def make_covariance_matrix(size, var, corr_scale, corr_func=None):
    cov = np.identity(size)*var
    if corr_func is None:
        cov[cov == 0] = corr_scale*var
    else:
        for c in it.combinations(range(size), 2):
            cij = var*corr_scale*corr_func(c[0], c[1])
            cov[c[0], c[1]] = cij
            cov[c[1], c[0]] = cij
    return cov

def _make_pairs(dists, n):
    pairs = np.ones((len(dists), 2))*(n/2 - max(dists)/2)
    pairs[:, 1] = pairs[:, 1] + dists
    return pairs

def make_distance_cov_func(tau):
    return lambda x, y: np.exp(-((x - y)**2)/tau)

def estimate_encoding_swap(rf_size, rf_spacing, noise_var, noise_corr, dists,
                           noise_samps=1000, rf_mult=10, space_size=None,
                           buff=None, buff_mult=4, corr_func=None,
                           distortion=nmc.mse_distortion,
                           give_real=True, basin_hop=True, parallel=False,
                           oo=False, n_hops=100, c=1, space_mult=2):
    if buff is None:
        buff = rf_size*buff_mult
    if space_size is None:
        space_size = np.ceil(max(dists)) + space_mult*buff
        space_size = int(space_size)
    trs, cents = generate_1d_rfs(space_size, rf_spacing, rf_size, mult=rf_mult)
    d_pairs = _make_pairs(dists, space_size)
    all_pts = d_pairs.reshape((2*len(dists), 1))
    trs_pts = trs(all_pts)
    noise_mean = np.zeros(len(cents))
    noise_cov = make_covariance_matrix(len(cents), noise_var, noise_corr,
                                       corr_func=corr_func)
    noise_distrib = sts.multivariate_normal(noise_mean, noise_cov)
    noise = noise_distrib.rvs((len(dists), noise_samps))
    noise = np.expand_dims(noise, axis=1)
    trs_pts_samp = np.repeat(trs_pts, noise_samps, axis=0)
    trs_pts_struct = trs_pts_samp.reshape((len(dists), 2, noise_samps,
                                           len(cents)))
    noise_pts = trs_pts_struct + noise
    noise_pts_flat = noise_pts.reshape((len(dists)*2*noise_samps, -1))
    all_pts_ns = np.repeat(all_pts, noise_samps, axis=0)
    if give_real:
        give_pts = all_pts_ns
    else:
        give_pts = None
    decoded_pts = nmc.decode_pop_resp(c, noise_pts_flat, trs,
                                      space_size - buff,
                                      buff, real_pts=give_pts,
                                      basin_hop=basin_hop, parallel=parallel,
                                      niter=n_hops)
    dist = distortion(all_pts_ns, decoded_pts, axis=1)
    struct_dist = dist.reshape(d_pairs.shape + (noise_samps,))
    struct_decoded_pts = decoded_pts.reshape(struct_dist.shape)
    struct_all_pts = all_pts_ns.reshape(struct_dist.shape)
    return struct_dist, struct_decoded_pts, struct_all_pts

def analyze_decoding(clean_pts, decoded_pts, distortion):
    mean_dec = np.expand_dims(np.mean(decoded_pts, axis=2), axis=2)
    bias = mean_dec[:, :, 0] - clean_pts[:, :, 0]
    mse = np.mean((decoded_pts - clean_pts)**2, axis=2)
    var = np.var(decoded_pts, axis=2)
    cov = np.mean((decoded_pts[:, 0] - mean_dec[:, 0])
                  *(decoded_pts[:, 1] - mean_dec[:, 1]), axis=1)
    decoded_dists = decoded_pts[:, 1, :] - decoded_pts[:, 0, :]
    switches = np.sum(decoded_dists < 0, axis=1)/decoded_pts.shape[2]
    return bias, var, cov, switches, decoded_dists, mse

def characterize_encoding(rf_size, rf_spacing, noise_var, noise_corr, dists,
                          noise_samps=1000, space_size=None, buff=None,
                          buff_mult=4, distortion=nmc.mse_distortion,
                          give_real=True, basin_hop=False, parallel=False,
                          oo=False, n_hops=100, rf_mult=10, corr_func=None,
                          space_mult=2):
    n_dists = len(dists)
    n_nc = len(noise_corr)
    bias = np.zeros((n_nc, n_dists, 2))
    var = np.zeros_like(bias)
    mse = np.zeros_like(var)
    cov = np.zeros((n_nc, n_dists))
    switch = np.zeros((n_nc, n_dists))
    d_dists = np.zeros((n_nc, n_dists, noise_samps))
    distorted = np.zeros((n_nc, n_dists, 2, noise_samps))
    decoded = np.zeros_like(distorted)
    orig = np.zeros_like(distorted)
    for i, nc in enumerate(noise_corr):
        out = estimate_encoding_swap(rf_size, rf_spacing, noise_var, nc, dists,
                                     basin_hop=basin_hop, give_real=give_real, 
                                     noise_samps=noise_samps, n_hops=n_hops,
                                     space_size=space_size, corr_func=corr_func,
                                     rf_mult=rf_mult, distortion=distortion,
                                     space_mult=space_mult)
        distorted[i], decoded[i], orig[i] = out
        out = analyze_decoding(orig[i], decoded[i], distorted[i])
        bias[i], var[i], cov[i], switch[i], d_dists[i], mse[i] = out
    return bias, var, cov, switch, d_dists, mse, distorted, decoded, orig
        
