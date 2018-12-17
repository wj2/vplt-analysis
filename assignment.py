
import numpy as np
import itertools as it
import general.utility as u
import scipy.special as ss
import scipy.stats as sts

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
    return errs
