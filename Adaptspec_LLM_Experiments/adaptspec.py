"""
adaptspec.py  — fixed Python translation of BayesSpec R package
Key fixes vs original:
  1. beta initialised by sampling from posterior (not zeros)
  2. tau_up_limit default = 1e4 (matches R default of 10000)
  3. x is linearly detrended before fitting (matches R: lm(x~x0)$res)
  4. within() nposs=3 case uses -2+new_index (not -1+new_index)
  5. birth() xi_prop for seg_cut>0: uses xi_curr[jj-1]-1+tmin+index (matches R exactly)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def _lin_basis_func(freq, nbeta):
    nbasis = nbeta - 1
    n = len(freq)
    omega = np.zeros((n, nbasis))
    for j in range(1, nbasis + 1):
        omega[:, j-1] = np.sqrt(2) * np.cos(2*j*np.pi*freq) / (2*np.pi*j)
    return np.column_stack([np.ones(n), omega])


def _whittle_like(prdgrm, fhat, n):
    fhat = np.asarray(fhat).ravel()
    Ae = prdgrm * np.exp(-fhat)
    n1 = n // 2
    if n % 2 == 1:
        f = (-np.sum(fhat[1:n1+1] + Ae[1:n1+1])
             - 0.5*(fhat[0] + Ae[0])
             - 0.5*n*np.log(2*np.pi))
    else:
        f = (-np.sum(fhat[1:n1] + Ae[1:n1])
             - 0.5*(fhat[0] + Ae[0])
             - 0.5*(fhat[n1] + Ae[n1])
             - 0.5*n*np.log(2*np.pi))
    return float(f)


def _beta_log_post_grad_hess(param, n, nu_mat, prdgrm, precs):
    n1 = n // 2
    xxb = nu_mat @ param
    Ae = prdgrm * np.exp(-xxb)
    if n % 2 == 1:
        f = (-np.sum(xxb[1:n1+1] + Ae[1:n1+1]) - 0.5*(xxb[0]+Ae[0])
             - 0.5*(param @ (precs*param)))
        g = (-nu_mat[1:n1+1].T @ (1-Ae[1:n1+1]) - 0.5*(1-Ae[0])*nu_mat[0]
             - precs*param)
        h = (-nu_mat[1:n1+1].T @ (Ae[1:n1+1, np.newaxis]*nu_mat[1:n1+1])
             - 0.5*Ae[0]*np.outer(nu_mat[0], nu_mat[0]) - np.diag(precs))
    else:
        f = (-np.sum(xxb[1:n1] + Ae[1:n1]) - 0.5*(xxb[0]+Ae[0])
             - 0.5*(xxb[n1]+Ae[n1]) - 0.5*(param @ (precs*param)))
        g = (-nu_mat[1:n1].T @ (1-Ae[1:n1]) - 0.5*(1-Ae[0])*nu_mat[0]
             - 0.5*(1-Ae[n1])*nu_mat[n1] - precs*param)
        h = (-nu_mat[1:n1].T @ (Ae[1:n1, np.newaxis]*nu_mat[1:n1])
             - 0.5*Ae[0]*np.outer(nu_mat[0], nu_mat[0])
             - 0.5*Ae[n1]*np.outer(nu_mat[n1], nu_mat[n1]) - np.diag(precs))
    return f, g, h


def _post_beta(j, nseg, x, xi, tau, nbeta, nbasis, sigmasqalpha):
    seg = x[:xi[j]] if j == 0 else x[xi[j-1]:xi[j]]
    nseg_actual = len(seg)
    nfreq = nseg_actual // 2
    freq = np.arange(nfreq+1) / (2*nfreq) if nfreq > 0 else np.array([0.0])
    dft = np.fft.fft(seg) / np.sqrt(nseg_actual)
    prdgrm = np.abs(dft[:nfreq+1])**2
    nu_mat = _lin_basis_func(freq, nbeta)
    precs = np.concatenate([[1.0/sigmasqalpha], np.full(nbasis, 1.0/tau)])
    param0 = np.zeros(nbeta)

    def neg_fg(p):
        f, g, _ = _beta_log_post_grad_hess(p, nseg_actual, nu_mat, prdgrm, precs)
        return -f, -g
    def neg_h(p):
        _, _, h = _beta_log_post_grad_hess(p, nseg_actual, nu_mat, prdgrm, precs)
        return -h

    res = minimize(neg_fg, param0, method="trust-ncg", jac=True, hess=neg_h,
                   options={"maxiter": 200, "gtol": 1e-8})
    beta_mean = res.x
    H = _beta_log_post_grad_hess(beta_mean, nseg_actual, nu_mat, prdgrm, precs)[2]
    try:
        beta_var = -np.linalg.inv(H)
        beta_var = 0.5*(beta_var + beta_var.T)
        ev = np.linalg.eigvalsh(beta_var)
        if ev.min() < 1e-12:
            beta_var += (1e-10 - ev.min()) * np.eye(nbeta)
    except LinAlgError:
        beta_var = np.eye(nbeta) * 1e-4
    return dict(beta_mean=beta_mean, beta_var=beta_var, nu_mat=nu_mat, prdgrm=prdgrm)


def _sample_mvn(mean, cov, rng):
    try:
        return rng.multivariate_normal(mean, cov)
    except Exception:
        return mean + rng.standard_normal(len(mean)) * 1e-6


def _log_mvn_pdf(x, mean, cov):
    try:
        return float(multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=True))
    except Exception:
        return -1e30


def _move_probs(nexp, nexp_max):
    if nexp == 1:       return 0.5, 0.0, 0.5
    elif nexp == nexp_max: return 0.0, 0.5, 0.5
    else:               return 1/3, 1/3, 1/3


def _log_prior_cut(xi, nexp, nobs, tmin):
    lp = 0.0
    for k in range(nexp - 1):
        denom = (nobs - (xi[k-1] if k > 0 else 0) - (nexp-k+1)*tmin + 1)
        if denom <= 0: return -1e30
        lp -= np.log(denom)
    return lp


def _birth(x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
           log_move_curr, log_move_prop,
           nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng):
    nexp_prop = nexp_curr + 1
    eligible = np.where(nseg_curr > 2*tmin)[0]
    if len(eligible) == 0: return None
    seg_cut = rng.choice(eligible)
    nposs_seg = len(eligible)
    nposs_cut = nseg_curr[seg_cut] - 2*tmin + 1
    index = rng.integers(1, nposs_cut+1)
    zz = rng.uniform()
    xi_prop = np.zeros(nexp_prop, dtype=int)
    nseg_prop = np.zeros(nexp_prop, dtype=int)
    tau_prop = np.ones(nexp_prop)
    beta_prop = np.zeros((nbeta, nexp_prop))

    for jj in range(nexp_curr):
        if jj < seg_cut:
            xi_prop[jj] = xi_curr[jj]; tau_prop[jj] = tau_curr[jj]
            nseg_prop[jj] = nseg_curr[jj]; beta_prop[:, jj] = beta_curr[:, jj]
        elif jj == seg_cut:
            # FIX: R uses xi_curr[jj-1]-1+tmin+index for seg_cut>0
            xi_prop[seg_cut] = (index+tmin-1) if seg_cut == 0 else (xi_curr[jj-1]-1+tmin+index)
            xi_prop[seg_cut+1] = xi_curr[jj]
            tau_prop[seg_cut] = tau_curr[seg_cut]*zz/(1-zz)
            tau_prop[seg_cut+1] = tau_curr[seg_cut]*(1-zz)/zz
            nseg_prop[seg_cut] = index+tmin-1
            nseg_prop[seg_cut+1] = nseg_curr[jj] - nseg_prop[seg_cut]
            for k in [jj, jj+1]:
                fit = _post_beta(k, nseg_prop[k], x, xi_prop, tau_prop[k], nbeta, nbasis, sigmasqalpha)
                beta_prop[:, k] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
        else:
            xi_prop[jj+1] = xi_curr[jj]; tau_prop[jj+1] = tau_curr[jj]
            nseg_prop[jj+1] = nseg_curr[jj]; beta_prop[:, jj+1] = beta_curr[:, jj]

    log_jacobian = np.log(2*tau_curr[seg_cut]/(zz*(1-zz)))

    log_beta_prop = log_tau_prior_prop = log_beta_prior_prop = loglike_prop = 0.0
    for jj in [seg_cut, seg_cut+1]:
        fit = _post_beta(jj, nseg_prop[jj], x, xi_prop, tau_prop[jj], nbeta, nbasis, sigmasqalpha)
        log_beta_prop += _log_mvn_pdf(beta_prop[:, jj], fit["beta_mean"], fit["beta_var"])
        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_prop[jj])]))
        log_beta_prior_prop += _log_mvn_pdf(beta_prop[:, jj], np.zeros(nbeta), prior_cov)
        log_tau_prior_prop -= np.log(tau_up_limit)
        loglike_prop += _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_prop[:, jj], nseg_prop[jj])

    log_prior_cut_prop = _log_prior_cut(xi_prop, nexp_prop, nobs, tmin)
    log_proposal_prop = log_beta_prop - np.log(nposs_seg) + log_move_prop - np.log(nposs_cut)
    log_target_prop = loglike_prop + log_beta_prior_prop + log_tau_prior_prop + log_prior_cut_prop

    fit = _post_beta(seg_cut, nseg_curr[seg_cut], x, xi_curr, tau_curr[seg_cut], nbeta, nbasis, sigmasqalpha)
    if nexp_curr == 1:
        log_beta_curr = _log_mvn_pdf(beta_curr.ravel(), fit["beta_mean"], fit["beta_var"])
        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_curr[seg_cut])]))
        log_beta_prior_curr = _log_mvn_pdf(beta_curr.ravel(), np.zeros(nbeta), prior_cov)
        fhat = fit["nu_mat"] @ beta_curr.ravel()
    else:
        log_beta_curr = _log_mvn_pdf(beta_curr[:, seg_cut], fit["beta_mean"], fit["beta_var"])
        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_curr[seg_cut])]))
        log_beta_prior_curr = _log_mvn_pdf(beta_curr[:, seg_cut], np.zeros(nbeta), prior_cov)
        fhat = fit["nu_mat"] @ beta_curr[:, seg_cut]

    loglike_curr = _whittle_like(fit["prdgrm"], fhat, nseg_curr[seg_cut])
    log_prior_cut_curr = _log_prior_cut(xi_curr, nexp_curr, nobs, tmin)
    log_target_curr = loglike_curr + log_beta_prior_curr - np.log(tau_up_limit) + log_prior_cut_curr
    log_proposal_curr = log_beta_curr + log_move_curr

    log_alpha = log_target_prop - log_target_curr + log_proposal_curr - log_proposal_prop + log_jacobian
    met_rat = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))
    return dict(met_rat=met_rat, nseg_prop=nseg_prop, xi_prop=xi_prop,
                tau_prop=tau_prop, beta_prop=beta_prop)


def _death(x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
           log_move_curr, log_move_prop,
           nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng):
    nexp_prop = nexp_curr - 1
    if nexp_prop < 1: return None
    cut_del = rng.integers(0, nexp_curr-1)
    xi_prop = np.zeros(nexp_prop, dtype=int)
    nseg_prop = np.zeros(nexp_prop, dtype=int)
    tau_prop = np.ones(nexp_prop)
    beta_prop = np.zeros((nbeta, nexp_prop))
    j = 0; k_prop = 0
    loglike_prop = log_beta_prop = log_beta_prior_prop = log_tau_prior_prop = 0.0
    log_beta_curr = log_beta_prior_curr = log_tau_prior_curr = loglike_curr = log_jacobian = 0.0

    while j < nexp_curr:
        if j == cut_del:
            xi_prop[k_prop] = xi_curr[j+1]
            tau_prop[k_prop] = np.sqrt(tau_curr[j]*tau_curr[j+1])
            nseg_prop[k_prop] = nseg_curr[j] + nseg_curr[j+1]
            fit = _post_beta(k_prop, nseg_prop[k_prop], x, xi_prop, tau_prop[k_prop], nbeta, nbasis, sigmasqalpha)
            beta_prop[:, k_prop] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
            loglike_prop = _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_prop[:, k_prop], nseg_prop[k_prop])
            log_beta_prop = _log_mvn_pdf(beta_prop[:, k_prop], fit["beta_mean"], fit["beta_var"])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_prop[k_prop])]))
            log_beta_prior_prop = _log_mvn_pdf(beta_prop[:, k_prop], np.zeros(nbeta), prior_cov)
            log_tau_prior_prop = -np.log(tau_up_limit)
            log_jacobian = -np.log(2*(np.sqrt(tau_curr[j])+np.sqrt(tau_curr[j+1]))**2)
            for jj in [j, j+1]:
                fit2 = _post_beta(jj, nseg_curr[jj], x, xi_curr, tau_curr[jj], nbeta, nbasis, sigmasqalpha)
                log_beta_curr += _log_mvn_pdf(beta_curr[:, jj], fit2["beta_mean"], fit2["beta_var"])
                prior_cov2 = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_curr[jj])]))
                log_beta_prior_curr += _log_mvn_pdf(beta_curr[:, jj], np.zeros(nbeta), prior_cov2)
                log_tau_prior_curr -= np.log(tau_up_limit)
                loglike_curr += _whittle_like(fit2["prdgrm"], fit2["nu_mat"] @ beta_curr[:, jj], nseg_curr[jj])
            j += 1
        else:
            xi_prop[k_prop] = xi_curr[j]; tau_prop[k_prop] = tau_curr[j]
            nseg_prop[k_prop] = nseg_curr[j]; beta_prop[:, k_prop] = beta_curr[:, j]
        j += 1; k_prop += 1

    log_prior_cut_prop = _log_prior_cut(xi_prop, nexp_prop, nobs, tmin)
    log_prior_cut_curr = _log_prior_cut(xi_curr, nexp_curr, nobs, tmin)
    log_target_prop = loglike_prop + log_tau_prior_prop + log_beta_prior_prop + log_prior_cut_prop
    log_target_curr = loglike_curr + log_beta_prior_curr + log_tau_prior_curr + log_prior_cut_curr
    log_proposal_prop = log_beta_prop - np.log(nexp_curr-1) + log_move_prop
    log_proposal_curr = log_move_curr + log_beta_curr
    log_alpha = log_target_prop - log_target_curr + log_proposal_curr - log_proposal_prop + log_jacobian
    met_rat = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))
    return dict(met_rat=met_rat, nseg_prop=nseg_prop, xi_prop=xi_prop,
                tau_prop=tau_prop, beta_prop=beta_prop)


def _within(x, nexp, xi_curr, beta_curr, nseg_curr, tau,
            nobs, nbeta, nbasis, sigmasqalpha, tmin, prob_mm1, rng):
    xi_prop = xi_curr.copy()
    beta_prop = beta_curr.copy()
    nseg_new = nseg_curr.copy()

    if nexp > 1:
        seg_temp = rng.integers(0, nexp-1)
        u = rng.uniform()
        cut_poss_curr = xi_curr[seg_temp]
        nposs_prior = nseg_curr[seg_temp] + nseg_curr[seg_temp+1] - 2*tmin + 1

        if u < prob_mm1:
            s1 = nseg_curr[seg_temp]; s2 = nseg_curr[seg_temp+1]
            if s1 == tmin and s2 == tmin:
                new_index = 1
                cut_poss_new = xi_curr[seg_temp] - 1 + new_index
            elif s1 == tmin:
                new_index = rng.integers(1, 3)
                cut_poss_new = xi_curr[seg_temp] - 1 + new_index
            elif s2 == tmin:
                new_index = rng.integers(1, 3)
                cut_poss_new = xi_curr[seg_temp] + 1 - new_index
            else:
                new_index = rng.integers(1, 4)
                # FIX: R uses -2+new_index for the general case
                cut_poss_new = xi_curr[seg_temp] - 2 + new_index
        else:
            new_index = rng.integers(1, nposs_prior+1)
            cut_poss_new = (-1+tmin+new_index) if seg_temp == 0 else (sum(nseg_curr[:seg_temp])-1+tmin+new_index)

        xi_prop[seg_temp] = cut_poss_new
        nseg_new[seg_temp] = xi_prop[seg_temp] - (xi_curr[seg_temp-1] if seg_temp > 0 else 0)
        nseg_new[seg_temp+1] = nseg_curr[seg_temp] + nseg_curr[seg_temp+1] - nseg_new[seg_temp]

        if nseg_new[seg_temp] < tmin or nseg_new[seg_temp+1] < tmin:
            return dict(epsilon=0.0, xi_prop=xi_curr.copy(), beta_prop=beta_curr.copy(),
                        nseg_new=nseg_curr.copy(), seg_temp=seg_temp)

        dist = abs(cut_poss_new - cut_poss_curr)
        s1_new = nseg_new[seg_temp]; s2_new = nseg_new[seg_temp+1]

        # FIX: R computes log_prop_cut_curr based on CURRENT sizes, log_prop_cut_prop based on NEW sizes
        def _log_prop_cut(s1_q, s2_q, dist_q):
            if dist_q > 1:
                return np.log(1-prob_mm1) - np.log(nposs_prior)
            elif s1_q == tmin and s2_q == tmin:
                return 0.0
            elif s1_q == tmin or s2_q == tmin:
                return np.log(1-prob_mm1)-np.log(nposs_prior)+np.log(0.5)+np.log(prob_mm1)
            else:
                return np.log(1-prob_mm1)-np.log(nposs_prior)+np.log(1/3)+np.log(prob_mm1)

        # R: log_prop_cut_curr uses CURRENT seg sizes (nseg_curr), log_prop_cut_prop uses NEW seg sizes
        log_prop_cut_curr = _log_prop_cut(nseg_curr[seg_temp], nseg_curr[seg_temp+1], dist)
        log_prop_cut_prop = _log_prop_cut(s1_new, s2_new, dist)

        loglike_curr = log_beta_curr_temp = log_prior_curr = 0.0
        for j in [seg_temp, seg_temp+1]:
            fit = _post_beta(j, nseg_curr[j], x, xi_curr, tau[j], nbeta, nbasis, sigmasqalpha)
            log_beta_curr_temp += _log_mvn_pdf(beta_curr[:, j], fit["beta_mean"], fit["beta_var"])
            loglike_curr += _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_curr[:, j], nseg_curr[j])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau[j])]))
            log_prior_curr += _log_mvn_pdf(beta_curr[:, j], np.zeros(nbeta), prior_cov)

        loglike_prop = log_beta_prop = log_prior_prop = 0.0
        for j in [seg_temp, seg_temp+1]:
            fit = _post_beta(j, nseg_new[j], x, xi_prop, tau[j], nbeta, nbasis, sigmasqalpha)
            beta_prop[:, j] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
            log_beta_prop += _log_mvn_pdf(beta_prop[:, j], fit["beta_mean"], fit["beta_var"])
            loglike_prop += _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_prop[:, j], nseg_new[j])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau[j])]))
            log_prior_prop += _log_mvn_pdf(beta_prop[:, j], np.zeros(nbeta), prior_cov)

        log_target_prop = loglike_prop + log_prior_prop + _log_prior_cut(xi_prop, nexp, nobs, tmin)
        log_target_curr = loglike_curr + log_prior_curr + _log_prior_cut(xi_curr, nexp, nobs, tmin)
        log_proposal_curr = log_beta_curr_temp + log_prop_cut_curr
        log_proposal_prop = log_beta_prop + log_prop_cut_prop

    else:
        seg_temp = 0
        fit = _post_beta(0, nobs, x, xi_prop, tau, nbeta, nbasis, sigmasqalpha)
        beta_prop = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng).reshape(-1, 1)
        log_beta_prop = _log_mvn_pdf(beta_prop.ravel(), fit["beta_mean"], fit["beta_var"])
        log_beta_curr_temp = _log_mvn_pdf(beta_curr.ravel(), fit["beta_mean"], fit["beta_var"])
        loglike_prop = _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_prop.ravel(), nobs)
        loglike_curr = _whittle_like(fit["prdgrm"], fit["nu_mat"] @ beta_curr.ravel(), nobs)
        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau)]))
        log_prior_prop = _log_mvn_pdf(beta_prop.ravel(), np.zeros(nbeta), prior_cov)
        log_prior_curr = _log_mvn_pdf(beta_curr.ravel(), np.zeros(nbeta), prior_cov)
        log_target_prop = loglike_prop + log_prior_prop
        log_target_curr = loglike_curr + log_prior_curr
        log_proposal_curr = log_beta_curr_temp
        log_proposal_prop = log_beta_prop

    log_alpha = log_target_prop - log_target_curr + log_proposal_curr - log_proposal_prop
    epsilon = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))
    return dict(epsilon=epsilon, xi_prop=xi_prop, beta_prop=beta_prop,
                nseg_new=nseg_new, seg_temp=seg_temp)


def _update_tau(beta_j, nbasis, tau_up_limit, rng):
    """Sample tau from truncated inverse-gamma full conditional (matches R's qigamma)."""
    from scipy.stats import invgamma
    b_smooth = beta_j[1:]
    shape = nbasis/2.0 + (-1)    # tau_prior_a = -1 in R default
    rate = 0.5*np.dot(b_smooth, b_smooth) + 0  # tau_prior_b = 0 in R default
    # shape = nbasis/2 + tau_prior_a = nbasis/2 - 1
    # R: tau_a <- nbasis/2 + tau_prior_a;  tau_b <- sum(beta[2:nbeta]^2)/2 + tau_prior_b
    # Then: u <- runif(1); const1 <- pigamma(tau_up_limit, tau_a, tau_b)
    #       tausq <- qigamma(u*const1, tau_a, tau_b)
    # This is CDF-inversion sampling of IG truncated to [0, tau_up_limit]
    a = nbasis/2.0 - 1.0   # tau_prior_a = -1
    b = 0.5*np.dot(b_smooth, b_smooth)
    if b < 1e-15:
        return rng.uniform(0, tau_up_limit)
    # IG(a, b): CDF at tau_up_limit
    cdf_limit = invgamma.cdf(tau_up_limit, a=a, scale=b)
    if cdf_limit < 1e-15:
        return rng.uniform(0, tau_up_limit)
    u = rng.uniform() * cdf_limit
    return float(invgamma.ppf(u, a=a, scale=b))


@dataclass
class AdaptSpecResult:
    nexp_samples: np.ndarray
    xi_samples: list
    tau_samples: list
    beta_samples: list
    nobs: int; nbeta: int; tmin: int; nexp_max: int; niter: int; nburn: int
    n_birth_accept: int = 0; n_birth_propose: int = 0
    n_death_accept: int = 0; n_death_propose: int = 0
    n_within_accept: int = 0; n_within_propose: int = 0


def adaptspec(x, nexp_max=10, nbeta=7, niter=5000, nburn=2000, tmin=40,
              sigmasqalpha=100.0, tau_up_limit=1e4,   # FIX: R default tau_up_limit=10000
              prob_mm1=0.8,                             # FIX: R default prob_mm1=0.8
              seed=None, verbose=True):
    """
    AdaptSPEC: Bayesian nonstationary spectral estimation via RJMCMC.
    Defaults match the R BayesSpec package.
    """
    x = np.asarray(x, dtype=float).ravel()
    # FIX: R detrends x before fitting: x <- lm(x ~ x0)$res
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)

    nobs = len(x)
    nbasis = nbeta - 1
    rng = np.random.default_rng(seed)

    if nobs < nexp_max * tmin:
        raise ValueError(f"Series too short: need {nexp_max*tmin}, got {nobs}.")

    # Initialise: 1 segment, tau random, beta sampled from posterior (FIX: not zeros)
    nexp_curr = 1
    xi_curr = np.array([nobs], dtype=int)
    nseg_curr = np.array([nobs], dtype=int)
    tau_curr = np.array([rng.uniform(0, tau_up_limit)])   # FIX: R uses runif(1)*tau_up_limit

    fit0 = _post_beta(0, nobs, x, xi_curr, tau_curr[0], nbeta, nbasis, sigmasqalpha)
    beta_curr = _sample_mvn(fit0["beta_mean"], fit0["beta_var"], rng).reshape(nbeta, 1)  # FIX: sample not zeros

    niter_kept = niter - nburn
    nexp_samples = np.zeros(niter_kept, dtype=int)
    xi_samples = []; tau_samples = []; beta_samples = []
    n_birth_accept = n_birth_propose = 0
    n_death_accept = n_death_propose = 0
    n_within_accept = n_within_propose = 0
    check_every = max(1, niter // 10)

    for it in range(niter):
        if verbose and it % check_every == 0:
            print(f"  [{100*it//niter:3d}%] iter {it:6d}/{niter}  nexp={nexp_curr}", flush=True)

        p_birth, p_death, p_within = _move_probs(nexp_curr, nexp_max)
        u = rng.uniform()
        if u < p_birth:           move = "birth"
        elif u < p_birth+p_death: move = "death"
        else:                     move = "within"

        if move == "birth":
            log_move_curr = np.log(p_birth) if p_birth > 0 else -np.inf
            p_b2, p_d2, _ = _move_probs(nexp_curr+1, nexp_max)
            log_move_prop = np.log(p_d2) if p_d2 > 0 else -np.inf
            n_birth_propose += 1
            result = _birth(x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
                            log_move_curr, log_move_prop,
                            nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng)
            if result is not None and rng.uniform() < result["met_rat"]:
                nexp_curr += 1; xi_curr = result["xi_prop"]; nseg_curr = result["nseg_prop"]
                tau_curr = result["tau_prop"]; beta_curr = result["beta_prop"]; n_birth_accept += 1

        elif move == "death":
            log_move_curr = np.log(p_death) if p_death > 0 else -np.inf
            p_b2, p_d2, _ = _move_probs(nexp_curr-1, nexp_max)
            log_move_prop = np.log(p_b2) if p_b2 > 0 else -np.inf
            n_death_propose += 1
            result = _death(x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
                            log_move_curr, log_move_prop,
                            nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng)
            if result is not None and rng.uniform() < result["met_rat"]:
                nexp_curr -= 1; xi_curr = result["xi_prop"]; nseg_curr = result["nseg_prop"]
                tau_curr = result["tau_prop"]; beta_curr = result["beta_prop"]; n_death_accept += 1

        else:
            n_within_propose += 1
            tau_arg = tau_curr[0] if nexp_curr == 1 else tau_curr
            result = _within(x, nexp_curr, xi_curr, beta_curr, nseg_curr, tau_arg,
                             nobs, nbeta, nbasis, sigmasqalpha, tmin, prob_mm1, rng)
            if rng.uniform() < result["epsilon"]:
                xi_curr = result["xi_prop"]; beta_curr = result["beta_prop"]
                nseg_curr = result["nseg_new"]; n_within_accept += 1

        # Tau Gibbs update
        for j in range(nexp_curr):
            b_j = beta_curr[:, j] if beta_curr.ndim == 2 else beta_curr.ravel()
            tau_curr[j] = _update_tau(b_j, nbasis, tau_up_limit, rng)

        if it >= nburn:
            idx = it - nburn
            nexp_samples[idx] = nexp_curr
            xi_samples.append(xi_curr.copy())
            tau_samples.append(tau_curr.copy())
            beta_samples.append(beta_curr.copy())

    if verbose:
        print(f"  [100%] Done.")
        print(f"  Birth  accept: {n_birth_accept}/{n_birth_propose} = {n_birth_accept/max(1,n_birth_propose):.2%}")
        print(f"  Death  accept: {n_death_accept}/{n_death_propose} = {n_death_accept/max(1,n_death_propose):.2%}")
        print(f"  Within accept: {n_within_accept}/{n_within_propose} = {n_within_accept/max(1,n_within_propose):.2%}")

    return AdaptSpecResult(
        nexp_samples=nexp_samples, xi_samples=xi_samples,
        tau_samples=tau_samples, beta_samples=beta_samples,
        nobs=nobs, nbeta=nbeta, tmin=tmin, nexp_max=nexp_max,
        niter=niter, nburn=nburn,
        n_birth_accept=n_birth_accept, n_birth_propose=n_birth_propose,
        n_death_accept=n_death_accept, n_death_propose=n_death_propose,
        n_within_accept=n_within_accept, n_within_propose=n_within_propose,
    )


def summarise(result):
    nobs = result.nobs; S = len(result.nexp_samples)
    mean_nexp = result.nexp_samples.mean()
    vals, counts = np.unique(result.nexp_samples, return_counts=True)
    nexp_probs = {int(v): float(c)/S for v, c in zip(vals, counts)}
    cp_count = np.zeros(nobs)
    for xi in result.xi_samples:
        for t in xi[:-1]:
            if 0 < t < nobs: cp_count[t] += 1
    changepoint_proba = cp_count / S
    modal_cps = sorted([t for t in range(1, nobs) if changepoint_proba[t] > 0.05],
                       key=lambda t: -changepoint_proba[t])
    modal_nexp = max(nexp_probs, key=nexp_probs.get)
    best_xi = None; best_score = -np.inf
    for xi in result.xi_samples:
        if len(xi) == modal_nexp:
            score = sum(changepoint_proba[t] for t in xi[:-1] if 0 < t < nobs)
            if score > best_score: best_score = score; best_xi = xi
    if best_xi is not None:
        boundaries = []; prev = 0
        for t in best_xi: boundaries.append((prev, t)); prev = t
    else:
        boundaries = [(0, nobs)]
    return dict(mean_nexp=mean_nexp, nexp_probs=nexp_probs,
                changepoint_proba=changepoint_proba, modal_changepoints=modal_cps,
                segment_boundaries=boundaries, modal_nexp=modal_nexp)