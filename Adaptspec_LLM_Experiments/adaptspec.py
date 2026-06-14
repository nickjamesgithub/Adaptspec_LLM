"""
adaptspec.py
============
Pure-Python replica of the AdaptSPEC algorithm from the BayesSpec R package.

Reference
---------
Rosen, O., Wood, S. and Stoffer, D. (2012).
"AdaptSPEC: Adaptive Spectral Estimation for Nonstationary Time Series."
Journal of the American Statistical Association, 107, 1575–1589.
doi:10.1080/01621459.2012.716340

Source translated from: https://github.com/mbertolacci/BayesSpec
  R/adaptspec.R, R/birth_fun.R, R/death_fun.R, R/within_fun.R,
  R/post_beta.R, R/beta_derivs.R, R/whittle_like.R, R/lin_basis_func.R

Usage
-----
    import numpy as np
    from adaptspec import adaptspec, summarise

    log_returns = np.random.randn(500)   # your time series

    result = adaptspec(
        x          = log_returns,
        nexp_max   = 10,    # max number of segments
        nbeta      = 7,     # number of basis coefficients (smoothing spline order)
        niter      = 5000,  # total MCMC iterations
        nburn      = 2000,  # burn-in to discard
        tmin       = 20,    # minimum segment length
    )

    summary = summarise(result)
    print("Posterior mean number of segments:", summary["mean_nexp"])
    print("Most likely change-points:", summary["modal_changepoints"])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import solve, LinAlgError
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Linear basis functions  (R: lin_basis_func)
# ---------------------------------------------------------------------------

def _lin_basis_func(freq: np.ndarray, nbeta: int) -> np.ndarray:
    """
    Cosine basis matrix on [0, 0.5].

    Parameters
    ----------
    freq  : 1-D array of Fourier frequencies (0/2nfreq, 1/2nfreq, …, 1/2)
    nbeta : total number of basis functions (1 intercept + nbeta-1 cosines)

    Returns
    -------
    nu : (len(freq), nbeta) matrix
    """
    nbasis = nbeta - 1
    n = len(freq)
    omega = np.zeros((n, nbasis))
    for j in range(1, nbasis + 1):
        omega[:, j - 1] = np.sqrt(2) * np.cos(2 * j * np.pi * freq) / (2 * np.pi * j)
    return np.column_stack([np.ones(n), omega])


# ---------------------------------------------------------------------------
# Whittle log-likelihood  (R: whittle_like)
# ---------------------------------------------------------------------------

def _whittle_like(prdgrm: np.ndarray, fhat: np.ndarray, n: int) -> float:
    """
    Whittle log-likelihood for a single segment.

    Parameters
    ----------
    prdgrm : periodogram ordinates (length nfreq+1)
    fhat   : log spectral density at same frequencies
    n      : segment length
    """
    fhat = np.asarray(fhat).ravel()
    Ae = prdgrm * np.exp(-fhat)
    n1 = n // 2
    if n % 2 == 1:                                         # odd n
        f = (-np.sum(fhat[1:n1 + 1] + Ae[1:n1 + 1])
             - 0.5 * (fhat[0] + Ae[0])
             - 0.5 * n * np.log(2 * np.pi))
    else:                                                  # even n
        f = (-np.sum(fhat[1:n1] + Ae[1:n1])
             - 0.5 * (fhat[0] + Ae[0])
             - 0.5 * (fhat[n1] + Ae[n1])
             - 0.5 * n * np.log(2 * np.pi))
    return float(f)


# ---------------------------------------------------------------------------
# Posterior mode / Hessian for beta  (R: beta_derivs + post_beta)
# ---------------------------------------------------------------------------

def _beta_log_posterior_and_grad_hess(
    param: np.ndarray,
    n: int,
    nu_mat: np.ndarray,
    prdgrm: np.ndarray,
    precs: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Log-posterior, gradient and Hessian for the spline coefficient beta.
    Matches R's beta_derivs exactly.
    """
    n1 = n // 2
    xxb = nu_mat @ param
    Ae = prdgrm * np.exp(-xxb)

    if n % 2 == 1:
        f = (- np.sum(xxb[1:n1 + 1] + Ae[1:n1 + 1])
             - 0.5 * (xxb[0] + Ae[0])
             - 0.5 * (param @ (precs * param)))
        g = (- nu_mat[1:n1 + 1, :].T @ (1 - Ae[1:n1 + 1])
             - 0.5 * (1 - Ae[0]) * nu_mat[0, :]
             - precs * param)
        h = (- nu_mat[1:n1 + 1, :].T @ (Ae[1:n1 + 1:, np.newaxis] * nu_mat[1:n1 + 1, :])
             - 0.5 * Ae[0] * np.outer(nu_mat[0, :], nu_mat[0, :])
             - np.diag(precs))
    else:
        f = (- np.sum(xxb[1:n1] + Ae[1:n1])
             - 0.5 * (xxb[0] + Ae[0])
             - 0.5 * (xxb[n1] + Ae[n1])
             - 0.5 * (param @ (precs * param)))
        g = (- nu_mat[1:n1, :].T @ (1 - Ae[1:n1])
             - 0.5 * (1 - Ae[0]) * nu_mat[0, :]
             - 0.5 * (1 - Ae[n1]) * nu_mat[n1, :]
             - precs * param)
        h = (- nu_mat[1:n1, :].T @ (Ae[1:n1, np.newaxis] * nu_mat[1:n1, :])
             - 0.5 * Ae[0] * np.outer(nu_mat[0, :], nu_mat[0, :])
             - 0.5 * Ae[n1] * np.outer(nu_mat[n1, :], nu_mat[n1, :])
             - np.diag(precs))
    return f, g, h


def _post_beta(
    j: int,
    nseg: int,
    x: np.ndarray,
    xi: np.ndarray,
    tau: float,
    nbeta: int,
    nbasis: int,
    sigmasqalpha: float,
) -> dict:
    """
    Compute Laplace-approximation mean and covariance for beta_j.
    Matches R's post_beta.

    xi   : 0-indexed endpoint array (last obs index in segment, 1-indexed in R)
           Here xi[j] == index of *last* element of segment j (0-based, inclusive).
    """
    # Extract the segment observations
    if j > 0:
        seg = x[xi[j - 1]:xi[j]]     # xi stored as exclusive upper bound (see below)
    else:
        seg = x[:xi[j]]

    nseg_actual = len(seg)
    nfreq = nseg_actual // 2
    freq = np.arange(nfreq + 1) / (2 * nfreq) if nfreq > 0 else np.array([0.0])

    dft = np.fft.fft(seg) / np.sqrt(nseg_actual)
    y = dft[:nfreq + 1]
    prdgrm = np.abs(y) ** 2

    nu_mat = _lin_basis_func(freq, nbeta)
    precs = np.concatenate([[1.0 / sigmasqalpha], np.full(nbasis, 1.0 / tau)])

    # Trust-region optimisation using scipy's Newton-CG (matches R's trust())
    param0 = np.zeros(nbeta)

    def neg_f_g(p):
        f, g, _ = _beta_log_posterior_and_grad_hess(p, nseg_actual, nu_mat, prdgrm, precs)
        return -f, -g

    def neg_hess(p):
        _, _, h = _beta_log_posterior_and_grad_hess(p, nseg_actual, nu_mat, prdgrm, precs)
        return -h

    res = minimize(neg_f_g, param0, method="trust-ncg",
                   jac=True, hess=neg_hess,
                   options={"maxiter": 200, "gtol": 1e-8})

    beta_mean = res.x
    # Hessian of log-posterior (negative of optimised negative-Hessian)
    H = _beta_log_posterior_and_grad_hess(beta_mean, nseg_actual, nu_mat, prdgrm, precs)[2]
    try:
        beta_var = -np.linalg.inv(H)
        # Ensure positive definiteness
        beta_var = 0.5 * (beta_var + beta_var.T)
        eigvals = np.linalg.eigvalsh(beta_var)
        if eigvals.min() < 1e-12:
            beta_var += (1e-10 - eigvals.min()) * np.eye(nbeta)
    except LinAlgError:
        beta_var = np.eye(nbeta) * 1e-4

    return dict(beta_mean=beta_mean, beta_var=beta_var, nu_mat=nu_mat, prdgrm=prdgrm,
                nseg=nseg_actual)


def _sample_mvn(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one sample from N(mean, cov); falls back gracefully on bad matrices."""
    try:
        return rng.multivariate_normal(mean, cov)
    except (LinAlgError, ValueError):
        return mean + rng.standard_normal(len(mean)) * 1e-6


def _log_mvn_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log density of multivariate normal."""
    try:
        return float(multivariate_normal.logpdf(x, mean=mean, cov=cov, allow_singular=True))
    except Exception:
        return -1e30


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _nseg_from_xi(xi: np.ndarray, nobs: int) -> np.ndarray:
    """
    Given endpoint array xi (exclusive upper boundaries, 0-based),
    return the length of each segment.
    xi has length nexp; xi[-1] == nobs.
    """
    nexp = len(xi)
    nseg = np.zeros(nexp, dtype=int)
    nseg[0] = xi[0]
    for j in range(1, nexp):
        nseg[j] = xi[j] - xi[j - 1]
    return nseg


# ---------------------------------------------------------------------------
# Move probabilities
# ---------------------------------------------------------------------------

def _move_probs(nexp: int, nexp_max: int) -> Tuple[float, float, float]:
    """Return (p_birth, p_death, p_within)."""
    if nexp == 1:
        return 0.5, 0.0, 0.5
    elif nexp == nexp_max:
        return 0.0, 0.5, 0.5
    else:
        return 1 / 3, 1 / 3, 1 / 3


# ---------------------------------------------------------------------------
# Prior for cut-points
# ---------------------------------------------------------------------------

def _log_prior_cut(xi: np.ndarray, nexp: int, nobs: int, tmin: int) -> float:
    lp = 0.0
    for k in range(nexp - 1):
        if k == 0:
            denom = nobs - (nexp - k + 1) * tmin + 1
        else:
            denom = nobs - xi[k - 1] - (nexp - k + 1) * tmin + 1
        if denom <= 0:
            return -1e30   # effectively zero probability
        lp -= np.log(denom)
    return lp


# ---------------------------------------------------------------------------
# BIRTH move  (R: birth_fun.R)
# ---------------------------------------------------------------------------

def _birth(
    x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
    log_move_curr, log_move_prop,
    nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng
):
    nexp_prop = nexp_curr + 1

    # Segments eligible for splitting
    eligible = np.where(nseg_curr > 2 * tmin)[0]
    if len(eligible) == 0:
        return None   # cannot split

    seg_cut = rng.choice(eligible)
    nposs_seg = len(eligible)
    nposs_cut = nseg_curr[seg_cut] - 2 * tmin + 1

    # Build proposed configuration
    xi_prop = np.zeros(nexp_prop, dtype=int)
    nseg_prop = np.zeros(nexp_prop, dtype=int)
    tau_prop = np.ones(nexp_prop)
    beta_prop = np.zeros((nbeta, nexp_prop))

    index = rng.integers(1, nposs_cut + 1)   # 1 to nposs_cut inclusive

    zz = rng.uniform()

    for jj in range(nexp_curr):
        if jj < seg_cut:
            xi_prop[jj] = xi_curr[jj]
            tau_prop[jj] = tau_curr[jj]
            nseg_prop[jj] = nseg_curr[jj]
            beta_prop[:, jj] = beta_curr[:, jj]
        elif jj == seg_cut:
            if seg_cut == 0:
                xi_prop[seg_cut] = index + tmin - 1
            else:
                xi_prop[seg_cut] = xi_curr[jj - 1] + tmin + index - 1
            xi_prop[seg_cut + 1] = xi_curr[jj]
            tau_prop[seg_cut] = tau_curr[seg_cut] * zz / (1 - zz)
            tau_prop[seg_cut + 1] = tau_curr[seg_cut] * (1 - zz) / zz
            nseg_prop[seg_cut] = index + tmin - 1
            nseg_prop[seg_cut + 1] = nseg_curr[jj] - nseg_prop[seg_cut]
            for k in [jj, jj + 1]:
                fit = _post_beta(k, nseg_prop[k], x, xi_prop, tau_prop[k],
                                 nbeta, nbasis, sigmasqalpha)
                beta_prop[:, k] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
        else:
            xi_prop[jj + 1] = xi_curr[jj]
            tau_prop[jj + 1] = tau_curr[jj]
            nseg_prop[jj + 1] = nseg_curr[jj]
            beta_prop[:, jj + 1] = beta_curr[:, jj]

    log_jacobian = np.log(2 * tau_curr[seg_cut] / (zz * (1 - zz)))

    # Proposed log densities
    log_beta_prop = 0.0
    log_tau_prior_prop = 0.0
    log_beta_prior_prop = 0.0
    loglike_prop = 0.0
    for jj in [seg_cut, seg_cut + 1]:
        fit = _post_beta(jj, nseg_prop[jj], x, xi_prop, tau_prop[jj],
                         nbeta, nbasis, sigmasqalpha)
        log_beta_prop += _log_mvn_pdf(beta_prop[:, jj], fit["beta_mean"], fit["beta_var"])
        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_prop[jj])]))
        log_beta_prior_prop += _log_mvn_pdf(beta_prop[:, jj], np.zeros(nbeta), prior_cov)
        log_tau_prior_prop -= np.log(tau_up_limit)
        fhat = fit["nu_mat"] @ beta_prop[:, jj]
        loglike_prop += _whittle_like(fit["prdgrm"], fhat, nseg_prop[jj])

    log_prior_cut_prop = _log_prior_cut(xi_prop, nexp_prop, nobs, tmin)
    log_proposal_prop = log_beta_prop - np.log(nposs_seg) + log_move_prop - np.log(nposs_cut)
    log_prior_prop = log_beta_prior_prop + log_tau_prior_prop + log_prior_cut_prop
    log_target_prop = loglike_prop + log_prior_prop

    # Current log densities
    fit = _post_beta(seg_cut, nseg_curr[seg_cut], x, xi_curr, tau_curr[seg_cut],
                     nbeta, nbasis, sigmasqalpha)
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

    log_tau_prior_curr = -np.log(tau_up_limit)
    loglike_curr = _whittle_like(fit["prdgrm"], fhat, nseg_curr[seg_cut])

    log_prior_cut_curr = _log_prior_cut(xi_curr, nexp_curr, nobs, tmin)
    log_proposal_curr = log_beta_curr + log_move_curr
    log_prior_curr = log_beta_prior_curr + log_tau_prior_curr + log_prior_cut_curr
    log_target_curr = loglike_curr + log_prior_curr

    log_alpha = (log_target_prop - log_target_curr
                 + log_proposal_curr - log_proposal_prop
                 + log_jacobian)
    met_rat = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))

    return dict(met_rat=met_rat, nseg_prop=nseg_prop, xi_prop=xi_prop,
                tau_prop=tau_prop, beta_prop=beta_prop)


# ---------------------------------------------------------------------------
# DEATH move  (R: death_fun.R)
# ---------------------------------------------------------------------------

def _death(
    x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
    log_move_curr, log_move_prop,
    nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng
):
    nexp_prop = nexp_curr - 1
    if nexp_prop < 1:
        return None

    cut_del = rng.integers(0, nexp_curr - 1)   # index of cut-point to remove (0-based)

    xi_prop = np.zeros(nexp_prop, dtype=int)
    nseg_prop = np.zeros(nexp_prop, dtype=int)
    tau_prop = np.ones(nexp_prop)
    beta_prop = np.zeros((nbeta, nexp_prop))

    j = 0
    k_prop = 0
    loglike_prop = 0.0
    log_beta_prop = 0.0
    log_beta_prior_prop = 0.0
    log_tau_prior_prop = 0.0
    log_beta_curr = 0.0
    log_beta_prior_curr = 0.0
    log_tau_prior_curr = 0.0
    loglike_curr = 0.0
    log_jacobian = 0.0

    while j < nexp_curr:
        if j == cut_del:
            # Merge segment j and j+1
            xi_prop[k_prop] = xi_curr[j + 1]
            tau_prop[k_prop] = np.sqrt(tau_curr[j] * tau_curr[j + 1])
            nseg_prop[k_prop] = nseg_curr[j] + nseg_curr[j + 1]

            fit = _post_beta(k_prop, nseg_prop[k_prop], x, xi_prop, tau_prop[k_prop],
                             nbeta, nbasis, sigmasqalpha)
            beta_prop[:, k_prop] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
            fhat = fit["nu_mat"] @ beta_prop[:, k_prop]
            loglike_prop = _whittle_like(fit["prdgrm"], fhat, nseg_prop[k_prop])

            log_beta_prop = _log_mvn_pdf(beta_prop[:, k_prop], fit["beta_mean"], fit["beta_var"])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_prop[k_prop])]))
            log_beta_prior_prop = _log_mvn_pdf(beta_prop[:, k_prop], np.zeros(nbeta), prior_cov)
            log_tau_prior_prop = -np.log(tau_up_limit)

            log_jacobian = -np.log(2 * (np.sqrt(tau_curr[j]) + np.sqrt(tau_curr[j + 1])) ** 2)

            # Current: two segments j and j+1
            for jj in [j, j + 1]:
                fit2 = _post_beta(jj, nseg_curr[jj], x, xi_curr, tau_curr[jj],
                                  nbeta, nbasis, sigmasqalpha)
                log_beta_curr += _log_mvn_pdf(beta_curr[:, jj], fit2["beta_mean"], fit2["beta_var"])
                prior_cov2 = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau_curr[jj])]))
                log_beta_prior_curr += _log_mvn_pdf(beta_curr[:, jj], np.zeros(nbeta), prior_cov2)
                log_tau_prior_curr -= np.log(tau_up_limit)
                fhat2 = fit2["nu_mat"] @ beta_curr[:, jj]
                loglike_curr += _whittle_like(fit2["prdgrm"], fhat2, nseg_curr[jj])

            j += 1  # skip the next segment (merged)
        else:
            xi_prop[k_prop] = xi_curr[j]
            tau_prop[k_prop] = tau_curr[j]
            nseg_prop[k_prop] = nseg_curr[j]
            beta_prop[:, k_prop] = beta_curr[:, j]

        j += 1
        k_prop += 1

    log_prior_cut_prop = _log_prior_cut(xi_prop, nexp_prop, nobs, tmin)
    log_prior_cut_curr = _log_prior_cut(xi_curr, nexp_curr, nobs, tmin)

    log_target_prop = (loglike_prop + log_tau_prior_prop + log_beta_prior_prop
                       + log_prior_cut_prop)
    log_target_curr = (loglike_curr + log_beta_prior_curr + log_tau_prior_curr
                       + log_prior_cut_curr)

    log_proposal_prop = log_beta_prop - np.log(nexp_curr - 1) + log_move_prop
    log_proposal_curr = log_move_curr + log_beta_curr

    log_alpha = (log_target_prop - log_target_curr
                 + log_proposal_curr - log_proposal_prop
                 + log_jacobian)
    met_rat = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))

    return dict(met_rat=met_rat, nseg_prop=nseg_prop, xi_prop=xi_prop,
                tau_prop=tau_prop, beta_prop=beta_prop)


# ---------------------------------------------------------------------------
# WITHIN move  (R: within_fun.R)
# ---------------------------------------------------------------------------

def _within(
    x, nexp, xi_curr, beta_curr, nseg_curr, tau,
    nobs, nbeta, nbasis, sigmasqalpha, tmin, prob_mm1, rng
):
    xi_prop = xi_curr.copy()
    beta_prop = beta_curr.copy()
    nseg_new = nseg_curr.copy()

    if nexp > 1:
        seg_temp = rng.integers(0, nexp - 1)   # segment whose right boundary we move
        u = rng.uniform()
        cut_poss_curr = xi_curr[seg_temp]
        nposs_prior = nseg_curr[seg_temp] + nseg_curr[seg_temp + 1] - 2 * tmin + 1

        if u < prob_mm1:
            # Small local move (±1 or 0)
            s1 = nseg_curr[seg_temp]
            s2 = nseg_curr[seg_temp + 1]
            if s1 == tmin and s2 == tmin:
                nposs = 1
                new_index = 1
            elif s1 == tmin:
                nposs = 2
                new_index = rng.integers(1, 3)
            elif s2 == tmin:
                nposs = 2
                new_index = rng.integers(1, 3)
                new_index = 3 - new_index   # mirror
            else:
                nposs = 3
                new_index = rng.integers(1, 4)

            if s2 == tmin and s1 != tmin:
                cut_poss_new = xi_curr[seg_temp] + 1 - new_index
            else:
                cut_poss_new = xi_curr[seg_temp] - 1 + new_index
        else:
            # Global move uniformly from all valid positions
            new_index = rng.integers(1, nposs_prior + 1)
            if seg_temp > 0:
                cut_poss_new = sum(nseg_curr[:seg_temp]) - 1 + tmin + new_index
            else:
                cut_poss_new = -1 + tmin + new_index
            nposs = nposs_prior

        xi_prop[seg_temp] = cut_poss_new
        if seg_temp > 0:
            nseg_new[seg_temp] = xi_prop[seg_temp] - xi_curr[seg_temp - 1]
        else:
            nseg_new[seg_temp] = xi_prop[seg_temp]
        nseg_new[seg_temp + 1] = (nseg_curr[seg_temp] + nseg_curr[seg_temp + 1]
                                  - nseg_new[seg_temp])

        # Proposal log densities for the cut-point
        dist = abs(cut_poss_new - cut_poss_curr)
        s1_new = nseg_new[seg_temp]
        s2_new = nseg_new[seg_temp + 1]

        def _log_prop_cut(s1_q, s2_q, dist_q):
            if dist_q > 1:
                return np.log(1 - prob_mm1) - np.log(nposs_prior)
            elif s1_q == tmin and s2_q == tmin:
                return 0.0
            elif s1_q == tmin or s2_q == tmin:
                return (np.log(1 - prob_mm1) - np.log(nposs_prior)
                        + np.log(0.5) + np.log(prob_mm1))
            else:
                return (np.log(1 - prob_mm1) - np.log(nposs_prior)
                        + np.log(1 / 3) + np.log(prob_mm1))

        log_prop_cut_prop = _log_prop_cut(s1_new, s2_new, dist)
        log_prop_cut_curr = _log_prop_cut(nseg_curr[seg_temp], nseg_curr[seg_temp + 1], dist)

        # Evaluate likelihood / prior at current and proposed
        loglike_curr = 0.0
        log_beta_curr_temp = 0.0
        log_prior_curr = 0.0
        for j in [seg_temp, seg_temp + 1]:
            fit = _post_beta(j, nseg_curr[j], x, xi_curr, tau[j], nbeta, nbasis, sigmasqalpha)
            log_beta_curr_temp += _log_mvn_pdf(beta_curr[:, j], fit["beta_mean"], fit["beta_var"])
            fhat = fit["nu_mat"] @ beta_curr[:, j]
            loglike_curr += _whittle_like(fit["prdgrm"], fhat, nseg_curr[j])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau[j])]))
            log_prior_curr += _log_mvn_pdf(beta_curr[:, j], np.zeros(nbeta), prior_cov)

        loglike_prop = 0.0
        log_beta_prop = 0.0
        log_prior_prop = 0.0
        for j in [seg_temp, seg_temp + 1]:
            fit = _post_beta(j, nseg_new[j], x, xi_prop, tau[j], nbeta, nbasis, sigmasqalpha)
            beta_prop[:, j] = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng)
            log_beta_prop += _log_mvn_pdf(beta_prop[:, j], fit["beta_mean"], fit["beta_var"])
            fhat = fit["nu_mat"] @ beta_prop[:, j]
            loglike_prop += _whittle_like(fit["prdgrm"], fhat, nseg_new[j])
            prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau[j])]))
            log_prior_prop += _log_mvn_pdf(beta_prop[:, j], np.zeros(nbeta), prior_cov)

        log_prior_cut_prop = _log_prior_cut(xi_prop, nexp, nobs, tmin)
        log_prior_cut_curr = _log_prior_cut(xi_curr, nexp, nobs, tmin)

        log_target_prop = loglike_prop + log_prior_prop + log_prior_cut_prop
        log_target_curr = loglike_curr + log_prior_curr + log_prior_cut_curr

        log_proposal_curr = log_beta_curr_temp + log_prop_cut_curr
        log_proposal_prop = log_beta_prop + log_prop_cut_prop

    else:
        # Only one segment — update beta only
        seg_temp = 0
        fit = _post_beta(0, nobs, x, xi_prop, tau, nbeta, nbasis, sigmasqalpha)
        beta_prop = _sample_mvn(fit["beta_mean"], fit["beta_var"], rng).reshape(-1, 1)
        log_beta_prop = _log_mvn_pdf(beta_prop.ravel(), fit["beta_mean"], fit["beta_var"])
        log_beta_curr_temp = _log_mvn_pdf(beta_curr.ravel(), fit["beta_mean"], fit["beta_var"])

        fhat_prop = fit["nu_mat"] @ beta_prop.ravel()
        fhat_curr = fit["nu_mat"] @ beta_curr.ravel()
        loglike_prop = _whittle_like(fit["prdgrm"], fhat_prop, nobs)
        loglike_curr = _whittle_like(fit["prdgrm"], fhat_curr, nobs)

        prior_cov = np.diag(np.concatenate([[sigmasqalpha], np.full(nbasis, tau)]))
        log_prior_prop = _log_mvn_pdf(beta_prop.ravel(), np.zeros(nbeta), prior_cov)
        log_prior_curr = _log_mvn_pdf(beta_curr.ravel(), np.zeros(nbeta), prior_cov)

        log_target_prop = loglike_prop + log_prior_prop
        log_target_curr = loglike_curr + log_prior_curr
        log_proposal_curr = log_beta_curr_temp
        log_proposal_prop = log_beta_prop

    log_alpha = (log_target_prop - log_target_curr
                 + log_proposal_curr - log_proposal_prop)
    epsilon = min(1.0, np.exp(np.clip(log_alpha, -500, 500)))

    return dict(epsilon=epsilon, xi_prop=xi_prop, beta_prop=beta_prop,
                nseg_new=nseg_new, seg_temp=seg_temp)


# ---------------------------------------------------------------------------
# Tau update (Gibbs, uniform prior on [0, tau_up_limit])
# ---------------------------------------------------------------------------

def _update_tau(beta_j: np.ndarray, nbasis: int, tau_up_limit: float,
                rng: np.random.Generator) -> float:
    """
    Sample tau from its full conditional: inverse-gamma with uniform prior,
    i.e. IG(nbasis/2, sum(beta[1:]^2)/2) truncated to [0, tau_up_limit].
    We use a simple rejection sampler (usually accepts quickly).
    """
    b_smooth = beta_j[1:]   # exclude intercept
    shape = nbasis / 2.0
    rate = 0.5 * np.dot(b_smooth, b_smooth)
    if rate < 1e-15:
        return rng.uniform(0, tau_up_limit)
    # Draw from IG(shape, rate) via 1/Gamma(shape, 1/rate)
    for _ in range(1000):
        g = rng.gamma(shape=shape, scale=1.0 / rate)
        tau_cand = 1.0 / g
        if 0 < tau_cand <= tau_up_limit:
            return tau_cand
    return tau_up_limit * rng.uniform(0.5, 1.0)


# ---------------------------------------------------------------------------
# Main AdaptSPEC function
# ---------------------------------------------------------------------------

@dataclass
class AdaptSpecResult:
    """Container for MCMC output."""
    nexp_samples: np.ndarray          # (niter_kept,) number of segments at each kept iteration
    xi_samples: list                  # list of arrays of change-point positions (exclusive end)
    tau_samples: list                 # list of tau arrays
    beta_samples: list                # list of beta matrices
    nobs: int
    nbeta: int
    tmin: int
    nexp_max: int
    niter: int
    nburn: int

    # Acceptance counts (diagnostic)
    n_birth_accept: int = 0
    n_birth_propose: int = 0
    n_death_accept: int = 0
    n_death_propose: int = 0
    n_within_accept: int = 0
    n_within_propose: int = 0


def adaptspec(
    x: np.ndarray,
    nexp_max: int = 10,
    nbeta: int = 7,
    niter: int = 5000,
    nburn: int = 2000,
    tmin: int = 20,
    sigmasqalpha: float = 100.0,
    tau_up_limit: float = 1.0,
    prob_mm1: float = 0.4,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> AdaptSpecResult:
    """
    Run AdaptSPEC: Bayesian spectral analysis with RJMCMC change-point detection.

    Parameters
    ----------
    x             : 1-D time series (e.g. log returns).
    nexp_max      : Maximum number of segments.
    nbeta         : Number of spline basis coefficients per segment (≥ 2).
    niter         : Total MCMC iterations (including burn-in).
    nburn         : Number of burn-in iterations to discard.
    tmin          : Minimum segment length. Must satisfy nobs ≥ nexp_max * tmin.
    sigmasqalpha  : Prior variance on the intercept coefficient β₀.
    tau_up_limit  : Upper bound of uniform prior on τ (spline smoothing parameter).
    prob_mm1      : Probability of small ±1 cutpoint move in the Within step.
    seed          : Random seed for reproducibility.
    verbose       : Print progress every 10 %.

    Returns
    -------
    AdaptSpecResult with MCMC samples from post-burn-in iterations.
    """
    x = np.asarray(x, dtype=float).ravel()
    nobs = len(x)
    nbasis = nbeta - 1
    rng = np.random.default_rng(seed)

    if nobs < nexp_max * tmin:
        raise ValueError(
            f"Series too short: need at least nexp_max*tmin={nexp_max * tmin} "
            f"observations, got {nobs}."
        )

    # ---- Initialise ----
    nexp_curr = 1
    xi_curr = np.array([nobs], dtype=int)
    nseg_curr = np.array([nobs], dtype=int)
    tau_curr = np.ones(1)
    beta_curr = np.zeros((nbeta, 1))

    # Storage
    niter_kept = niter - nburn
    nexp_samples = np.zeros(niter_kept, dtype=int)
    xi_samples: List = []
    tau_samples: List = []
    beta_samples: List = []

    n_birth_accept = n_birth_propose = 0
    n_death_accept = n_death_propose = 0
    n_within_accept = n_within_propose = 0

    check_every = max(1, niter // 10)

    for it in range(niter):
        if verbose and it % check_every == 0:
            pct = 100 * it // niter
            print(f"  [{pct:3d}%] iter {it:6d}/{niter}  nexp={nexp_curr}", flush=True)

        # ---- Choose move type ----
        p_birth, p_death, p_within = _move_probs(nexp_curr, nexp_max)

        u = rng.uniform()
        if u < p_birth:
            move = "birth"
        elif u < p_birth + p_death:
            move = "death"
        else:
            move = "within"

        if move == "birth":
            log_move_curr = np.log(p_birth) if p_birth > 0 else -np.inf
            p_b2, p_d2, _ = _move_probs(nexp_curr + 1, nexp_max)
            log_move_prop = np.log(p_d2) if p_d2 > 0 else -np.inf
            n_birth_propose += 1

            result = _birth(
                x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
                log_move_curr, log_move_prop,
                nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng
            )
            if result is not None and rng.uniform() < result["met_rat"]:
                nexp_curr += 1
                xi_curr = result["xi_prop"]
                nseg_curr = result["nseg_prop"]
                tau_curr = result["tau_prop"]
                beta_curr = result["beta_prop"]
                n_birth_accept += 1

        elif move == "death":
            log_move_curr = np.log(p_death) if p_death > 0 else -np.inf
            p_b2, p_d2, _ = _move_probs(nexp_curr - 1, nexp_max)
            log_move_prop = np.log(p_b2) if p_b2 > 0 else -np.inf
            n_death_propose += 1

            result = _death(
                x, nexp_curr, xi_curr, nseg_curr, beta_curr, tau_curr,
                log_move_curr, log_move_prop,
                nobs, nbeta, nbasis, sigmasqalpha, tmin, tau_up_limit, rng
            )
            if result is not None and rng.uniform() < result["met_rat"]:
                nexp_curr -= 1
                xi_curr = result["xi_prop"]
                nseg_curr = result["nseg_prop"]
                tau_curr = result["tau_prop"]
                beta_curr = result["beta_prop"]
                n_death_accept += 1

        else:  # within
            n_within_propose += 1
            tau_arg = tau_curr[0] if nexp_curr == 1 else tau_curr

            result = _within(
                x, nexp_curr, xi_curr, beta_curr, nseg_curr, tau_arg,
                nobs, nbeta, nbasis, sigmasqalpha, tmin, prob_mm1, rng
            )
            if rng.uniform() < result["epsilon"]:
                xi_curr = result["xi_prop"]
                beta_curr = result["beta_prop"]
                nseg_curr = result["nseg_new"]
                n_within_accept += 1

        # ---- Update tau (Gibbs) for each segment ----
        for j in range(nexp_curr):
            b_j = beta_curr[:, j] if beta_curr.ndim == 2 else beta_curr.ravel()
            tau_curr[j] = _update_tau(b_j, nbasis, tau_up_limit, rng)

        # ---- Store post-burn-in ----
        if it >= nburn:
            idx = it - nburn
            nexp_samples[idx] = nexp_curr
            xi_samples.append(xi_curr.copy())
            tau_samples.append(tau_curr.copy())
            beta_samples.append(beta_curr.copy())

    if verbose:
        print(f"  [100%] Done.")
        print(f"  Birth  accept rate: {n_birth_accept}/{n_birth_propose} "
              f"= {n_birth_accept / max(1, n_birth_propose):.2%}")
        print(f"  Death  accept rate: {n_death_accept}/{n_death_propose} "
              f"= {n_death_accept / max(1, n_death_propose):.2%}")
        print(f"  Within accept rate: {n_within_accept}/{n_within_propose} "
              f"= {n_within_accept / max(1, n_within_propose):.2%}")

    return AdaptSpecResult(
        nexp_samples=nexp_samples,
        xi_samples=xi_samples,
        tau_samples=tau_samples,
        beta_samples=beta_samples,
        nobs=nobs,
        nbeta=nbeta,
        tmin=tmin,
        nexp_max=nexp_max,
        niter=niter,
        nburn=nburn,
        n_birth_accept=n_birth_accept,
        n_birth_propose=n_birth_propose,
        n_death_accept=n_death_accept,
        n_death_propose=n_death_propose,
        n_within_accept=n_within_accept,
        n_within_propose=n_within_propose,
    )


# ---------------------------------------------------------------------------
# Summarisation helpers
# ---------------------------------------------------------------------------

def summarise(result: AdaptSpecResult) -> dict:
    """
    Compute posterior summaries from an AdaptSpecResult.

    Returns a dict with:
      - mean_nexp             : posterior mean number of segments
      - nexp_probs            : dict {k: P(nexp=k)}
      - changepoint_proba     : array of length nobs, P(change-point at t)
      - modal_changepoints    : list of time indices with highest marginal cp probability
      - segment_boundaries    : list of (start, end) tuples for the modal segmentation
    """
    nobs = result.nobs
    S = len(result.nexp_samples)

    mean_nexp = result.nexp_samples.mean()

    # Posterior distribution over number of segments
    vals, counts = np.unique(result.nexp_samples, return_counts=True)
    nexp_probs = {int(v): float(c) / S for v, c in zip(vals, counts)}

    # Marginal change-point probability at each time point
    cp_count = np.zeros(nobs)
    for xi in result.xi_samples:
        # xi contains exclusive upper boundaries; change-points are at xi[0:-1]
        for t in xi[:-1]:  # last boundary is always nobs, not a change-point
            if 0 < t < nobs:
                cp_count[t] += 1
    changepoint_proba = cp_count / S

    # Modal change-points: local maxima above 5 % threshold
    threshold = 0.05
    modal_cps = sorted(
        [t for t in range(1, nobs) if changepoint_proba[t] > threshold],
        key=lambda t: -changepoint_proba[t]
    )

    # Most probable number of segments and their boundaries
    modal_nexp = max(nexp_probs, key=nexp_probs.get)
    # Find the iteration whose xi most closely matches modal
    # (use the sample with nexp == modal_nexp that has highest cp proba sum)
    best_xi = None
    best_score = -np.inf
    for xi in result.xi_samples:
        if len(xi) == modal_nexp:
            score = sum(changepoint_proba[t] for t in xi[:-1] if 0 < t < nobs)
            if score > best_score:
                best_score = score
                best_xi = xi

    if best_xi is not None:
        boundaries = []
        prev = 0
        for t in best_xi:
            boundaries.append((prev, t))
            prev = t
    else:
        boundaries = [(0, nobs)]

    return dict(
        mean_nexp=mean_nexp,
        nexp_probs=nexp_probs,
        changepoint_proba=changepoint_proba,
        modal_changepoints=modal_cps,
        segment_boundaries=boundaries,
        modal_nexp=modal_nexp,
    )


def plot_results(x: np.ndarray, result: AdaptSpecResult, figsize=(14, 8)):
    """
    Plot the time series with change-point probabilities.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")

    summary = summarise(result)
    cp_proba = summary["changepoint_proba"]
    boundaries = summary["segment_boundaries"]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, hspace=0.4)

    # -- Panel 1: time series with segment shading --
    ax1 = fig.add_subplot(gs[0:2])
    ax1.plot(x, color="#333333", lw=0.8, label="Series")
    colours = ["#d6eaf8", "#d5f5e3", "#fef9e7", "#fdedec", "#f5eef8"]
    for i, (s, e) in enumerate(boundaries):
        ax1.axvspan(s, e - 1, alpha=0.35, color=colours[i % len(colours)],
                    label=f"Segment {i + 1}")
    ax1.set_title(
        f"AdaptSPEC — posterior modal segmentation "
        f"(nexp={summary['modal_nexp']}, mean={summary['mean_nexp']:.2f})"
    )
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper right", fontsize=7, ncol=3)

    # -- Panel 2: change-point probability --
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.fill_between(np.arange(len(cp_proba)), cp_proba, color="#e74c3c", alpha=0.6)
    ax2.axhline(0.05, color="grey", lw=0.8, ls="--", label="5 % threshold")
    ax2.set_xlabel("Time index")
    ax2.set_ylabel("P(change-point)")
    ax2.set_title("Marginal change-point probabilities")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("AdaptSPEC Python — demo with synthetic piecewise AR series")
    print("=" * 60)

    rng_demo = np.random.default_rng(42)
    n = 512

    # Piecewise AR: three regimes separated at t=170 and t=340
    seg1 = np.zeros(170)
    seg1[0] = rng_demo.standard_normal()
    for t in range(1, 170):
        seg1[t] = 0.9 * seg1[t - 1] + rng_demo.standard_normal() * 0.5

    seg2 = np.zeros(170)
    seg2[0] = rng_demo.standard_normal()
    for t in range(1, 170):
        seg2[t] = -0.7 * seg2[t - 1] + rng_demo.standard_normal() * 0.5

    seg3 = rng_demo.standard_normal(172) * 0.3

    x_demo = np.concatenate([seg1, seg2, seg3])

    result = adaptspec(
        x_demo,
        nexp_max=8,
        nbeta=7,
        niter=3000,
        nburn=1000,
        tmin=30,
        seed=42,
        verbose=True,
    )

    summary = summarise(result)
    print()
    print(f"Posterior mean number of segments : {summary['mean_nexp']:.3f}")
    print(f"Most probable nexp                : {summary['modal_nexp']}")
    print(f"nexp posterior distribution       : {summary['nexp_probs']}")
    print(f"Modal segment boundaries          : {summary['segment_boundaries']}")

    top_cps = sorted(summary["modal_changepoints"][:5],
                     key=lambda t: -summary["changepoint_proba"][t])
    if top_cps:
        print(f"Top change-point positions        : {top_cps}")
        for t in top_cps:
            print(f"  t={t:4d}  P(cp)={summary['changepoint_proba'][t]:.3f}")

    # Attempt plot
    try:
        fig = plot_results(x_demo, result)
        fig.savefig("adaptspec_demo.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to adaptspec_demo.png")
    except ImportError:
        print("\nInstall matplotlib to generate the plot: pip install matplotlib")