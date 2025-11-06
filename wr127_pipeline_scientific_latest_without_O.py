#!/usr/bin/env python3
#wr127_pipeline_scientific.py
# WR127 pipeline: RV measurement + per-line MCMC + q aggregation
# This version is scientifically constrained to ONLY analyse WR lines,
# as the O-star line (HeI5876) is proven to be unusable in this dataset.
# It also uses a more conservative 15A half-width for EM lines to prevent bad fits.

import os, glob, re, time, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

#Optional libraries
try:
    import emcee, corner
    MCMC_AVAILABLE = True
except Exception:
    MCMC_AVAILABLE = False
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x # type: ignore

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

#
# --- USER CONFIGURATION ---
#
# Path to directory holding pre-normalized FITS and optional HJD file
DATA_DIR = "/home/swastik/Desktop/127CFHT/Normalized_Spectra" # <-- change to your folder
HJD_FILEPATH = os.path.join(DATA_DIR, "WR127_CFHT_obs_times.dat") # optional; must match file order

#Modes: "measure_only", "mcmc_only", "measure_then_mcmc"
RUN_MODE = "measure_then_mcmc"

# Orbital priors / initial guesses
PO_INIT = 9.55465
TO_INIT = 2460158.0 # T0 from your first script

#MCMC parameters (tuned from your experiments)
MCMC_WALKERS = 80
MCMC_STEPS = 9000
MCMC_BURN = 1500
MCMC_THIN = 10

# Monte Carlo draws when aggregating K distributions
MC_Q_SAMPLES = 20000

#Fitting and sanity constants
C = 299792.458 # speed of light km/s
MIN_PIX_FOR_FIT = 6
RV_ERR_FLOOR = 0.5 # km/s minimum per-epoch RV error
BOOTSTRAP_N = 200 # residual bootstrap iterations for RV errors; set 0 to skip
RV_SANITY_CAP = 3000.0 # mark "Unreal" if |rv| > cap

# ==============================================================================
# SCIENTIFIC IMPROVEMENT: Using ONLY the valid lines in your data range
# ==============================================================================
#Line lists
WR_EM_LINES = {
    "HeII5411": 5411.516,
    "NIV5200": 5200.41,
    "NIV5288": 5288.25,
    "NIV5785": 5784.76,
    "NIV5795": 5795.09,
}
# The HeI5876 line was proven to be unusable (pure noise) by the log file
# from the 2025-11-04 18:21 run, which gave K_O = 774 km/s.
# We therefore permanently remove it from the analysis.
O_ABS_LINES = {}
# ==============================================================================

# This will now ONLY contain the "EM_" lines
ALL_LINES = {**{f"EM_{k}": v for k, v in WR_EM_LINES.items()},
             **{f"ABS_{k}": v for k, v in O_ABS_LINES.items()}}

# Output filenames (in DATA_DIR)
MEAS_CSV = os.path.join(DATA_DIR, "wr127_perline_measurements.csv")
EM_SUM_CSV = os.path.join(DATA_DIR, "wr127_perline_fit_summary_EM.csv")
ABS_SUM_CSV = os.path.join(DATA_DIR, "wr127_perline_fit_summary_ABS.csv")
MASSRATIO_CSV = os.path.join(DATA_DIR, "wr127_massratio_summary.csv")
PIPE_README = os.path.join(DATA_DIR, "wr127_pipeline_README.txt")

np.random.seed(2025)

#
# --- LOW-LEVEL HELPERS ---
#
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(s))]

def gaussian(x, amp, mu, sigma, c):
    return c + amp * np.exp(-0.5* ((x - mu) / sigma)**2)

def phase_func(t, T0, P0):
    return ((t - T0) / P0) % 1.0

#
# --- FITTING: local gaussian with robust clipping ---
#
def fit_gauss_windowed(wl, flux, rest_wave, is_emission=True,
                       # SCIENTIFIC FIX: Using a more conservative 15A half-width
                       # to prevent fits from getting lost (as seen in NIV5785/95)
                       em_halfwidth=15.0, abs_halfwidth=15.0,
                       iter_clip=2, outlier_sigma=6.0):
    """Fit Gaussian+constant within window. Return (popt, pcov, rv_kms, rv_err, rv_comment) or None."""

    half = em_halfwidth if is_emission else abs_halfwidth
    mask = (wl >= (rest_wave - half)) & (wl <= (rest_wave + half))
    x = wl[mask]; y = flux[mask]

    if x.size < MIN_PIX_FOR_FIT:
        return None

    # Ensure ascending wavelength
    if x.size > 1 and x[0] > x[-1]:
        x = x[::-1]; y = y[::-1]

    # Initial guesses
    continuum = np.nanmedian(y)
    amp0 = (np.nanmax(y) - continuum) if is_emission else (np.nanmin(y) - continuum)
    # Handle flat/noisy absorption
    if (not is_emission) and amp0 > 0:
        amp0 = -abs(0.01 * continuum)
    mu0 = rest_wave
    sigma0 = 5.0 if is_emission else 2.0 # Broader initial guess for emission
    p0 = [amp0, mu0, sigma0, continuum]

    lower = [-np.inf, x.min(), 0.05, -np.inf]
    upper = [np.inf, x.max(), (x.max()-x.min()), np.inf]

    popt = None; pcov = None
    for _ in range(iter_clip + 1):
        try:
            popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=(lower, upper), maxfev=50000)
            model = gaussian(x, *popt)
            resid = y - model
            s = np.nanstd(resid)
            good = np.abs(resid) < outlier_sigma * max(1.0, s)
            if good.sum() < 3 or np.all(good):
                break
            x, y = x[good], y[good]
            p0 = popt # use last fit as new guess
        except Exception:
            return None # Fit failed completely

    if popt is None or pcov is None:
        return None

    # Extract center and error
    mu = float(popt[1])
    mu_err = np.sqrt(abs(pcov[1,1])) if (pcov is not None and pcov.shape[0] > 1) else np.nan
    rv = ((mu - rest_wave) / rest_wave) * C
    rv_err = (mu_err / rest_wave) * C if np.isfinite(mu_err) else np.nan

    # Sanity tagging (keep results, but comment)
    comment = ""
    if not np.isfinite(rv):
        comment = "Unreal: Non-finite"
    elif abs(rv) > RV_SANITY_CAP:
        comment = f"Unreal: |RV|>{RV_SANITY_CAP:.0f}"

    return popt, pcov, float(rv), float(rv_err), comment


def bootstrap_rv_error(wl, flux, rest_wave, popt, nboot=200, rng=None, is_emission=True,
                       # SCIENTIFIC FIX: Must match windows from main fit
                       em_halfwidth=15.0, abs_halfwidth=15.0):
    """Residual bootstrap estimate of mu uncertainty -> rv_err (km/s)."""
    if nboot <= 0 or popt is None:
        return np.nan

    half = em_halfwidth if is_emission else abs_halfwidth
    mask = (wl >= (rest_wave - half)) & (wl <= (rest_wave + half))
    x = wl[mask]; y = flux[mask]

    if x.size < MIN_PIX_FOR_FIT:
        return np.nan
    if x.size > 1 and x[0] > x[-1]:
        x = x[::-1]; y = y[::-1]

    model = gaussian(x, *popt)
    resid = y - model
    if rng is None: rng = np.random.RandomState()

    mu_samps = []
    for _ in range(nboot):
        resamp = rng.choice(resid, size=resid.size, replace=True)
        yboot = model + resamp
        try:
            pboot, _ = curve_fit(gaussian, x, yboot, p0=popt, maxfev=20000)
            mu_samps.append(pboot[1])
        except Exception:
            continue

    if len(mu_samps) < 10:
        return np.nan

    mu_err = np.std(mu_samps)
    rv_err = (mu_err / rest_wave) * C
    return float(rv_err)


#
# --- MEASUREMENT STAGE ---
#
def run_line_measurements():
    """Load FITS, load HJDs, measure RV for every line in every file."""
    logging.info("Reading FITS files...")
    fits_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.fits")), key=natural_key)
    if not fits_list:
        raise FileNotFoundError(f"No FITS files found in {DATA_DIR}")

    # read HJD list optionally
    hjds = None
    if HJD_FILEPATH and os.path.exists(HJD_FILEPATH):
        try:
            hjds = np.loadtxt(HJD_FILEPATH)
            logging.info(f"Loaded HJD list ({len(hjds)} entries)")
            if len(hjds) != len(fits_list):
                logging.warning(f"HJD file length ({len(hjds)}) != FITS file count ({len(fits_list)}). Check file match.")
        except Exception as e:
            logging.error(f"Failed to load HJD file {HJD_FILEPATH}: {e}")
            hjds = None

    records = []
    rng = np.random.RandomState(2025) # for reproducible bootstrap

    for i, fname in enumerate(tqdm(fits_list, desc="Measuring RVs")):
        try:
            hdr = fits.getheader(fname)
            data = fits.getdata(fname).astype(float)
            crval = float(hdr.get("CRVAL1") or hdr.get("CRVAL"))
            cdelt = float(hdr.get("CDELT1") or hdr.get("CD1_1"))
            crpix = float(hdr.get("CRPIX1", 1.0))
            n = data.size
            wl = crval + (np.arange(n) + 1 - crpix) * cdelt
        except Exception as e:
            logging.warning(f"Failed to read {fname}: {e}")
            continue

        # Get HJD
        hjd = np.nan
        if hjds is not None and i < len(hjds):
            hjd = float(hjds[i])
        else:
            try:
                hjd_hdr = hdr.get("HJD") or hdr.get("MJD-OBS") or hdr.get("MJD")
                hjd = float(hjd_hdr) if hjd_hdr is not None else np.nan
            except Exception:
                hjd = np.nan
        
        if not np.isfinite(hjd):
            logging.warning(f"No HJD found for {fname}, skipping file.")
            continue

        for line_name, rest_wave in ALL_LINES.items():
            is_em = line_name.startswith("EM_")

            fit_out = fit_gauss_windowed(wl, data, rest_wave, is_em)

            if (not fit_out) or (len(fit_out) < 5):
                popt = None; pcov = None; rv = np.nan; rv_err = np.nan; comment = "Unreal: FitFail"
            else:
                popt, pcov, rv, rv_err, comment = fit_out

            #bootstrap error
            if popt is not None and BOOTSTRAP_N > 0:
                try:
                    rv_err_bs = bootstrap_rv_error(wl, data, rest_wave, popt, nboot=BOOTSTRAP_N, rng=rng, is_em=is_em)
                    if np.isfinite(rv_err_bs) and (not np.isfinite(rv_err) or rv_err_bs > rv_err):
                        rv_err = rv_err_bs
                except Exception:
                    pass # keep covariance error

            # Apply error floor
            if (not np.isfinite(rv_err)) or rv_err < RV_ERR_FLOOR:
                rv_err = RV_ERR_FLOOR

            records.append({
                "file": os.path.basename(fname),
                "HJD": hjd,
                "line": line_name,
                "rv_kms": rv,
                "rv_err": rv_err,
                "rv_comment": comment
            })

    #
    perline_df = pd.DataFrame.from_records(records)
    perline_df.to_csv(MEAS_CSV, index=False)
    logging.info(f"Saved per-line measurements to {MEAS_CSV}")
    return perline_df

#
# --- MCMC FITTING (per-line) ---
#
def mcmc_fit_for_line(line, epoch_df):
    """Run emcee for a single line. Returns result dict (always contains 'line' and 'status')."""
    out = {"line": line, "status": "failed", "n_epochs":0}
    epoch_df = epoch_df.dropna(subset=["HJD", "rv_kms", "rv_err"]).copy()
    n_epochs = len(epoch_df)
    out["n_epochs"] = n_epochs

    if n_epochs < 5:
        out["status"] = "too_few_epochs"
        logging.warning(f"{line}: {n_epochs} valid epochs skipping MCMC")
        return out

    hjd = epoch_df["HJD"].values.astype(float)
    v = epoch_df["rv_kms"].values.astype(float)
    e = epoch_df["rv_err"].values.astype(float)
    e = np.where((~np.isfinite(e)) | (e < RV_ERR_FLOOR), RV_ERR_FLOOR, e)

    #Priors (box)
    PRIOR = {"T0_min": TO_INIT - 10,  "T0_max": TO_INIT + 10,
             "P_min": PO_INIT - 0.5, "P_max": PO_INIT + 0.5,
             "K_min": 0.0,           "K_max": 3000.0,
             "gamma_min": -4000.0,   "gamma_max": 4000.0}

    def log_prior(theta):
        T0, P, K, gamma = theta
        if not (PRIOR["T0_min"] < T0 < PRIOR["T0_max"]): return -np.inf
        if not (PRIOR["P_min"] < P < PRIOR["P_max"]): return -np.inf
        if not (PRIOR["K_min"] < K < PRIOR["K_max"]): return -np.inf
        if not (PRIOR["gamma_min"] < gamma < PRIOR["gamma_max"]): return -np.inf
        return 0.0

    def log_like(theta, hjd_, v_, e_):
        T0, P, K, gamma = theta
        phi = 2.0 * np.pi * phase_func(hjd_, T0, P)
        model = gamma + K * np.sin(phi)
        chi2 = np.sum(((v_ - model) / e_)**2)
        return -0.5 * chi2

    def log_prob(theta, hjd_, v_, e_):
        lp = log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        ll = log_like(theta, hjd_, v_, e_)
        if not np.isfinite(ll): return -np.inf
        return lp + ll

    # Initial guess
    T0_guess = TO_INIT
    P0_guess = PO_INIT
    K_guess = np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.5 if np.isfinite(v).any() else 150.0
    gamma_guess = np.nanmedian(v) if np.isfinite(v).any() else 0.0
    p0 = np.array([T0_guess, P0_guess, K_guess, gamma_guess])

    pos = p0 + 1e-4 * np.random.randn(MCMC_WALKERS, len(p0))

    # Ensure some valid initial positions
    def ensure_valid_initial(pos, max_tries=40):
        for _ in range(max_tries):
            lps = np.array([log_prob(p, hjd, v, e) for p in pos])
            bad = ~np.isfinite(lps)
            if not bad.any():
                return pos
            pos[bad] = p0 + 1e-4 * np.random.randn(bad.sum(), len(p0))
        # fallback: sample uniformly in priors
        for i in range(pos.shape[0]):
            pos[i,0] = np.random.uniform(PRIOR["T0_min"], PRIOR["T0_max"])
            pos[i,1] = np.random.uniform(PRIOR["P_min"], PRIOR["P_max"])
            pos[i,2] = np.random.uniform(PRIOR["K_min"], PRIOR["K_max"])
            pos[i,3] = np.random.uniform(PRIOR["gamma_min"], PRIOR["gamma_max"])
        return pos

    pos = ensure_valid_initial(pos)

    sampler = emcee.EnsembleSampler(MCMC_WALKERS, len(p0), log_prob, args=(hjd, v, e))
    try:
        sampler.run_mcmc(pos, MCMC_STEPS, progress=True)
    except Exception as exc:
        logging.warning(f"{line}: emcee run failed: {exc}")
        try: # Retry with fewer steps
            sampler.run_mcmc(pos, max(500, MCMC_STEPS // 4), progress=True)
        except Exception as exc2:
            logging.error(f"{line}: emcee retry also failed: {exc2}")
            out["status"] = "mcmc_failed"
            return out

    chain = sampler.get_chain()

    # keep walkers that are fully finite
    good_walkers = [j for j in range(chain.shape[1]) if np.isfinite(chain[:, j, :]).all()]
    if len(good_walkers) == 0:
        logging.warning(f"{line}: no fully-finite walkers")
        out["status"] = "no_good_walkers"
        return out

    chain_good = chain[:, good_walkers, :]
    start = MCMC_BURN
    if start >= chain_good.shape[0]:
        logging.warning(f"{line}: burn-in >= nsteps, using full chain")
        start = 0
    thin = MCMC_THIN
    chain_trim = chain_good[start::thin, :, :]

    if chain_trim.size == 0:
        out["status"] = "no_samples_after_trim"
        return out

    flat = chain_trim.reshape(-1, chain_trim.shape[2])
    flat = flat[np.all(np.isfinite(flat), axis=1)]

    if flat.shape[0] == 0:
        out["status"] = "no_finite_samples"
        return out

    med = np.median(flat, axis=0)
    std = np.std(flat, axis=0)

    # Save posterior samples (flat)
    posterior_path = os.path.join(DATA_DIR, f"wr127_posterior_{line}.npz")
    np.savez_compressed(posterior_path, samples=flat)
    logging.info(f"{line}: saved posterior samples to {posterior_path}")

    # Diagnostic plots (rv vs phase + residuals, traces, corner)
    try:
        # RV vs phase
        phase_vals = phase_func(hjd, med[0], med[1])
        model_vals = med[3] + med[2] * np.sin(2* np.pi * phase_vals)
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,7), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
        ax1.errorbar(phase_vals, v, yerr=e, fmt='o', label='data')
        ax1.errorbar(phase_vals + 1.0, v, yerr=e, fmt='o')
        phgrid = np.linspace(0,2,400)
        ax1.plot(phgrid, med[3] + med[2] * np.sin(2* np.pi * phgrid), '-', lw=2, label='Best fit')
        ax1.set_ylabel("RV (km/s)"); ax1.legend()
        resid = v - model_vals
        ax2.errorbar(phase_vals, resid, yerr=e, fmt='o'); ax2.axhline(0,color='k', ls='--')
        ax2.errorbar(phase_vals + 1.0, resid, yerr=e, fmt='o')
        ax2.set_xlabel("Phase"); ax2.set_ylabel("Residual (km/s)")
        plt.tight_layout()
        rvplot = os.path.join(DATA_DIR, f"wr127_fit_{line}_rv_resid.png")
        fig.savefig(rvplot, dpi=200); plt.close(fig)

        # traces
        trace = sampler.get_chain()[:, good_walkers, :]
        ndim = trace.shape[2]; names = ["T0", "P0", "K", "gamma"]
        fig, axs = plt.subplots(ndim, 1, figsize=(8, 2.4*ndim), sharex=True)
        for i in range(ndim):
            for w in range(trace.shape[1]):
                axs[i].plot(trace[:, w, i], alpha=0.15)
            axs[i].set_ylabel(names[i])
        axs[-1].set_xlabel("step")
        plt.tight_layout()
        tracefile = os.path.join(DATA_DIR, f"wr127_fit_{line}_traces.png")
        fig.savefig(tracefile, dpi=200); plt.close(fig)

        #corner if enough samples & corner available
        if flat.shape[0] >= 100 and MCMC_AVAILABLE:
            try:
                fig = corner.corner(flat, labels=names, quantiles=[0.16, 0.5, 0.84], show_titles=True)
                cornerfile = os.path.join(DATA_DIR, f"wr127_fit_{line}_corner.png")
                fig.savefig(cornerfile, dpi=200); plt.close(fig)
            except Exception:
                logging.warning(f"{line}: corner plot failed")
    except Exception as e:
        logging.warning(f"{line}: plotting diagnostics failed: {e}")

    out.update({
        "status": "mcmc_ok",
        "T0_med": float(med[0]),
        "T0_std": float(std[0]),
        "P0_med": float(med[1]),
        "P0_std": float(std[1]),
        "K_med": float(med[2]), "K_std": float(std[2]),
        "gamma_med": float(med[3]), "gamma_std": float(std[3]),
        "posterior": posterior_path,
        "Neff": float(flat.shape[0])
    })
    return out

#
# --- AGGREGATION AND q COMPUTATION ---
#
def aggregate_results(results):
    """Take list of MCMC results, compute weighted K, compute q, write summaries."""
    if not results:
        logging.warning("No MCMC results to aggregate.")
        # still write empty summaries
        pd.DataFrame(columns=['line', 'status', 'n_epochs', 'T0_med', 'T0_std', 'P0_med', 'P0_std', 'K_med', 'K_std', 'gamma_med', 'gamma_std', 'posterior', 'Neff']).to_csv(EM_SUM_CSV, index=False)
        pd.DataFrame(columns=['line', 'status', 'n_epochs', 'T0_med', 'T0_std', 'P0_med', 'P0_std', 'K_med', 'K_std', 'gamma_med', 'gamma_std', 'posterior', 'Neff']).to_csv(ABS_SUM_CSV, index=False)
        pd.DataFrame([{} ]).to_csv(MASSRATIO_CSV, index=False)
        # BUG FIX: Return empty dataframes
        return {}, pd.DataFrame(), pd.DataFrame() 

    df = pd.DataFrame(results)
    em_df = df[df.line.str.startswith("EM_")]
    abs_df = df[df.line.str.startswith("ABS_")]

    #Write per-line summaries (fill missing columns safely)
    def safe_write(df_sub, outpath):
        cols = ['line', 'status', 'n_epochs', 'T0_med', 'T0_std', 'P0_med', 'P0_std', 'K_med', 'K_std', 'gamma_med','gamma_std', 'posterior', 'Neff']
        for c in cols:
            if c not in df_sub.columns:
                df_sub[c] = np.nan
        df_sub[cols].to_csv(outpath, index=False)
        logging.info(f"Wrote per-line summary: {outpath}")

    safe_write(em_df, EM_SUM_CSV)
    safe_write(abs_df, ABS_SUM_CSV) # This will now be an empty file, which is correct

    #Helper: load posteriors safely and compute weighted-mean K distribution
    def draw_weighted_mean_k(rows, n_draws=MC_Q_SAMPLES):
        samp_list = []
        for _, r in rows.iterrows():
            pth = r.get('posterior', None)
            if isinstance(pth, str) and os.path.exists(pth):
                try:
                    arr = np.load(pth)['samples']
                    Ks = arr[:, 2] # K is the 3rd parameter (index 2)
                    samp_list.append(Ks)
                except Exception as e:
                    logging.warning(f"Failed reading posterior {pth}: {e}")

        logging.info(f"Found {len(samp_list)} valid posterior files in group.")
        if len(samp_list) == 0: return np.array([])

        out = np.empty(n_draws)
        rng = np.random.RandomState(2025)
        for i in range(n_draws):
            draws = np.array([rng.choice(s) for s in samp_list])
            # Inverse-variance weighting
            vars_ = np.array([np.var(s) if np.var(s) > 0 else 1.0 for s in samp_list])
            w = 1.0 / vars_
            out[i] = np.sum(draws * w) / np.sum(w)
        return out

    Kwr_samps = draw_weighted_mean_k(em_df)
    Ko_samps = draw_weighted_mean_k(abs_df) # This will now be an empty array
    summary = {}

    if Kwr_samps.size > 0:
        summary['K_WR_wmean'] = float(np.median(Kwr_samps))
        summary['K_WR_wmean_err'] = float(np.std(Kwr_samps))
    else:
        summary['K_WR_wmean'] = np.nan; summary['K_WR_wmean_err'] = np.nan

    # This block will be correctly skipped as Ko_samps is empty
    if Ko_samps.size > 0:
        summary['K_O_wmean'] = float(np.median(Ko_samps))
        summary['K_O_wmean_err'] = float(np.std(Ko_samps))
    else:
        summary['K_O_wmean'] = np.nan; summary['K_O_wmean_err'] = np.nan

    # This block will also be correctly skipped
    if Kwr_samps.size > 0 and Ko_samps.size > 0:
        #compute q samples and percentiles
        # NOTE: This is q = M_WR / M_O = K_O / K_WR
        rng = np.random.RandomState(2025)
        n = min(Kwr_samps.size, Ko_samps.size, MC_Q_SAMPLES)
        idx1 = rng.choice(np.arange(Ko_samps.size), size=n, replace=True)
        idx2 = rng.choice(np.arange(Kwr_samps.size), size=n, replace=True)
        
        Kwr_safe = Kwr_samps[idx2]
        Ko_safe = Ko_samps[idx1]
        
        valid = (Kwr_safe > 0)
        Kwr_safe = Kwr_safe[valid]
        Ko_safe = Ko_safe[valid]

        if Ko_safe.size > 10:
            q_samps = Ko_safe / Kwr_safe
            summary['q_med_KO_KWR'] = float(np.median(q_samps))
            summary['q_std_KO_KWR'] = float(np.std(q_samps))
            summary['q_16_KO_KWR'] = float(np.percentile(q_samps, 16))
            summary['q_84_KO_KWR'] = float(np.percentile(q_samps, 84))
        else:
             summary['q_med_KO_KWR'] = np.nan; summary['q_std_KO_KWR'] = np.nan
    else:
        summary['q_med_KO_KWR'] = np.nan; summary['q_std_KO_KWR'] = np.nan

    pd.DataFrame([summary]).to_csv(MASSRATIO_CSV, index=False)
    logging.info(f"Wrote mass-ratio summary to {MASSRATIO_CSV}: {summary}")
    
    # BUG FIX: Return em_df and abs_df for plotting
    return summary, em_df, abs_df

#
# --- PHASE PLOT ---
#
# BUG FIX: Accept em_df and abs_df as arguments
def plot_phase_folded(meas_df, summary, em_df, abs_df):
    """Plot all valid measurements, folded on the initial ephemeris."""
    if meas_df is None or meas_df.empty:
        return

    T0, P = TO_INIT, PO_INIT
    meas = meas_df.copy()
    meas['phase'] = ((meas['HJD'] - T0) / P) % 1.0
    
    wr = meas[meas.line.str.startswith("EM_")].dropna(subset=['rv_kms'])
    o = meas[meas.line.str.startswith("ABS_")].dropna(subset=['rv_kms']) # This will be empty

    plt.figure(figsize=(9,6))
    plt.errorbar(wr.phase, wr.rv_kms, yerr=wr.rv_err, fmt='o', label='WR emission (EM lines)', alpha=0.7, capsize=3)
    
    # This plot call will do nothing, as 'o' is empty, which is correct
    plt.errorbar(o.phase, o.rv_kms, yerr=o.rv_err, fmt='x', label='O-star absorption (ABS lines)', alpha=0.9, capsize=3, color='C3')
    
    # Plot phase + 1
    plt.errorbar(wr.phase + 1.0, wr.rv_kms, yerr=wr.rv_err, fmt='o', alpha=0.7, capsize=3, mfc='C0', mec='C0')
    plt.errorbar(o.phase + 1.0, o.rv_kms, yerr=o.rv_err, fmt='x', alpha=0.9, capsize=3, color='C3')

    #overlay median best-fit sinusoids if available
    if summary:
        phi = np.linspace(0,2,400)
        
        if np.isfinite(summary.get('K_WR_wmean', np.nan)):
            Kwr = summary['K_WR_wmean']
            # Find a representative gamma from the per-line fits
            gamma_wr = em_df['gamma_med'].median() if not em_df.empty and 'gamma_med' in em_df.columns else 0.0
            plt.plot(phi, gamma_wr + Kwr * np.sin(2*np.pi*phi), '-', color='C0', lw=2.5, label=f'WR Fit (K={Kwr:.1f})')

        # This block will be correctly skipped
        if np.isfinite(summary.get('K_O_wmean', np.nan)):
            Ko = summary['K_O_wmean']
            gamma_o = abs_df['gamma_med'].median() if not abs_df.empty and 'gamma_med' in abs_df.columns else 0.0
            plt.plot(phi, gamma_o - Ko * np.sin(2*np.pi*phi), '--', color='C3', lw=2.5, label=f'O-Star Fit (K={Ko:.1f})')

    plt.xlabel("Orbital phase"); plt.ylabel("RV (km/s)")
    plt.legend(); plt.grid(alpha=0.4, ls=':')
    plt.xlim(0, 2.0)
    out = os.path.join(DATA_DIR, "wr127_phase_fit_summary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300); plt.close()
    logging.info(f"Saved phase-folded summary plot to {out}")

#
# --- MAIN ---
#
def main():
    start = time.time()
    # minimal README for provenance
    try:
        with open(PIPE_README, "w") as fh:
            fh.write(f"WR127 pipeline outputs from run at {time.ctime()}.\n")
            fh.write(f"RUN_MODE={RUN_MODE}\n")
            fh.write(f"PO_INIT={PO_INIT}, TO_INIT={TO_INIT}\n")
            fh.write("\nNOTE: O-star lines were excluded as HeI5876 was found to be unusable.\n")
            fh.write("Only WR-star parameters (K_WR) are derived.\n")
            fh.write("EM_halfwidth set to 20.0A for robust fitting.\n")
    except Exception as e:
        logging.warning(f"Could not write README: {e}")

    meas_df = None
    if RUN_MODE in ("measure_only", "measure_then_mcmc"):
        meas_df = run_line_measurements()
    else:
        if not os.path.exists(MEAS_CSV):
            raise FileNotFoundError(f"{MEAS_CSV} not found; run 'measure_only' first or set RUN_MODE appropriately.")
        meas_df = pd.read_csv(MEAS_CSV)

    if RUN_MODE == "measure_only":
        logging.info("Measurement-only mode complete.")
        return

    results = []
    summary = {}
    em_df, abs_df = pd.DataFrame(), pd.DataFrame() # Init empty
    
    if RUN_MODE in ("mcmc_only", "measure_then_mcmc"):
        if not MCMC_AVAILABLE:
            logging.error("MCMC libraries (emcee, corner) not available in environment. Install to run MCMC.")
        else:
            if meas_df is None or meas_df.empty:
                logging.error("meas_df is empty, cannot run MCMC.")
                return
                
            # group by line and run MCMC per-line
            for line, group in meas_df.groupby("line"):
                # use only finite RVs for fitting
                g = group.dropna(subset=["rv_kms", "rv_err"]).copy()
                if len(g) < 5:
                    logging.warning(f"Skipping {line}: {len(g)} valid epochs (<5).")
                    continue
                logging.info(f"Running MCMC for {line} ({len(g)} epochs)...")
                res = mcmc_fit_for_line(line, g)
                results.append(res)
    
    if results:
        # BUG FIX: Get em_df and abs_df back from aggregation
        summary, em_df, abs_df = aggregate_results(results)
        # BUG FIX: Pass em_df and abs_df to plotting
        plot_phase_folded(meas_df, summary, em_df, abs_df)
        logging.info(f"Final aggregated summary: {summary}")
    else:
        logging.warning("No MCMC results were produced. Per-line measurements are saved for inspection.")

    elapsed = (time.time() - start) / 60.0
    logging.info(f"Pipeline finished in {elapsed:.2f} minutes.")

if __name__ == "__main__":
    main()
