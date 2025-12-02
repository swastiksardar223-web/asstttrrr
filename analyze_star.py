import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# ==========================================
#           CONFIGURATION AREA
# ==========================================
# UNCOMMENT the star you want to analyze tomorrow:

# --- STAR 1: WR 127 ---
STAR_CONFIG = {
    'name': 'WR127',
    'data_dir': 'dta',           # Folder with FITS files
    'template_file': 'Output/ADIS*.txt', # Disentangled Template
    'period': 9.55465,
    'lit_T0': 2460187.81,        # Shaposhnikov 2024
    'w_min': 5350, 'w_max': 5500, # He II 5411 region
    'K2_lit': 125.0,             # Fixed K2 for mass calc
    'gamma_guess': 1270.0        # Approximate gamma offset
}

# --- STAR 2: WR 139 (Uncomment to use) ---
# STAR_CONFIG = {
#     'name': 'WR139',
#     'data_dir': 'dta_WR139',
#     'template_file': 'Output/ADIS*.txt',
#     'period': 4.212424,
#     'lit_T0': 2441164.332,       # Shaposhnikov 2023
#     'w_min': 5350, 'w_max': 5500,
#     'K2_lit': 130.0,
#     'gamma_guess': 35.0
# }

# ==========================================
#           MAIN PIPELINE
# ==========================================

def run_pipeline(cfg):
    print(f"🚀 STARTING ANALYSIS FOR {cfg['name']}...")
    
    # --- 1. LOAD TEMPLATE ---
    try:
        t_files = glob.glob(cfg['template_file'])
        if not t_files: raise Exception("No ADIS file found! Run disentangling first.")
        temp = np.loadtxt(t_files[0])
        wave_t, flux_t = temp[:,0], temp[:,1]
        mask_t = (wave_t >= cfg['w_min']) & (wave_t <= cfg['w_max'])
        wave_t, flux_t = wave_t[mask_t], flux_t[mask_t]
        print("✅ Template Loaded.")
    except Exception as e:
        print(f"❌ Template Error: {e}"); return

    # --- 2. MEASURE RVs ---
    raw_files = sorted(glob.glob(f"{cfg['data_dir']}/*.fits"))
    print(f"measuring {len(raw_files)} spectra...")
    
    rv_results = []
    c_light = 299792.458
    
    for f in raw_files:
        try:
            with fits.open(f) as hdul:
                h = hdul[0].header
                d = hdul[0].data.astype(float)
                # WCS Construction
                w = h['CRVAL1'] + (np.arange(1, h['NAXIS1']+1) - h['CRPIX1']) * h.get('CDELT1', h.get('CD1_1'))
                mjd = h['MJD-OBS']
                
                # Cut and Interpolate
                mask = (w >= cfg['w_min']) & (w <= cfg['w_max'])
                w_obs, f_obs = w[mask], d[mask]
                f_interp = interp1d(w_obs, f_obs, kind='linear', fill_value="extrapolate")(wave_t)
                
                # CCF
                ccf = np.correlate(f_interp - np.mean(f_interp), flux_t - np.mean(flux_t), mode='same')
                peak = np.argmax(ccf)
                # Shift Calculation
                mid = len(wave_t)//2
                disp = (wave_t[-1]-wave_t[0])/len(wave_t)
                rv = -1 * ((peak - mid) * disp / wave_t[mid]) * c_light
                rv_results.append([mjd, rv])
        except: pass
    
    rv_data = np.array(rv_results)
    time, rv = rv_data[:,0], rv_data[:,1]
    
    # --- 3. SOLVE ORBIT (Circular) ---
    def model(t, K, gamma, T0):
        return gamma + K * np.sin(2*np.pi*(t-T0)/cfg['period'])
    
    p0 = [200.0, cfg['gamma_guess'], time[0]]
    params, cov = curve_fit(model, time, rv, p0=p0)
    K_fit, g_fit, T0_fit = params
    K_err = np.sqrt(np.diag(cov))[0]
    
    # Residuals
    res = rv - model(time, *params)
    rms = np.std(res)
    
    # Masses
    const = 1.0361e-7
    M_WR = const * (K_fit + cfg['K2_lit'])**2 * cfg['K2_lit'] * cfg['period']
    M_O  = const * (K_fit + cfg['K2_lit'])**2 * K_fit * cfg['period']
    
    # --- 4. O-C ANALYSIS ---
    T0_hjd = T0_fit + 2400000.5
    cycles = (cfg['lit_T0'] - T0_hjd) / cfg['period']
    E = round(cycles*2)/2 # Allow 0.5 shift
    T0_calc = T0_hjd + E*cfg['period']
    OC = cfg['lit_T0'] - T0_calc
    
    # --- REPORT ---
    print("\n" + "="*40)
    print(f"   FINAL RESULTS FOR {cfg['name']}")
    print("="*40)
    print(f"K1 (WR Velocity):  {K_fit:.2f} +/- {K_err:.2f} km/s")
    print(f"Gamma:             {g_fit:.2f} km/s")
    print(f"RMS Scatter:       {rms:.2f} km/s")
    print("-" * 40)
    print(f"M_WR (sin^3 i):    {M_WR:.2f} M_sun")
    print(f"M_O  (sin^3 i):    {M_O:.2f} M_sun")
    print("-" * 40)
    print(f"O-C Residual:      {OC:.4f} days")
    print("="*40)

# --- EXECUTE ---
run_pipeline(STAR_CONFIG)
