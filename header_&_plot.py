from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from specutils import Spectrum1D
from specutils.manipulation import extract_region

# --- Replace with the actual path to your FITS file ---
fits_file_path = '/home/swastik/Desktop/WR127distng/Disentangling_Shift_And_Add/dta/WR127_0.fits' # If in the same directory as notebook
# fits_file_path = '/path/to/your/spectral/data/spectrum.fits' # Or full path

spectrum = None # Initialise spectrum to None, will be set if loaded successfully

try:
    # Open the FITS file
    with fits.open(fits_file_path) as hdul:
        print(f"Opened FITS file: {fits_file_path}")
        print("\n--- FITS HDU List Info ---")
        hdul.info()

        print("\nAttempting to load spectrum from primary HDU (hdul[0]) using WCS...")
        try:
            # Try to load using Spectrum1D.read, which handles FITS WCS
            spectrum = Spectrum1D.read(fits_file_path, format='wcs1d-fits')
            print("Spectrum loaded successfully from primary HDU using WCS.")

        except Exception as e_wcs:
            print(f"Failed to load as WCS1D-FITS: {e_wcs}")
            print("\nAttempting to find spectrum in a table HDU...")

            # Fallback: Look for data in a table HDU (e.g., HDU 1)
            # You might need to inspect hdul.info() carefully to find the right HDU
            # and column names (e.g., 'FLUX', 'WAVELENGTH', 'WAVE', 'SPEC').
            try:
                # Assuming the spectrum is in the second HDU (index 1) as a table
                table_hdu = hdul[1] # <--- You might need to change this index (e.g., hdul[2])

                wave_col_name = None
                flux_col_name = None

                for col_name in ['WAVELENGTH', 'WAVE', 'LAMBDA', 'CRVAL1', 'AXIS_1']:
                    if col_name in table_hdu.columns.names:
                        wave_col_name = col_name
                        break
                for col_name in ['FLUX', 'SPEC', 'DATA']:
                    if col_name in table_hdu.columns.names:
                        flux_col_name = col_name
                        break

                if wave_col_name and flux_col_name:
                    wavelength_values = table_hdu.data[wave_col_name]
                    flux_values = table_hdu.data[flux_col_name]

                    # Assign units. You MUST verify these are correct for your file!
                    # Example common units:
                    # Wavelength: u.Angstrom, u.nm, u.AA, u.micron, u.Hz
                    # Flux: u.erg/u.s/u.cm**2/u.Angstrom, u.Jy, u.W/u.m**2/u.micron, u.count/u.s
                    wavelength = wavelength_values * u.Angstrom # <--- CHECK AND ADJUST THIS UNIT
                    flux = flux_values * u.electron / u.s     # <--- CHECK AND ADJUST THIS UNIT

                    spectrum = Spectrum1D(flux=flux, spectral_axis=wavelength)
                    print(f"Spectrum loaded successfully from Table HDU ({table_hdu.name}).")
                else:
                    raise ValueError("Could not find suitable 'WAVELENGTH' and 'FLUX' columns in table HDU.")

            except Exception as e_table:
                print(f"Failed to load from table HDU (index 1): {e_table}")
                print("\nCould not automatically determine spectrum structure.")
                print("Manual inspection of FITS structure (hdul.info(), hdul[X].header, hdul[X].columns.names) is needed.")

# Main exception handlers for the entire process (FileNotFoundError, general Exception)
except FileNotFoundError:
    print(f"Error: FITS file not found at '{fits_file_path}'. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred during FITS file processing: {e}")

# --- If spectrum was successfully loaded, proceed with plotting ---
if spectrum is not None:
    print(f"\n--- Spectrum Loaded ---")
    print(f"Spectral Axis: {spectrum.spectral_axis.shape} {spectrum.spectral_axis.unit}")
    print(f"Flux: {spectrum.flux.shape} {spectrum.flux.unit}")

    # --- Plot the Spectrum ---
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum.spectral_axis, spectrum.flux, drawstyle='steps-mid')
    plt.xlabel(f"Wavelength ({spectrum.spectral_axis.unit})")
    plt.ylabel(f"Flux ({spectrum.flux.unit})")
    plt.title(f"Spectrum from {fits_file_path}")
    plt.grid(True)
    plt.minorticks_on()
    plt.tick_params(which='minor', length=4, color='gray', axis='both')
    plt.tight_layout()
    plt.show()

    # --- Accessing Header Information from Spectrum1D ---
    # If loaded with Spectrum1D.read(), header info might be available via meta
    if hasattr(spectrum, 'meta') and 'header' in spectrum.meta:
        print("\n--- Spectrum Header (from Spectrum1D.meta) ---")
        print(repr(spectrum.meta['header']))
    else:
        # If loaded manually from the table, you can access the original HDU header
        print("\n--- Original FITS Header (Primary HDU) ---")
        with fits.open(fits_file_path) as hdul_header_only: # Re-open just for header access
            print(repr(hdul_header_only[0].header))
else:
    print("\nSpectrum object was not created. Cannot proceed with plotting or further analysis.")
