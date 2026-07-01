import os
from pathlib import Path
import numpy as np
import pandas as pd
import cdflib
from src.config import RAW_DIR

def datetime_index_to_epoch(index):
    """Convert a pandas DatetimeIndex directly to CDF epoch format."""
    components = np.stack([
        index.year, index.month, index.day,
        index.hour, index.minute, index.second,
        (index.microsecond / 1000).astype(int)
    ], axis=1)
    return cdflib.cdfepoch.compute_epoch(components.tolist())

def generate_synthetic_cdf_data(start_year=2016, end_year=2017):
    """
    Generates physically realistic synthetic CDF files for GOES and Wind
    to serve as a local test/fallback if downloading fails.
    """
    goes_dir = RAW_DIR / "goes"
    swe_dir = RAW_DIR / "wind_swe"
    mfi_dir = RAW_DIR / "wind_mfi"
    
    goes_dir.mkdir(parents=True, exist_ok=True)
    swe_dir.mkdir(parents=True, exist_ok=True)
    mfi_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Generating Synthetic CDF Data ---")
    
    for year in range(start_year, end_year + 1):
        # 1-minute cadence for GOES and MFI, 90-second for SWE
        dates_1min = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:59:00", freq="1min")
        dates_swe = pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:59:00", freq="90s")
        
        n_1min = len(dates_1min)
        n_swe = len(dates_swe)
        
        # Seed for reproducibility
        np.random.seed(42 + year)
        
        # ------------------ Wind SWE (Plasma) ------------------
        # Solar wind speed modeled as an AR(1) process with random high-speed streams
        v_sw = np.zeros(n_swe)
        current_v = 400.0
        for i in range(n_swe):
            # 0.5% chance of starting a storm (high speed stream)
            if np.random.rand() < 0.0001:
                current_v = np.random.uniform(600.0, 900.0)
            else:
                # Decays back to baseline (400 km/s)
                current_v = 0.999 * current_v + 0.001 * 400.0 + np.random.randn() * 1.5
            v_sw[i] = current_v
            
        # Density (n_sw) is anti-correlated with speed, with sudden compression spikes
        n_sw = 5.0 + 10.0 * (800.0 - v_sw) / 400.0 + np.random.exponential(scale=2.0, size=n_swe)
        n_sw = np.clip(n_sw, 0.1, 80.0)
        
        # ------------------ Wind MFI (Magnetic Field) ------------------
        # IMF Bz fluctuates, with large negative (southward) excursions during high speed streams
        # Interpolated speed to 1-minute grid to couple it
        v_sw_1min = np.interp(dates_1min.asi8, dates_swe.asi8, v_sw)
        bz = np.random.randn(n_1min) * 3.0
        # south Bz excursions align with high wind speed
        storm_mask = v_sw_1min > 550.0
        bz[storm_mask] -= np.random.exponential(scale=6.0, size=storm_mask.sum())
        
        # ------------------ GOES-15 Electron Flux ------------------
        # Electron flux couples physically:
        # - Lags wind speed by ~36 hours (1.5 days = 1440 mins)
        # - Crashes immediately if dynamic pressure Pdyn = density * speed^2 is high (dropout)
        # - Daily cycle based on hour of day
        
        # Interpolate SWE parameters to 1-minute grid
        n_sw_1min = np.interp(dates_1min.asi8, dates_swe.asi8, n_sw)
        pdyn_1min = 1.6726e-6 * n_sw_1min * (v_sw_1min ** 2)
        
        # Base flux AR(1) driven by lagged speed
        lag_mins = 2160 # 36h lag
        v_sw_lagged = np.roll(v_sw_1min, lag_mins)
        # Fill rolled start with baseline
        v_sw_lagged[:lag_mins] = 400.0
        
        log_flux = np.zeros(n_1min)
        current_flux = 2.0
        
        for i in range(n_1min):
            # Speed driver enhancement (strengthened coupling for realistic variations)
            flux_target = 2.0 + 4.0 * (v_sw_lagged[i] - 300.0) / 500.0
            current_flux = 0.995 * current_flux + 0.005 * flux_target + np.random.randn() * 0.05
            
            # Dynamic pressure dropout effect (threshold raised to 8.0 nPa so it only triggers during storm compressions)
            if pdyn_1min[i] > 8.0:
                # sharp crash
                current_flux -= (pdyn_1min[i] - 8.0) * 0.2
                
            log_flux[i] = current_flux
            
        # Add hour-of-day daily cycle (local time effect at GEO)
        hod = dates_1min.hour + dates_1min.minute / 60.0
        log_flux += 0.3 * np.sin(2 * np.pi * hod / 24.0)
        
        # Clip flux to realistic range [0.5, 6.5]
        log_flux = np.clip(log_flux, 0.5, 6.5)
        # Convert back to physical flux pfu
        e_flux = 10 ** log_flux
        
        # Proton flux (for contamination simulation) - spikes during solar proton events
        p_flux = np.random.exponential(scale=0.1, size=n_1min)
        # Introduce a few proton events
        proton_event_starts = np.random.choice(n_1min, size=3, replace=False)
        for start in proton_event_starts:
            duration = np.random.randint(60, 240) # 1 to 4 hours
            p_flux[start:start+duration] = np.random.uniform(50.0, 500.0)
            
        # ------------------ Write to CDF Files ------------------
        # Mocking GOES-15
        goes_path = goes_dir / f"goes_{year}.cdf"
        if goes_path.exists(): goes_path.unlink()
        cdf_goes = cdflib.cdfwrite.CDF(str(goes_path))
        
        epoch_1min = datetime_index_to_epoch(dates_1min)
        cdf_goes.write_globalattrs({"Project": {0: "ISRO Synthetic"}, "Source": {0: "GOES-15"}})
        cdf_goes.write_var({"Variable": "Epoch", "Data_Type": cdflib.cdfwrite.CDF.CDF_EPOCH, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []}, var_data=epoch_1min)
        
        # Write e_flux
        cdf_goes.write_var(
            {"Variable": "e_flux", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": 0.0, "VALIDMAX": 1.0e9, "UNITS": "pfu"},
            var_data=e_flux
        )
        
        # Write p_flux
        cdf_goes.write_var(
            {"Variable": "p_flux", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": 0.0, "VALIDMAX": 1.0e9, "UNITS": "pfu"},
            var_data=p_flux
        )
        cdf_goes.close()
        print(f"Generated synthetic GOES CDF: {goes_path}")
        
        # Mocking Wind SWE
        swe_path = swe_dir / f"swe_{year}.cdf"
        if swe_path.exists(): swe_path.unlink()
        cdf_swe = cdflib.cdfwrite.CDF(str(swe_path))
        
        epoch_swe = datetime_index_to_epoch(dates_swe)
        cdf_swe.write_globalattrs({"Project": {0: "ISRO Synthetic"}, "Source": {0: "Wind"}})
        cdf_swe.write_var({"Variable": "Epoch", "Data_Type": cdflib.cdfwrite.CDF.CDF_EPOCH, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []}, var_data=epoch_swe)
        
        # Write v_sw
        cdf_swe.write_var(
            {"Variable": "v_sw", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": 0.0, "VALIDMAX": 2000.0, "UNITS": "km/s"},
            var_data=v_sw
        )
        
        # Write n_sw
        cdf_swe.write_var(
            {"Variable": "n_sw", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": 0.0, "VALIDMAX": 500.0, "UNITS": "cm^-3"},
            var_data=n_sw
        )
        cdf_swe.close()
        print(f"Generated synthetic Wind SWE CDF: {swe_path}")
        
        # Mocking Wind MFI
        mfi_path = mfi_dir / f"mfi_{year}.cdf"
        if mfi_path.exists(): mfi_path.unlink()
        cdf_mfi = cdflib.cdfwrite.CDF(str(mfi_path))
        
        cdf_mfi.write_globalattrs({"Project": {0: "ISRO Synthetic"}, "Source": {0: "Wind"}})
        cdf_mfi.write_var({"Variable": "Epoch", "Data_Type": cdflib.cdfwrite.CDF.CDF_EPOCH, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []}, var_data=epoch_1min)
        
        # Write bz
        cdf_mfi.write_var(
            {"Variable": "bz", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": -500.0, "VALIDMAX": 500.0, "UNITS": "nT"},
            var_data=bz
        )
        cdf_mfi.close()
        print(f"Generated synthetic Wind MFI CDF: {mfi_path}")

def generate_synthetic_grasp_data(start_year=2016, end_year=2018):
    """
    Generates synthetic GRASP/GSAT electron flux data for geostationary orbit at Indian longitude.
    Models:
    - Local time phase shift (10 hours shift = 600 minutes roll at 1-min cadence)
    - Systematic attenuation (scaled by 0.9)
    - Systematic offset (subtract 0.2 log pfu)
    - Local longitude magnetic environment noise (std 0.08)
    """
    grasp_dir = RAW_DIR / "grasp"
    grasp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Generating Synthetic GRASP/GSAT CDF Data ---")
    
    for year in range(start_year, end_year + 1):
        goes_path = RAW_DIR / "goes" / f"goes_{year}.cdf"
        if not goes_path.exists():
            print(f"Warning: GOES CDF for {year} not found. Cannot generate aligned GRASP data.")
            continue
            
        cdf_goes = cdflib.CDF(str(goes_path))
        epoch = cdf_goes.varget("Epoch")
        e_flux_goes = cdf_goes.varget("e_flux")
        
        # log10 flux from GOES
        log_flux_goes = np.log10(np.clip(e_flux_goes, 1.0, None))
        
        # Apply 10 hours shift (600 minutes)
        roll_steps = 600
        log_flux_shifted = np.roll(log_flux_goes, roll_steps)
        log_flux_shifted[:roll_steps] = log_flux_shifted[roll_steps] # Pad start of rolled array
        
        # Apply systematic scaling, offset, and local noise
        np.random.seed(99 + year)
        noise = np.random.normal(0, 0.08, len(log_flux_goes))
        log_flux_grasp = 0.9 * log_flux_shifted - 0.2 + noise
        log_flux_grasp = np.clip(log_flux_grasp, 0.5, 6.5)
        
        e_flux_grasp = 10 ** log_flux_grasp
        
        grasp_path = grasp_dir / f"grasp_{year}.cdf"
        if grasp_path.exists():
            grasp_path.unlink()
            
        cdf_grasp = cdflib.cdfwrite.CDF(str(grasp_path))
        cdf_grasp.write_globalattrs({"Project": {0: "ISRO Synthetic"}, "Source": {0: "GSAT GRASP"}})
        cdf_grasp.write_var(
            {"Variable": "Epoch", "Data_Type": cdflib.cdfwrite.CDF.CDF_EPOCH, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_data=epoch
        )
        cdf_grasp.write_var(
            {"Variable": "e_flux", "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE, "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
            var_attrs={"FILLVAL": -1.0e31, "VALIDMIN": 0.0, "VALIDMAX": 1.0e9, "UNITS": "pfu"},
            var_data=e_flux_grasp
        )
        cdf_grasp.close()
        print(f"Generated synthetic GRASP CDF: {grasp_path}")

if __name__ == "__main__":
    generate_synthetic_cdf_data(2016, 2018)
    generate_synthetic_grasp_data(2016, 2018)

