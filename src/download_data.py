import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from cdasws import CdasWs
import cdflib
from src.config import RAW_DIR

def write_df_to_cdf(df, path, var_specs):
    """
    Write a pandas DataFrame to a CDF file using cdflib.
    Args:
        df: pd.DataFrame with sorted DatetimeIndex
        path: Path or str, destination path
        var_specs: dict of column name -> dict with cdf metadata:
            {
                "col_name": {
                    "var_name": "CDF_var_name",
                    "FILLVAL": val,
                    "VALIDMIN": val,
                    "VALIDMAX": val
                }
            }
    """
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize CDF file
    path_str = str(path)
    if os.path.exists(path_str):
        os.remove(path_str)

    cdf = cdflib.cdfwrite.CDF(path_str)

    # 1. Write Global Attributes
    global_attrs = {
        "Project": {0: "ISRO Space Weather Electron Flux Forecasting"},
        "Description": {0: f"Downloaded real scientific data saved in CDF format."}
    }
    cdf.write_globalattrs(global_attrs)

    # 2. Write Epoch variable (standard CDF_EPOCH: milliseconds since Year 0)
    # Convert DatetimeIndex to date/time components (required by cdflib.cdfepoch.compute_epoch)
    components = np.stack([
        df.index.year, df.index.month, df.index.day,
        df.index.hour, df.index.minute, df.index.second,
        (df.index.microsecond / 1000).astype(int)
    ], axis=1)
    epoch_data = cdflib.cdfepoch.compute_epoch(components.tolist())

    epoch_spec = {
        "Variable": "Epoch",
        "Data_Type": cdflib.cdfwrite.CDF.CDF_EPOCH,
        "Num_Elements": 1,
        "Rec_Vary": True,
        "Dim_Sizes": []
    }
    cdf.write_var(epoch_spec, var_data=epoch_data)

    # 3. Write each data variable
    for col_name, spec in var_specs.items():
        var_name = spec["var_name"]
        data = df[col_name].values

        # Prepare variable attributes
        attrs = {}
        if "FILLVAL" in spec:
            attrs["FILLVAL"] = spec["FILLVAL"]
        if "VALIDMIN" in spec:
            attrs["VALIDMIN"] = spec["VALIDMIN"]
        if "VALIDMAX" in spec:
            attrs["VALIDMAX"] = spec["VALIDMAX"]
        attrs["UNITS"] = spec.get("UNITS", "Dimensionless")

        # Prepare variable specifications
        var_spec = {
            "Variable": var_name,
            "Data_Type": cdflib.cdfwrite.CDF.CDF_DOUBLE,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": []
        }

        # Write variable data and attributes
        cdf.write_var(var_spec, var_attrs=attrs, var_data=data)

    cdf.close()
    print(f"Successfully wrote CDF file: {path}")

def download_dataset(cdas, dataset_id, variables, start_time, end_time, retries=3):
    """Download data from CDAWeb with retries."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Retrieving {dataset_id} variables {variables} from {start_time} to {end_time} (Attempt {attempt})...")
            status, data = cdas.get_data(dataset_id, variables, start_time, end_time)
            if status == 200 or data is not None:
                return data
            else:
                print(f"Error: CDAWeb returned status {status} on attempt {attempt}")
        except Exception as e:
            print(f"Exception on attempt {attempt}: {e}")
        time.sleep(5)
    raise RuntimeError(f"Failed to download dataset {dataset_id} after {retries} attempts.")

def validate_cdf_file(filepath):
    """
    Validates that a CDF file exists, is not empty, and can be read by cdflib.
    """
    path = Path(filepath)
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        cdf = cdflib.CDF(str(path))
        info = cdf.cdf_info()
        # Verify it has some variables
        if not info.zVariables and not info.rVariables:
            return False
        return True
    except Exception:
        return False

def download_and_save_real_data(start_year=2013, end_year=2016):
    """
    Downloads Wind and GOES data from CDAWeb in monthly chunks
    and saves them as local CDF files. Each source (GOES, SWE, MFI) is
    checked independently — only missing sources are downloaded, so a
    missing MFI file doesn't trigger re-downloading GOES and SWE that
    are already cached on disk.
    """
    cdas = CdasWs()

    # Create output directories
    goes_dir = RAW_DIR / "goes"
    swe_dir = RAW_DIR / "wind_swe"
    mfi_dir = RAW_DIR / "wind_mfi"

    goes_dir.mkdir(parents=True, exist_ok=True)
    swe_dir.mkdir(parents=True, exist_ok=True)
    mfi_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        goes_cdf = goes_dir / f"goes_{year}.cdf"
        swe_cdf = swe_dir / f"swe_{year}.cdf"
        mfi_cdf = mfi_dir / f"mfi_{year}.cdf"

        # Check each source independently — only download what's missing.
        need_goes = not validate_cdf_file(goes_cdf)
        need_swe = not validate_cdf_file(swe_cdf)
        need_mfi = not validate_cdf_file(mfi_cdf)

        if not need_goes and not need_swe and not need_mfi:
            print(f"Year {year} data already exists and is validated. Skipping download.")
            continue

        needs = []
        if need_goes: needs.append("GOES")
        if need_swe:  needs.append("Wind-SWE")
        if need_mfi:  needs.append("Wind-MFI")
        print(f"\n==========================================")
        print(f"Downloading {', '.join(needs)} for Year {year} ...")
        print(f"==========================================")

        goes_dfs = []
        swe_dfs = []
        mfi_dfs = []

        for month in range(1, 13):
            start_time = f"{year}-{month:02d}-01T00:00:00Z"
            if month == 12:
                end_time = f"{year}-12-31T23:59:59Z"
            else:
                end_time = f"{year}-{month+1:02d}-01T00:00:00Z"

            print(f"\n--- Month {year}-{month:02d} ---")

            # 1. GOES-15 Electron Flux (only if missing)
            if need_goes:
                try:
                    goes_ds = "GOES15_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN"
                    goes_vars = ["E2W_COR_FLUX", "E2W_DQF"]
                    goes_raw = download_dataset(cdas, goes_ds, goes_vars, start_time, end_time)
                    if goes_raw is not None and "E2W_COR_FLUX" in goes_raw:
                        df_goes = pd.DataFrame({
                            "e_flux": goes_raw["E2W_COR_FLUX"].values,
                            "p_flux": goes_raw["E2W_DQF"].values
                        }, index=pd.to_datetime(goes_raw["Epoch"].values))
                        df_goes = df_goes[df_goes.index.notna()]
                        goes_dfs.append(df_goes)
                except Exception as e:
                    print(f"Error downloading GOES for {year}-{month:02d}: {e}")

            # 2. Wind SWE Plasma (only if missing)
            if need_swe:
                try:
                    swe_ds = "WI_H1_SWE"
                    swe_vars = ["Proton_V_nonlin", "Proton_Np_nonlin"]
                    swe_raw = download_dataset(cdas, swe_ds, swe_vars, start_time, end_time)
                    if swe_raw is not None and "Proton_V_nonlin" in swe_raw:
                        df_swe = pd.DataFrame({
                            "v_sw": swe_raw["Proton_V_nonlin"].values,
                            "n_sw": swe_raw["Proton_Np_nonlin"].values
                        }, index=pd.to_datetime(swe_raw["Epoch"].values))
                        df_swe = df_swe[df_swe.index.notna()]
                        swe_dfs.append(df_swe)
                except Exception as e:
                    print(f"Error downloading Wind SWE for {year}-{month:02d}: {e}")

            # 3. Wind MFI Magnetic Field (only if missing)
            if need_mfi:
                try:
                    mfi_ds = "WI_H0_MFI"
                    mfi_vars = ["BGSM"]
                    mfi_raw = download_dataset(cdas, mfi_ds, mfi_vars, start_time, end_time)
                    if mfi_raw is not None and "BGSM" in mfi_raw:
                        bgsm_data = mfi_raw["BGSM"].values
                        bz_data = bgsm_data[:, 2]  # Extract GSM Bz component (index 2)
                        df_mfi = pd.DataFrame({
                            "bz": bz_data
                        }, index=pd.to_datetime(mfi_raw["Epoch"].values))
                        df_mfi = df_mfi[df_mfi.index.notna()]
                        mfi_dfs.append(df_mfi)
                except Exception as e:
                    print(f"Error downloading Wind MFI for {year}-{month:02d}: {e}")

        # Concatenate and write yearly CDF files — only for sources that
        # were actually downloaded this pass.
        if need_goes and goes_dfs:
            df_goes_year = pd.concat(goes_dfs).sort_index()
            goes_specs = {
                "e_flux": {"var_name": "e_flux", "FILLVAL": -1e31, "VALIDMIN": 0.0, "VALIDMAX": 1e9, "UNITS": "pfu"},
                "p_flux": {"var_name": "p_flux", "FILLVAL": -1.0, "VALIDMIN": 0.0, "VALIDMAX": 1000.0, "UNITS": "flag"}
            }
            write_df_to_cdf(df_goes_year, goes_cdf, goes_specs)

        if need_swe and swe_dfs:
            df_swe_year = pd.concat(swe_dfs).sort_index()
            swe_specs = {
                "v_sw": {"var_name": "v_sw", "FILLVAL": -1e31, "VALIDMIN": 200.0, "VALIDMAX": 1200.0, "UNITS": "km/s"},
                "n_sw": {"var_name": "n_sw", "FILLVAL": -1e31, "VALIDMIN": 0.0, "VALIDMAX": 100.0, "UNITS": "cm^-3"}
            }
            write_df_to_cdf(df_swe_year, swe_cdf, swe_specs)

        if need_mfi and mfi_dfs:
            df_mfi_year = pd.concat(mfi_dfs).sort_index()
            mfi_specs = {
                "bz": {"var_name": "bz", "FILLVAL": -1e31, "VALIDMIN": -100.0, "VALIDMAX": 100.0, "UNITS": "nT"}
            }
            write_df_to_cdf(df_mfi_year, mfi_cdf, mfi_specs)

if __name__ == "__main__":
    download_and_save_real_data(start_year=2013, end_year=2016)
