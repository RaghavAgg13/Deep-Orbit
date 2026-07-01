import cdflib
import numpy as np
import pandas as pd
from pathlib import Path

def list_cdf_vars(path):
    """List all zVariables and rVariables in a CDF file."""
    cdf = cdflib.CDF(path)
    info = cdf.cdf_info()
    return info.zVariables + info.rVariables

def read_cdf_to_df(path, time_var="Epoch", var_map=None):
    """
    Read selected variables from a CDF file into a Pandas DataFrame.
    
    Args:
        path: str or Path, path to CDF file
        time_var: str, name of time epoch variable in CDF
        var_map: dict, mappings from DataFrame column name to CDF variable name
                 e.g. {"v_sw": "V_SW"}
                 
    Returns:
        pd.DataFrame indexed by UTC time.
    """
    if var_map is None:
        var_map = {}
        
    path_str = str(path)
    cdf = cdflib.CDF(path_str)
    
    # 1. Read and parse time Epoch
    try:
        epoch_raw = cdf.varget(time_var)
        epoch_arr = np.asarray(epoch_raw, dtype="float64")
        
        # Fast path: CDF_EPOCH values are milliseconds since Year 0, approx 6.3e13 for modern dates.
        # We check if the values lie in the expected CDF_EPOCH range.
        if len(epoch_arr) > 0 and 6.0e13 < epoch_arr[0] < 7.0e13:
            # Offset of Jan 1, 1970 00:00:00 in milliseconds since Year 0 is 62,167,219,200,000
            time_index = pd.to_datetime(epoch_arr - 62167219200000.0, unit="ms", utc=True)
        else:
            # Fallback for TT2000 (nanoseconds since J2000) or other CDF epoch formats
            times = cdflib.cdfepoch.to_datetime(epoch_raw)
            time_index = pd.to_datetime(times, utc=True)
    except Exception as e:
        raise ValueError(f"Error parsing time variable '{time_var}' in CDF {path}: {e}")
        
    out_dict = {"time": time_index}
    
    # 2. Read each variable and apply cleaning based on CDF metadata
    for col_name, cdf_var in var_map.items():
        try:
            arr = np.asarray(cdf.varget(cdf_var), dtype="float64")
            
            # Fetch attributes for FILLVAL, VALIDMIN, VALIDMAX
            attrs = cdf.varattsget(cdf_var)
            
            # Handle FILLVAL
            fill_val = attrs.get("FILLVAL")
            if fill_val is not None:
                # If fill_val is a list/array, extract scalar
                if isinstance(fill_val, (list, np.ndarray)) and len(fill_val) > 0:
                    fill_val = fill_val[0]
                arr = np.where(np.isclose(arr, fill_val), np.nan, arr)
                
            # Handle VALIDMIN
            valid_min = attrs.get("VALIDMIN")
            if valid_min is not None:
                if isinstance(valid_min, (list, np.ndarray)) and len(valid_min) > 0:
                    valid_min = valid_min[0]
                arr = np.where(arr < valid_min, np.nan, arr)
                
            # Handle VALIDMAX
            valid_max = attrs.get("VALIDMAX")
            if valid_max is not None:
                if isinstance(valid_max, (list, np.ndarray)) and len(valid_max) > 0:
                    valid_max = valid_max[0]
                arr = np.where(arr > valid_max, np.nan, arr)
                
            out_dict[col_name] = arr
            
        except Exception as e:
            # If variable is missing or corrupt, fill with NaN
            print(f"Warning: could not read var '{cdf_var}' from {path}: {e}")
            out_dict[col_name] = np.full(len(time_index), np.nan)
            
    df = pd.DataFrame(out_dict).set_index("time").sort_index()
    return df

def read_cdf_directory(directory_path, time_var="Epoch", var_map=None, suffix="*.cdf"):
    """
    Read all CDF files in a directory and concatenate them chronologically.
    """
    dir_path = Path(directory_path)
    files = sorted(list(dir_path.glob(suffix)))
    
    if not files:
        print(f"Warning: No files found matching {suffix} in {directory_path}")
        # Return empty DataFrame with time index
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            df_file = read_cdf_to_df(f, time_var, var_map)
            dfs.append(df_file)
        except Exception as e:
            print(f"Error reading CDF file {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    df_all = pd.concat(dfs)
    # De-duplicate timestamps, keeping the last reprocessed entry
    df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
    return df_all
