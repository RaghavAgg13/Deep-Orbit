from cdasws import CdasWs
import sys

def main():
    print("Initializing CDAS Web Service client...")
    cdas = CdasWs()
    
    # 1. Search for Wind SWE datasets
    print("\n=== Wind SWE Datasets ===")
    try:
        datasets = cdas.get_datasets(observatory='Wind', instrument='Plasma and Solar Wind')
        for ds in datasets:
            ds_id = ds.get('Id')
            if 'SWE' in ds_id:
                print(f"ID: {ds_id} | Name: {ds.get('Label')}")
    except Exception as e:
        print("Error searching Wind SWE:", e)
        
    # 2. Search for Wind MFI datasets
    print("\n=== Wind MFI Datasets ===")
    try:
        datasets = cdas.get_datasets(observatory='Wind', instrument='Magnetic Fields (space)')
        for ds in datasets:
            ds_id = ds.get('Id')
            if 'MFI' in ds_id:
                print(f"ID: {ds_id} | Name: {ds.get('Label')}")
    except Exception as e:
        print("Error searching Wind MFI:", e)

    # 3. Search for GOES datasets
    print("\n=== GOES Datasets ===")
    try:
        datasets = cdas.get_datasets(observatory='GOES')
        for ds in datasets:
            ds_id = ds.get('Id')
            # Check for EPEAD or MPSH or MAGED
            if any(term in ds_id.upper() for term in ['EPEAD', 'MPSH', 'MAGED', 'EPS']):
                print(f"ID: {ds_id} | Name: {ds.get('Label')}")
    except Exception as e:
        print("Error searching GOES:", e)

if __name__ == "__main__":
    main()
