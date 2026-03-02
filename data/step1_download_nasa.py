import os
import io
import shutil
import requests
import zipfile
from pathlib import Path

def download_nasa_dataset():
    """
    Step 1: Downloads the NASA Ames Battery Aging Dataset.
    Contains cells B0005, B0006, B0007, B0018.
    
    The PHM S3 mirror hosts a nested zip:
      outer.zip -> "5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip" -> *.mat
    """
    # Primary URL is the PHM Society S3 mirror (the original NASA URL is dead)
    urls = [
        "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
        "https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC-FY08Q4.zip",
    ]
    
    # Define directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "nasa"
    zip_path = data_dir / "nasa_battery.zip"
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have the extracted files
    target_files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]
    has_files = all((data_dir / f).exists() for f in target_files)
    
    if has_files:
        print("✅ NASA dataset already downloaded and extracted in data/nasa/.")
        return

    # Remove any previously corrupted download
    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            print("⚠️  Removing corrupted previous download...")
            os.remove(zip_path)

    if not zip_path.exists():
        downloaded = False
        for url in urls:
            try:
                print(f"⏳ Downloading NASA Battery Dataset from:\n   {url}")
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()
                
                # Verify we're actually getting a zip, not an HTML redirect
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type.lower():
                    print(f"⚠️  URL returned HTML (likely a redirect). Trying next mirror...")
                    continue
                
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                
                # Double-check the downloaded file is a valid zip
                if not zipfile.is_zipfile(zip_path):
                    print(f"⚠️  Downloaded file is not a valid zip. Trying next mirror...")
                    os.remove(zip_path)
                    continue
                
                downloaded = True
                break
            except requests.RequestException as e:
                print(f"⚠️  Download failed ({e}). Trying next mirror...")
                if zip_path.exists():
                    os.remove(zip_path)
                continue
        
        if not downloaded:
            raise RuntimeError(
                "❌ Could not download the NASA battery dataset from any mirror.\n"
                "Please download it manually and place the .mat files in data/nasa/"
            )

    print("📦 Extracting dataset...")
    
    _extract_mat_files(zip_path, data_dir, target_files)
                    
    # Clean up zip file
    if zip_path.exists():
        os.remove(zip_path)
    
    # Verify extraction
    missing = [f for f in target_files if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"❌ Failed to extract: {missing}")
    
    print(f"✅ Download complete! Files saved to: {data_dir}")


def _extract_mat_files(zip_path, data_dir, target_files):
    """
    Handle both flat zips (mat files at top level) and nested zips
    (PHM mirror: outer zip contains inner zips which contain the mat files).
    """
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        names = outer_zip.namelist()
        
        # Case 1: .mat files directly in this zip
        mat_names = [n for n in names if any(t in n for t in target_files)]
        if mat_names:
            for file in mat_names:
                file_name = os.path.basename(file)
                if not file_name:
                    continue
                with outer_zip.open(file) as src, open(data_dir / file_name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            return
        
        # Case 2: Nested zip — look for inner zip containing "FY08Q4"
        inner_zips = [n for n in names if n.endswith('.zip') and 'FY08Q4' in n]
        if not inner_zips:
            # Fallback: try ALL inner zips
            inner_zips = [n for n in names if n.endswith('.zip')]
        
        for inner_name in inner_zips:
            print(f"   📂 Opening nested archive: {inner_name}")
            inner_bytes = outer_zip.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
                for file in inner_zip.namelist():
                    if any(t in file for t in target_files):
                        file_name = os.path.basename(file)
                        if not file_name:
                            continue
                        with inner_zip.open(file) as src, open(data_dir / file_name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
            # Check if we got everything after this inner zip
            if all((data_dir / f).exists() for f in target_files):
                return

if __name__ == "__main__":
    download_nasa_dataset()
