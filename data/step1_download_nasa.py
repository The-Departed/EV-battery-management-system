import os
import requests
import zipfile
from pathlib import Path

def download_nasa_dataset():
    """
    Step 1: Downloads the NASA Ames Battery Aging Dataset (FY08Q4).
    Contains cells B0005, B0006, B0007, B0018.
    """
    url = "https://ti.arc.nasa.gov/m/project/prognostic-repository/BatteryAgingARC-FY08Q4.zip"
    
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

    print("⏳ Downloading NASA 18650 Battery Aging Dataset (This may take a minute)...")
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("📦 Extracting dataset...")
    
    # Extract the specific .mat files we need
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if any(target in file for target in target_files):
                file_name = os.path.basename(file)
                if not file_name: continue
                source = zip_ref.open(file)
                target = open(data_dir / file_name, "wb")
                with source, target:
                    import shutil
                    shutil.copyfileobj(source, target)
                    
    # Clean up zip file
    os.remove(zip_path)
    print(f"✅ Download complete! Files saved to: {data_dir}")

if __name__ == "__main__":
    download_nasa_dataset()
