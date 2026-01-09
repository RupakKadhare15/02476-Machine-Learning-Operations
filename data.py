import requests
import shutil
from pathlib import Path

# --- CONFIG ---
DRIVE_ID = "1czsN8ebcoAkwAhs6rKdw3Enz0oBzdzTP"
DATA_PATH = Path("data")
# --------------

def main():
    # This creates the directory
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_PATH / "data.zip"

    print("Downloading...")
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': DRIVE_ID}, stream=True)
    
    # Handle Google's virus warning token
    token = next((v for k, v in response.cookies.items() if k.startswith('download_warning')), None)
    if token:
        response = session.get(url, params={'id': DRIVE_ID, 'confirm': token}, stream=True)

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)

    print("Extracting...")
    shutil.unpack_archive(zip_path, DATA_PATH)
    print("Done.")

if __name__ == "__main__":
    main()