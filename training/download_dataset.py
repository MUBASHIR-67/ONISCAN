from pathlib import Path
import requests
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://dataverse.harvard.edu/api/access/dataset/"
    ":persistentId/?persistentId=doi:10.7910/DVN/HAUXLC"
)

ZIP_FILE = RAW_DIR / "bad_onion_dataset.zip"

print("=" * 60)
print("          ONI-SCAN DATASET DOWNLOADER")
print("=" * 60)

print("\n[1/3] Connecting to Harvard Dataverse...")
response = requests.get(URL, stream=True, timeout=60)
response.raise_for_status()

total = int(response.headers.get("content-length", 0))
downloaded = 0

print("[OK] Dataset connection established.")
print("[2/3] Downloading dataset...")

with open(ZIP_FILE, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = downloaded * 100 / total
                print(f"\r      {percent:6.2f}%", end="")

print("\n[OK] Download complete.")

print("[3/3] Extracting dataset...")

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(RAW_DIR)

ZIP_FILE.unlink()

print("[OK] Dataset extracted.")
print(f"\nDataset location: {RAW_DIR}")

print("\n" + "=" * 60)
print("          DOWNLOAD COMPLETE")
print("=" * 60)
