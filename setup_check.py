from pathlib import Path
import sys

print("=" * 50)
print("        ONI-SCAN SETUP CHECK")
print("=" * 50)

print(f"\nPython: {sys.version.split()[0]}")

folders = [
    "backend",
    "dataset",
    "dataset/raw",
    "dataset/processed",
    "training",
    "models",
]

print("\nChecking project folders...")

for folder in folders:
    path = Path(folder)

    if path.exists():
        print(f"[OK]      {folder}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"[CREATED] {folder}")

print("\nChecking AI libraries...")

try:
    import ultralytics
    print(f"[OK]      Ultralytics {ultralytics.__version__}")
except Exception as e:
    print("[ERROR]   Ultralytics is not available")
    print(e)

try:
    import torch
    print(f"[OK]      PyTorch {torch.__version__}")
    print(f"[INFO]    CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print("[ERROR]   PyTorch is not available")
    print(e)

try:
    import fastapi
    print(f"[OK]      FastAPI {fastapi.__version__}")
except Exception as e:
    print("[ERROR]   FastAPI is not available")
    print(e)

print("\n" + "=" * 50)
print("ONI-SCAN SETUP CHECK COMPLETE")
print("=" * 50)
