import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFERRED_PORT = int(os.environ.get("ONISCAN_PORT", "8000"))
HOST = os.environ.get("ONISCAN_HOST", "127.0.0.1")


def get_free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((HOST, preferred))
            return preferred
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


port = get_free_port(PREFERRED_PORT)
cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "backend.main:app",
    "--host",
    HOST,
    "--port",
    str(port),
]

print(f"Starting ONI-SCAN on http://{HOST}:{port}")
print("Using command:")
print(" ".join(cmd))

os.chdir(ROOT)
subprocess.run(cmd, check=False)
