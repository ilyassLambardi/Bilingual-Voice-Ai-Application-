"""Resumable download for whisper large-v3-turbo model.

Run:  python download_model.py

It will resume from where it left off if interrupted.
Once complete, `whisper.load_model('large-v3-turbo')` will use the cached file.
"""

import hashlib
import os
import time

import requests

URL = "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
EXPECTED_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"

# Whisper default cache dir
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
DEST = os.path.join(CACHE_DIR, "small.pt")

MAX_RETRIES = 50
CHUNK_SIZE = 1024 * 256  # 256 KB chunks
TIMEOUT = 30  # seconds per request


def download():
    os.makedirs(CACHE_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        # Check how much we already have
        downloaded = os.path.getsize(DEST) if os.path.exists(DEST) else 0

        headers = {}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"[Attempt {attempt}] Resuming from {downloaded / 1e6:.1f} MB ...")
        else:
            print(f"[Attempt {attempt}] Starting download ...")

        try:
            resp = requests.get(URL, headers=headers, stream=True, timeout=TIMEOUT)

            # If server doesn't support Range or file is complete
            if resp.status_code == 416:
                print("Server says file is already complete.")
                break
            if resp.status_code == 200 and downloaded > 0:
                # Server ignored Range header, restart
                print("Server does not support resume. Restarting ...")
                downloaded = 0
                mode = "wb"
            elif resp.status_code == 206:
                mode = "ab"
            elif resp.status_code == 200:
                mode = "wb"
            else:
                print(f"Unexpected status {resp.status_code}. Retrying ...")
                time.sleep(5)
                continue

            total = int(resp.headers.get("content-length", 0)) + downloaded
            print(f"Total size: {total / 1e6:.1f} MB")

            with open(DEST, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded / total * 100 if total else 0
                        print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.1f}%)", end="", flush=True)

            print()  # newline after progress
            if downloaded >= total:
                print("Download complete!")
                break

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            wait = min(attempt * 3, 30)
            print(f"Retrying in {wait}s ...")
            time.sleep(wait)

    # Verify hash
    print("Verifying checksum ...")
    sha = hashlib.sha256()
    with open(DEST, "rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha.update(data)
    if sha.hexdigest() == EXPECTED_SHA:
        print(f"OK! Model saved to {DEST}")
    else:
        print(f"WARNING: SHA256 mismatch! Got {sha.hexdigest()}")
        print(f"Expected {EXPECTED_SHA}")
        print("The file may be corrupted. Delete it and re-run this script.")


if __name__ == "__main__":
    download()
