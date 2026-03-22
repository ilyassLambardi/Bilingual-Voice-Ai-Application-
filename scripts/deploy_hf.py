"""
Deploy to HuggingFace Spaces via Python API (no git needed).
Stages files into a temp folder, then uploads in one commit.
"""

import fnmatch
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

REPO_ID = "ilyass1/starch"
REPO_TYPE = "space"
PROJECT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {"__pycache__", "node_modules", "dist", ".vite", ".git", "data",
             ".ipynb_checkpoints", ".vscode", ".idea", ".windsurf"}
SKIP_FILES = {".env", ".env.example"}
SKIP_PATTERNS = ["*.pyc", "*.pyo", "*.log", "*.ipynb"]


def copytree_filtered(src, dst):
    """Copy directory tree, skipping junk."""
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if item in SKIP_DIRS:
                continue
            copytree_filtered(s, d)
        else:
            if item in SKIP_FILES:
                continue
            if any(fnmatch.fnmatch(item, p) for p in SKIP_PATTERNS):
                continue
            shutil.copy2(s, d)


def main():
    api = HfApi()
    who = api.whoami()
    print(f"Logged in as: {who['name']}")
    print(f"Target: {REPO_ID}\n")

    # Stage all files into a clean temp folder
    stage = Path(tempfile.mkdtemp(prefix="hf_stage_"))
    print(f"[1/3] Staging files to {stage}...")

    # README.md (from HF_README.md)
    shutil.copy2(PROJECT / "HF_README.md", stage / "README.md")
    print("  README.md")

    # Root files
    for f in ["Dockerfile", "requirements.txt", ".dockerignore"]:
        src = PROJECT / f
        if src.exists():
            shutil.copy2(src, stage / f)
            print(f"  {f}")

    # Backend
    copytree_filtered(str(PROJECT / "backend"), str(stage / "backend"))
    n_backend = sum(1 for _ in (stage / "backend").rglob("*") if _.is_file())
    print(f"  backend/ ({n_backend} files)")

    # Frontend
    copytree_filtered(str(PROJECT / "frontend"), str(stage / "frontend"))
    n_frontend = sum(1 for _ in (stage / "frontend").rglob("*") if _.is_file())
    print(f"  frontend/ ({n_frontend} files)")

    # VAD model
    os.makedirs(stage / "models", exist_ok=True)
    vad = PROJECT / "models" / "silero_vad.jit"
    if vad.exists():
        shutil.copy2(vad, stage / "models" / "silero_vad.jit")
        print(f"  models/silero_vad.jit ({vad.stat().st_size // 1024}KB)")

    total = sum(1 for _ in stage.rglob("*") if _.is_file())
    print(f"\n  Total: {total} files staged")

    # Upload everything in one commit
    print(f"\n[2/3] Uploading to {REPO_ID} (this may take a few minutes)...")
    upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(stage),
        commit_message="Deploy: Noise cancellation VAD, redesigned UI, EdgeTTS, full pipeline",
        delete_patterns=["*"],  # remove old files not in this upload
    )

    # Cleanup
    print("\n[3/3] Cleaning up...")
    shutil.rmtree(stage, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  DEPLOYED: https://huggingface.co/spaces/{REPO_ID}")
    print(f"  LIVE URL: https://ilyass1-starch.hf.space")
    print(f"{'='*60}")
    print(f"\nDocker build will start automatically (~10 min).")
    print(f"Set GROQ_API_KEY in: https://huggingface.co/spaces/{REPO_ID}/settings")


if __name__ == "__main__":
    main()
