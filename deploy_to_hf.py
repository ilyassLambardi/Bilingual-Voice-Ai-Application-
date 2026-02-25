"""Deploy to HuggingFace Spaces — uploads only the files needed for cloud mode."""
import shutil
import tempfile
import os
from pathlib import Path
from huggingface_hub import upload_folder, upload_file

REPO_ID = "ilyass1/bilingual-voice-ai"
PROJECT = Path(__file__).resolve().parent

# Step 1: Upload the HF README (with YAML frontmatter) as README.md
print("[Deploy] Uploading README.md ...")
upload_file(
    path_or_fileobj=str(PROJECT / "HF_README.md"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space",
)

# Step 2: Upload Dockerfile
print("[Deploy] Uploading Dockerfile ...")
upload_file(
    path_or_fileobj=str(PROJECT / "Dockerfile"),
    path_in_repo="Dockerfile",
    repo_id=REPO_ID,
    repo_type="space",
)

# Step 3: Upload .dockerignore
print("[Deploy] Uploading .dockerignore ...")
upload_file(
    path_or_fileobj=str(PROJECT / ".dockerignore"),
    path_in_repo=".dockerignore",
    repo_id=REPO_ID,
    repo_type="space",
)

# Step 4: Upload requirements.txt
print("[Deploy] Uploading requirements.txt ...")
upload_file(
    path_or_fileobj=str(PROJECT / "requirements.txt"),
    path_in_repo="requirements.txt",
    repo_id=REPO_ID,
    repo_type="space",
)

# Step 5: Upload backend/
print("[Deploy] Uploading backend/ ...")
upload_folder(
    folder_path=str(PROJECT / "backend"),
    path_in_repo="backend",
    repo_id=REPO_ID,
    repo_type="space",
    ignore_patterns=["__pycache__/*", "*.pyc", ".env", "models/*"],
)

# Step 6: Upload frontend/
print("[Deploy] Uploading frontend/ ...")
upload_folder(
    folder_path=str(PROJECT / "frontend"),
    path_in_repo="frontend",
    repo_id=REPO_ID,
    repo_type="space",
    ignore_patterns=["node_modules/*", "dist/*"],
)

# Step 7: Upload only the Silero models (small, needed for TTS/VAD)
print("[Deploy] Uploading Silero models ...")
for model_file in ["silero_vad.jit", "v3_en.pt", "v3_de.pt"]:
    fpath = PROJECT / "models" / model_file
    if fpath.exists():
        print(f"  -> {model_file} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")
        upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f"models/{model_file}",
            repo_id=REPO_ID,
            repo_type="space",
        )
    else:
        print(f"  !! {model_file} not found, skipping")

print("\n" + "=" * 60)
print(f"  DEPLOYED! Your site is live at:")
print(f"  https://huggingface.co/spaces/{REPO_ID}")
print(f"  https://{REPO_ID.replace('/', '-')}.hf.space")
print("=" * 60)
