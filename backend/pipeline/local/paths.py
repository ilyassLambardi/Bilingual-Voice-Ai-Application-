"""
Central path definitions for local pipeline modules.

All local modules should import from here instead of computing paths from __file__.
"""

from pathlib import Path

# local/ → pipeline/ → backend/ → project_root/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
