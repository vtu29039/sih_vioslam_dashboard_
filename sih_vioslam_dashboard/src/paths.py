import os

# Resolve config relative to the project root so the app works from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "camera_params.yaml")
