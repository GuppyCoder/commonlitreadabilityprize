from pathlib import Path
import os

COMP_NAME = "commonlitreadabilityprize"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = Path(
	os.environ.get("COMMONLIT_INPUT_PATH", str(PROJECT_ROOT / "input"))
)
OUTPUT_PATH = Path(
	os.environ.get("COMMONLIT_OUTPUT_PATH", str(PROJECT_ROOT / "output"))
)
CONFIG_PATH = Path(
	os.environ.get("COMMONLIT_CONFIG_PATH", str(PROJECT_ROOT / "hyperparams.yml"))
)
MODEL_CACHE = Path(
	os.environ.get(
		"COMMONLIT_MODEL_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface")
	)
)

