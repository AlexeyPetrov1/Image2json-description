import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parents[0]
PROMPT_DIRECTORY = SRC_DIR / 'config' / 'prompts'
MAX_IMAGES_PER_PREDICT = 2
DEFAULT_DEVICE = 'cuda'
LANGSAM_DEFAULT_MODEL_SAM = "sam2.1_hiera_large"
LANGSAM_BOX_THRESHOLD = 0.29
DATA_DIR = ROOT_DIR / 'data'
DETECTIONS_DIR = DATA_DIR / 'marked_detections'
DETECTIONS_PICTURES_DIR = DATA_DIR / 'marked_detections_pictures'
ANNOTATIONS_DIR = DATA_DIR / 'OUTPUT_JSON'
CLASSES_DIR = DATA_DIR / 'classes'