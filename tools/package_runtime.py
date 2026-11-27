"""Use the bundled source trees without pip-installing or modifying a shared env."""
import os
from pathlib import Path
import sys

PACKAGE=Path(__file__).resolve().parents[1]
REPO=PACKAGE/'vendor/DiffSynth-Studio'
NATIVE=REPO/'examples/wanvideo/model_training/dense_lora'


def setup():
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD','True')
    os.environ.setdefault('DIFFSYNTH_ATTENTION_IMPLEMENTATION','torch')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    for path in (PACKAGE/'core',REPO,NATIVE):
        if str(path) not in sys.path:sys.path.append(str(path))
    return PACKAGE
