"""Verify delivered file hashes and check required local model assets if requested."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from package_runtime import PACKAGE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root');a=p.parse_args()
    manifest=json.loads((PACKAGE/'PACKAGE_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        path=PACKAGE/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'Missing/modified file: {name}')
    parsed=0
    for folder in ('core','tools','dense_pixel'):
        for path in (PACKAGE/folder).rglob('*.py'):ast.parse(path.read_text());parsed+=1
    if a.project_root:
        model=Path(a.project_root)/'models/Wan2.2-TI2V-5B'
        required=['diffusion_pytorch_model-00001-of-00003.safetensors',
                  'diffusion_pytorch_model-00002-of-00003.safetensors',
                  'diffusion_pytorch_model-00003-of-00003.safetensors',
                  'models_t5_umt5-xxl-enc-bf16.pth','Wan2.2_VAE.pth','google/umt5-xxl']
        missing=[name for name in required if not (model/name).exists()]
        if missing:raise FileNotFoundError(f'Missing model assets: {missing}')
    print(json.dumps(dict(package_files_verified=len(manifest['files']),project_python_files_parsed=parsed,
                         note='File/dependency presence check, not training quality validation')))


if __name__=='__main__':main()
