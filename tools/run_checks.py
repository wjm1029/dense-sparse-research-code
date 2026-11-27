"""Run CPU tests plus a real two-process dispersion-gradient comparison."""
import argparse
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from package_runtime import setup,PACKAGE,REPO,NATIVE
setup()
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',default=str(PACKAGE/'evidence/CHECKS_CURRENT.json'));a=p.parse_args()
    torch.set_num_threads(2);records=[];begin=time.monotonic()
    # Core modules and workflow tests use unittest; historical scripts also contain
    # pytest-style plain functions which unittest discovery would silently ignore.
    for module_name in ('test_core','test_workflow','test_wan22_opsd_prefix_cache'):
        module=importlib.import_module(module_name)
        suite=unittest.defaultTestLoader.loadTestsFromModule(module)
        result=unittest.TextTestRunner(verbosity=1).run(suite)
        records.append(dict(module=module_name,tests=result.testsRun,passed=result.wasSuccessful()))
        if not result.wasSuccessful():raise SystemExit(1)
    for module_name in ('test_wan22_dense6_sampler','test_wan22_causal_attention','test_wan22_ar_teacher_forcing',
                        'test_wan22_ar_self_forcing','test_wan22_opsd_bounded_causal_attention'):
        module=importlib.import_module(module_name);count=0
        for name,function in inspect.getmembers(module,inspect.isfunction):
            if name.startswith('test_') and function.__module__==module.__name__:
                if inspect.signature(function).parameters:raise ValueError(f'Unsupported test fixture: {name}')
                function();count+=1
        if not count:raise RuntimeError(f'No tests found in {module_name}')
        records.append(dict(module=module_name,tests=count,passed=True))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        PYTHONPATH=os.pathsep.join((str(REPO),str(NATIVE))))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
             str(NATIVE/'test_wan22_dense6_dispersion.py')]
    completed=subprocess.run(command,cwd=REPO,env=environment,text=True,capture_output=True)
    records.append(dict(module='two_process_dispersion_gradient',passed=completed.returncode==0,
                        output=completed.stdout+completed.stderr))
    if completed.returncode:print(completed.stdout+completed.stderr);raise SystemExit(completed.returncode)
    # Parse the exact training entries, so accidental missing/invalid source files fail.
    from entrypoints import STAGES
    for key,row in STAGES.items():
        base=PACKAGE if key=='jta-bidirectional' else NATIVE
        ast.parse((base/row['trainer']).read_text())
    report=dict(scope='CPU numerical/gradient/integrity tests; not full historical training reproduction',
        passed=True,unit_test_count=sum(r.get('tests',0) for r in records),checks=records,
        all_stage_sources_parsed=True,elapsed_seconds=time.monotonic()-begin)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True);Path(a.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
