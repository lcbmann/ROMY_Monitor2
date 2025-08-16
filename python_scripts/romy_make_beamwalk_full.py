#!/usr/bin/env python3
"""Exact visual replication of makeplot_beamwalk.py -> new_figures/html_beamwalk.png"""
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ORIG = SCRIPT_DIR / 'makeplot_beamwalk.py'
OUT_DIR = REPO / 'new_figures'
OUT_DIR.mkdir(exist_ok=True)

code = ORIG.read_text()
code = re.sub(r"config\['path_to_figs'\]\s*=.*", f"config['path_to_figs'] = r'{OUT_DIR.as_posix()}/'", code, count=1)

ns = {'__name__':'__main__'}
exec(compile(code, str(ORIG), 'exec'), ns, ns)
print('✔ beamwalk full replication →', OUT_DIR/'html_beamwalk.png')
