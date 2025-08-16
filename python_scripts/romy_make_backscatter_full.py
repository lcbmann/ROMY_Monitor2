#!/usr/bin/env python3
"""Exact visual replication of makeplot_backscatter.py
Redirects output to new_figures/html_backscatter.png while preserving original plot code.
Usage: python romy_make_backscatter_full.py [RING]
"""
from pathlib import Path
import sys, re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ORIG = SCRIPT_DIR / 'makeplot_backscatter.py'
OUT_DIR = REPO / 'new_figures'
OUT_DIR.mkdir(exist_ok=True)

ring_override = sys.argv[1].upper() if len(sys.argv)>1 and sys.argv[1].upper() in 'ZUVW' else None

code = ORIG.read_text()
code = re.sub(r"config\['path_to_figs'\]\s*=.*", f"config['path_to_figs'] = r'{OUT_DIR.as_posix()}/'", code, count=1)

ns = {'__name__':'__main__'}
exec(compile(code, str(ORIG), 'exec'), ns, ns)
if ring_override and 'config' in ns:
    ns['config']['ring'] = ring_override
print('✔ backscatter full replication →', OUT_DIR/'html_backscatter.png')
