"""On-demand generation helpers for helicorder & rotation spectra (live/archival).

Used by:
  * live/live_server.py (API mode)
  * python_scripts/romy_build_live_archive.py (rolling archive)

Outputs are directed to a caller supplied directory so they never override
the daily monitoring images. Functions create placeholder images when data
or dependencies are missing.
"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, Stream

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / 'develop' / 'spectra'))
try:
	from spectra import spectra  # type: ignore
except Exception:
	spectra = None

MOUNT    = Path(os.getenv('ROMY_MOUNT', '/import/freenas-ffb-01-data')).expanduser()
SDS_ARCH = MOUNT / 'romy_archive'
RINGS = set('ZUVW')

def _ring_seed(ring: str) -> tuple[str,str]:
	loc = dict(Z='10', U='', V='', W='')[ring]
	seed = f"BW.ROMY.{loc}.BJ{ring}"
	return seed, loc

def _read_sds(seed: str, t1, t2) -> Stream:
	from obspy.clients.filesystem.sds import Client
	if not SDS_ARCH.is_dir():
		return Stream()
	n,s,l,c = seed.split('.')
	try:
		cli = Client(str(SDS_ARCH), sds_type='D')
		return cli.get_waveforms(n,s,l,c, UTCDateTime(t1), UTCDateTime(t2), merge=-1)
	except Exception:
		return Stream()

def _ensure_dir(d: Path):
	d.mkdir(parents=True, exist_ok=True)

def _placeholder(out: Path, title: str, subtitle: str):
	fig, ax = plt.subplots(figsize=(8,4))
	ax.text(0.5,0.62, title, ha='center', va='center', fontsize=16, weight='bold')
	ax.text(0.5,0.40, subtitle, ha='center', va='center', fontsize=11)
	ax.text(0.5,0.20, 'PLACEHOLDER', ha='center', va='center', fontsize=9, color='crimson')
	ax.axis('off')
	fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
	return out

def _normalise_date(datestr: str) -> str:
	try:
		return datetime.strptime(datestr, '%Y-%m-%d').strftime('%Y-%m-%d')
	except ValueError:
		raise ValueError('Date must be YYYY-MM-DD')

def generate_helicorder(ring: str, date_str: str, out_dir: str | Path) -> Path:
	"""Generate a colour helicorder (hour‑stacked) matching romy_make_helicorder style.

	Each hour is normalised independently and coloured using jet_r colormap.
	Old images are left intact (idempotent) – delete manually to force regen.
	"""
	ring = ring.upper()
	if ring not in RINGS:
		raise ValueError('Ring must be one of Z,U,V,W')
	date_str = _normalise_date(date_str)
	out_dir = Path(out_dir); _ensure_dir(out_dir)
	out_png = out_dir / f"helicorder_live_R{ring}_{date_str}.png"
	if out_png.exists():
		return out_png
	seed, _ = _ring_seed(ring)

	# Collect hour traces individually (keeps gaps + per-hour normalisation)
	hour_traces: list[np.ndarray] = []
	for hour in range(24):
		t1 = UTCDateTime(f"{date_str} 00:00:00") + hour*3600
		t2 = t1 + 3600
		st = _read_sds(seed, t1-30, t2+30)
		if not st:
			hour_traces.append(np.array([]))
			continue
		tr = st[0].copy()
		tr.trim(t1, t2)
		hour_traces.append(tr.data.astype(float))

	if all(len(h) == 0 for h in hour_traces):
		return _placeholder(out_png, f"R{ring} Helicorder", f"{date_str} – no data")

	colours = plt.cm.jet_r(np.linspace(0,1,24))
	fig, ax = plt.subplots(figsize=(11,7.5))
	# Plot oldest hour at top: reverse list so hour 23 at bottom visually consistent
	for idx, arr in enumerate(hour_traces[::-1]):
		if arr.size == 0:
			continue
		norm = np.nanmax(np.abs(arr)) or 1.0
		x = np.linspace(0, 60, arr.size)
		colour = colours[idx]
		ax.plot(x, 0.8*arr/norm + idx, lw=0.45, color=colour)

	ax.set(ylim=(-1,24),
		   yticks=np.arange(0,24,3),
		   yticklabels=[f"{h:02d}:00" for h in range(23,-1,-3)],
		   xlabel='Time (minutes)', ylabel='Hour (UTC)',
		   title=f"Rotation Helicorder R{ring} – {date_str}")
	ax.grid(ls=':', alpha=.4)
	fig.tight_layout()
	fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
	plt.close(fig)
	return out_png

def generate_rotation_spectra(ring: str, date_str: str, out_dir: str | Path) -> Path:
	ring = ring.upper()
	if ring not in RINGS:
		raise ValueError('Ring must be one of Z,U,V,W')
	date_str = _normalise_date(date_str)
	out_dir = Path(out_dir); _ensure_dir(out_dir)
	out_png = out_dir / f"rotation_spectrum_live_R{ring}_{date_str}.png"
	if out_png.exists():
		return out_png
	if spectra is None:
		return _placeholder(out_png, f"R{ring} Rotation Spectrum", f"{date_str} – dependency missing")
	seed, _ = _ring_seed(ring)
	big = Stream()
	for hr in range(24):
		t1 = UTCDateTime(f"{date_str} 00:00:00") + hr*3600
		t2 = t1 + 3600
		st = _read_sds(seed, t1-30, t2+30)
		if st: big += st
	if not big:
		return _placeholder(out_png, f"R{ring} Rotation Spectrum", f"{date_str} – no data")
	big.merge(method=1, fill_value='interpolate')
	cfg = dict(tbeg=f"{date_str} 00:00:00", tend=f"{date_str} 23:59:59", seeds=[seed],
			   path_to_sds=str(SDS_ARCH), path_to_data_out=str(REPO_ROOT/'local_output'/'rotation_data'),
			   path_to_figures_out=str(REPO_ROOT/'local_output'/'rotation_plots'), tinterval=3600, toverlap=0,
			   method='welch', fmin=0.001, fmax=5.0, apply_average=True, fraction_of_octave=12,
			   averaging='median', data_unit='rad/s', quality_filter='good', threshold=1e-15,
			   zero_seq_limit=20, high_seq_limit=20, flat_seq_limit=20, verbose=False)
	sp = spectra(cfg)
	sp.add_trace(big[0])
	sp.get_collection(cfg['tinterval'], cfg['toverlap'], cfg['method'])
	if cfg['apply_average']:
		sp.get_fband_average(cfg['fraction_of_octave'], cfg['averaging'])
	if cfg['quality_filter']:
		sp.classify_collection_quality(threshold=cfg['threshold'], zero_seq_limit=cfg['zero_seq_limit'],
									   high_seq_limit=cfg['high_seq_limit'], flat_seq_limit=cfg['flat_seq_limit'])
	n_int = len(sp.collection['time'])
	colours = plt.cm.jet_r(np.linspace(0,1,n_int))[::-1]
	fig, ax = plt.subplots(figsize=(8,6))
	for i,(f,s) in enumerate(zip(sp.collection['freq'], sp.collection['spec'])):
		col = 'lightgrey' if s is None or len(s)==0 or np.all(np.isnan(s)) else colours[i]
		ax.plot(f, s, lw=0.6, alpha=0.85, color=col)
	ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.001,5.0)
	ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (rad^2/s^2/Hz)')
	ax.set_title(f"R{ring} Rotation Spectra – {date_str}")
	ax.grid(True, which='both', ls='--', alpha=.3)
	fig.savefig(out_png, dpi=140, bbox_inches='tight'); plt.close(fig)
	return out_png

def generate_all(rings: list[str], date_str: str, out_dir: str | Path) -> dict:
	out = {}
	for r in rings:
		try:
			h = generate_helicorder(r, date_str, out_dir)
			s = generate_rotation_spectra(r, date_str, out_dir)
			out[r.upper()] = {'helicorder': str(h), 'rotation': str(s)}
		except Exception as e:
			out[r.upper()] = {'error': str(e)}
	return out

__all__ = ['generate_helicorder','generate_rotation_spectra','generate_all']
