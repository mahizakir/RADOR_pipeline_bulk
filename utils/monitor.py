"""
System monitor utility: report CPU, memory and GPU usage at a fixed interval.

Usage:
  - As a standalone script:
	  python utils/monitor.py --interval 5 --logfile /path/to/logfile.log

  - As a background monitor inside code:
	  from utils.monitor import SystemMonitor
	  monitor = SystemMonitor(interval=5)
	  monitor.start()
	  ... run work ...
	  monitor.stop()

The monitor will try to use `psutil` and `pynvml` if installed; otherwise it falls back
to parsing `/proc` and calling `nvidia-smi` for GPU stats.
"""

import threading
import time
import logging
import shutil
import subprocess
import argparse
from typing import Optional

try:
	import psutil
	_HAS_PSUTIL = True
except Exception:
	psutil = None
	_HAS_PSUTIL = False

try:
	import pynvml
	pynvml.nvmlInit()
	_HAS_PYNVML = True
except Exception:
	pynvml = None
	_HAS_PYNVML = False


def _get_cpu_mem():
	if _HAS_PSUTIL:
		cpu_perc = psutil.cpu_percent(interval=None)
		cpu_perc_per_core = psutil.cpu_percent(interval=None, percpu=True)
		vm = psutil.virtual_memory()
		return {
			"cpu": cpu_perc,
			"cpu_per_core": cpu_perc_per_core,
			"mem_total": vm.total,
			"mem_used": vm.used,
			"mem_percent": vm.percent,
		}
	# fallback: use /proc/meminfo and loadavg
	try:
		with open('/proc/meminfo', 'r') as f:
			lines = f.readlines()
		info = {}
		for line in lines:
			parts = line.split(':')
			key = parts[0]
			val = parts[1].strip().split()[0]
			info[key] = int(val) * 1024
		mem_total = info.get('MemTotal', 0)
		mem_free = info.get('MemFree', 0) + info.get('Buffers', 0) + info.get('Cached', 0)
		mem_used = mem_total - mem_free
		# cpu percent fallback: use loadavg as heuristic
		load1 = float(open('/proc/loadavg').read().split()[0])
		# On multi-core systems, normalize by CPU count
		try:
			nproc = shutil.cpu_count()
			cpu_perc = min(100.0, (load1 / max(1, nproc)) * 100.0)
		except Exception:
			cpu_perc = load1
		return {
			"cpu": cpu_perc,
			"cpu_per_core": None,
			"mem_total": mem_total,
			"mem_used": mem_used,
			"mem_percent": (mem_used / mem_total * 100.0) if mem_total else 0.0,
		}
	except Exception:
		return {"cpu": None, "cpu_per_core": None, "mem_total": None, "mem_used": None, "mem_percent": None}


def _get_gpu_stats():
	"""Return list of dicts with gpu stats or empty list if no GPUs or info unavailable."""
	# Prefer pynvml if available
	if _HAS_PYNVML:
		gpus = []
		try:
			device_count = pynvml.nvmlDeviceGetCount()
			for i in range(device_count):
				handle = pynvml.nvmlDeviceGetHandleByIndex(i)
				util = pynvml.nvmlDeviceGetUtilizationRates(handle)
				mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
				temp = None
				try:
					temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
				except Exception:
					temp = None
				gpus.append({
					"index": i,
					"util_percent": util.gpu,
					"mem_used": mem.used,
					"mem_total": mem.total,
					"temperature": temp,
				})
		except Exception:
			return []
		return gpus

	# fallback to nvidia-smi if present
	if shutil.which('nvidia-smi'):
		try:
			cmd = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits']
			out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True)
			lines = [l.strip() for l in out.splitlines() if l.strip()]
			gpus = []
			for ln in lines:
				parts = [p.strip() for p in ln.split(',')]
				if len(parts) >= 5:
					idx, util, mem_used, mem_total, temp = parts[:5]
					gpus.append({
						'index': int(idx),
						'util_percent': float(util),
						'mem_used': int(float(mem_used)) * 1024 * 1024,
						'mem_total': int(float(mem_total)) * 1024 * 1024,
						'temperature': float(temp),
					})
			return gpus
		except Exception:
			return []
	return []


class SystemMonitor:
	"""Background system monitor.

	Example:
		monitor = SystemMonitor(interval=5, logger=logging.getLogger('rador'))
		monitor.start()
		... do work ...
		monitor.stop()
	"""

	def __init__(self, interval: int = 5, logger: Optional[logging.Logger] = None, prefix: str = 'monitor'):
		self.interval = max(1, int(interval))
		self.logger = logger or logging.getLogger(prefix)
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._loop, daemon=True)

	def start(self):
		if not self._thread.is_alive():
			self._stop_event.clear()
			self._thread = threading.Thread(target=self._loop, daemon=True)
			self._thread.start()
			self.logger.info('SystemMonitor started (interval=%ds)', self.interval)

	def stop(self):
		self._stop_event.set()
		self._thread.join(timeout=self.interval + 1)
		self.logger.info('SystemMonitor stopped')

	def _loop(self):
		while not self._stop_event.is_set():
			ts = time.strftime('%Y-%m-%d %H:%M:%S')
			cpu_mem = _get_cpu_mem()
			gpus = _get_gpu_stats()
			msg = {
				'timestamp': ts,
				'cpu_percent': cpu_mem.get('cpu'),
				'mem_percent': cpu_mem.get('mem_percent'),
				'mem_used': cpu_mem.get('mem_used'),
				'mem_total': cpu_mem.get('mem_total'),
				'gpus': gpus,
			}
			# Log a concise single-line message
			gpu_summary = ', '.join([f"gpu{g['index']}:util={g.get('util_percent')}% mem={g.get('mem_used')//1024//1024}MiB" for g in gpus]) if gpus else 'no-gpu'
			self.logger.info('[%s] CPU=%s%% MEM=%s%% GPU=%s', ts, msg['cpu_percent'], msg['mem_percent'], gpu_summary)
			time.sleep(self.interval)

if __name__ == '__main__':
	ap = argparse.ArgumentParser(description='Simple CPU/GPU monitor')
	ap.add_argument('--interval', '-i', type=int, default=5, help='Interval in seconds')
	ap.add_argument('--logfile', '-l', default=None, help='Optional logfile to write logs')
	args = ap.parse_args()

	fmt = '%(asctime)s [%(levelname)s] %(message)s'
	if args.logfile:
		logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler(), logging.FileHandler(args.logfile)])
	else:
		logging.basicConfig(level=logging.INFO, format=fmt)

	mon = SystemMonitor(interval=args.interval, logger=logging.getLogger('system_monitor'))
	try:
		mon.start()
		while True:
			time.sleep(3600)
	except KeyboardInterrupt:
		mon.stop()

