"""Read log deltas and resource health for this experiment only."""
import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess

out = Path(__file__).absolute().parents[1]
control = out / '_launcher'
log = control / ('formal_host.log' if (control / 'formal_started_at').exists() else 'smoke_host.log')
cursor = control / ('.' + log.stem + '.poll')
offset = int(cursor.read_text()) if cursor.exists() else 0
if offset > log.stat().st_size:
    offset = 0
with log.open('rb') as handle:
    handle.seek(offset)
    content = handle.read().decode(errors='replace')
    cursor.write_text(str(handle.tell()))
lines = re.sub(r'\x1b\[[0-9;]*m', '', content).replace('\r', '\n').splitlines()
important = [s for s in lines if any(k in s for k in ('Traceback', 'Error executing', 'RuntimeError', 'OutOfMemory',
    'SwanLab', 'swanlab:', 'Iterations:', 'Epoch:', 'global_step', 'optimizer_steps', 'Checkpoint saved',
    'Global Step:', 'Elapsed:', 'ppo_loss=', 'sft_loss=', 'grad_norm=', 'Saving checkpoint',
    'Multi-task restore', 'gradient_checkpointing=enabled', 'ERROR:', 'status": "PASS"'))]
resources = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()
panes = subprocess.check_output(['tmux', 'list-panes', '-t', 'taccorl-pirl-step400', '-F', 'dead=#{pane_dead} exit=#{pane_dead_status}'], text=True).strip()
result = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'log_bytes': log.stat().st_size,
          'log_age_seconds': datetime.datetime.now().timestamp() - log.stat().st_mtime,
          'free_disk_gib': round(shutil.disk_usage(out).free / 2**30, 1),
          'pane': panes, 'events': important[-12:], 'tail': lines[-5:], 'gpu_processes': resources,
          'checkpoint_audits': sorted(p.name for p in (out / 'validation').glob('*-step*.json'))}
with (out / 'telemetry.jsonl').open('a') as handle:
    handle.write(json.dumps(result) + '\n')
print(json.dumps(result, indent=2))
