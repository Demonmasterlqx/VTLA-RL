"""Check final merged weights on CPU without running inference."""
import hashlib
import json
import os
from pathlib import Path

import torch
from openpi.training.config import get_config

out = Path(os.environ['TACCORL_RUN_DIR'])
export = out / 'exports/step_400'
metadata = json.loads((export / 'export_meta.json').read_text())
source = metadata['source_ckpt_metadata']
assert source['global_step'] == source['target_global_step'] == 400
assert source['is_final'] and source['discrete_state_input'] is False
cfg = get_config(source['deployment_config_name'])
model = cfg.model.load_pytorch(cfg, str(export / 'model.safetensors'))
assert all(torch.isfinite(value).all() for value in model.parameters())
files = []
for path in sorted(export.rglob('*')):
    if path.is_file():
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        files.append({'path': str(path.relative_to(export)), 'size': path.stat().st_size, 'sha256': digest})
result = {'status': 'PASS', 'boundary': 'CPU strict load, finite weights and file hashes; no inference or evaluation',
          'step': 400, 'config_name': cfg.name, 'files': files}
(out / 'validation/export-step400.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
