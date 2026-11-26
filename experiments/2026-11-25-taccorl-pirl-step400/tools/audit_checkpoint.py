"""Validate a completed joint PPO/SFT checkpoint and its logged updates."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = Path(os.environ['TACCORL_RUN_DIR'])
checkpoint, kind, step_text = sys.argv[1:4]
checkpoint = Path(checkpoint)
step = int(step_text)
actor = checkpoint / 'actor'
task = json.loads((checkpoint / 'multi_task_state.json').read_text())
joint = json.loads((actor / 'sft_training_state.json').read_text())
cursor = json.loads((actor / 'data_state.json').read_text())
assert task['step'] == joint['step'] == step
assert task['names'] == ['vitasoy', 'coca_cola']
assert joint['contract']['data_kwargs']['episodes'] == list(range(60))
reader = FileSystemReader(actor / 'dcp_checkpoint')
metadata = reader.read_metadata()
coverage = Counter(key.split('.')[1] for key in metadata.state_dict_metadata)
assert all(coverage[name] for name in ['model', 'optimizers', 'lr_schedulers', 'rng'])
for storage in metadata.storage_data.values():
    shard = actor / 'dcp_checkpoint' / storage.relative_path
    assert shard.stat().st_size >= storage.offset + storage.length, shard
optimizer_steps = {name: torch.empty(meta.size, dtype=meta.properties.dtype)
                   for name, meta in metadata.state_dict_metadata.items()
                   if '.optimizers.state.' in name and name.endswith('.step') and isinstance(meta, TensorStorageMetadata)}
assert optimizer_steps
load(optimizer_steps, storage_reader=reader)
counts = Counter(float(value.item()) for value in optimizer_steps.values())
assert min(counts) > 0
weights = torch.load(actor / 'model_state_dict/trainable_weights.pt', map_location='cpu', weights_only=False)
assert weights['metadata']['global_step'] == step
assert weights['metadata']['discrete_state_input'] is False
assert weights['metadata']['gradient_checkpointing'] is True
assert weights['metadata']['world_size'] == 8
assert all(torch.isfinite(t).all() for t in weights['model'].values())
events = EventAccumulator(str(out / kind / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
metrics = {key: [{'step': e.step, 'value': e.value} for e in events.Scalars(key)] for key in events.Tags()['scalars']}
for key in ['train/ppo_loss', 'train/sft_loss', 'train/sft_weighted_loss']:
    points = [p for p in metrics[key] if p['step'] < step]
    assert {p['step'] for p in points} == set(range(step)), key
    assert all(torch.isfinite(torch.tensor(p['value'])) for p in points), key
for name in task['names']:
    assert any(p['value'] > 0 for p in metrics[f'train/multi_task/{name}/valid_samples'])
if kind == 'smoke':
    assert set(counts) == {float(10 * step)}
    batches_per_epoch = 8676 // (8 * 48)
    expected_epoch, last_batch = divmod(10 * step - 1, batches_per_epoch)
    assert (cursor['data_epoch'], cursor['data_iter_offset']) == (expected_epoch, last_batch + 1)
    if step == 3:
        prior = json.loads((out / 'validation/smoke-step2.json').read_text())
        assert prior['sft_training_state']['contract'] == joint['contract']
        before = torch.load(checkpoint.parent / 'global_step_2/actor/model_state_dict/trainable_weights.pt', map_location='cpu', weights_only=False)['model']
        assert any(not torch.equal(value, before[name]) for name, value in weights['model'].items())
files = []
for path in sorted(checkpoint.rglob('*')):
    if path.is_file():
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        files.append({'path': str(path.relative_to(checkpoint)), 'size': path.stat().st_size, 'sha256': digest})
result = {'step': step, 'checkpoint': str(checkpoint), 'coverage': dict(coverage),
          'optimizer_steps': dict(counts), 'sft_training_state': joint, 'data_state': cursor,
          'trainable_metadata': weights['metadata'], 'metrics': metrics, 'files': files}
(out / f'validation/{kind}-step{step}.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'checkpoint': str(checkpoint), 'step': step, 'optimizer_steps': dict(counts), 'status': 'PASS'}))
