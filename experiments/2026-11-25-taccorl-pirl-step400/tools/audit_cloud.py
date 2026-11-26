"""Compare the joint PPO/SFT cloud metrics with TensorBoard."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import swanlab
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = Path(os.environ['TACCORL_RUN_DIR'])
kind, target = sys.argv[1:3]
step = int(target)
run_id = json.loads((out / '_launcher/run_ids.json').read_text())[kind]
name = out.parent.name if kind == 'train' else f'{out.parent.name}-{kind}'
run = swanlab.Api().run(f'Demonmaster/VTLA-RL/{run_id}')
assert run.name == name
events = EventAccumulator(str(out / kind / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
keys = ['train/ppo_loss', 'train/sft_loss', 'train/sft_weighted_loss',
        'train/multi_task/vitasoy/valid_samples', 'train/multi_task/coca_cola/valid_samples']
response = run.metrics(keys=keys, range_query={'start': 0, 'end': step - 1})
cloud = {item['key']: item['metrics'] for item in response['list']}
assert set(cloud) == set(keys)
summary = {}
for key in keys:
    points = cloud[key]
    tb = [event for event in events.Scalars(key) if event.step < step]
    assert [event.step for event in tb] == [point['step'] for point in points] == list(range(step)), key
    for expected, actual in zip(tb, points):
        assert math.isfinite(actual['value']), key
        assert math.isclose(expected.value, actual['value'], rel_tol=1e-6, abs_tol=1e-9), (key, expected.step)
    summary[key] = {'count': len(points), 'last': points[-1]['value']}
assert run.state == 'FINISHED', run.state
result = {'status': 'PASS', 'checked_at': datetime.now(timezone.utc).isoformat(),
          'matched_updates': step, 'run_name': name, 'run_id': run_id,
          'run_state': run.state, 'url': run.url, 'metrics': summary}
(out / f'validation/{kind}-cloud-step{step}.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
