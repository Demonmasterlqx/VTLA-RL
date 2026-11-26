"""Audit the actual container, datasets and resolved two-stage contract."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

root = Path('/root/VTLA-RL')
out = Path(os.environ['TACCORL_RUN_DIR'])
manifest = json.loads((out / 'image/source_manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    prefix = 'RLinf/examples/embodiment/config/'
    if name.startswith(prefix):
        assert hashlib.sha256((out / 'config' / name[len(prefix):]).read_bytes()).hexdigest() == expected, name
    tool_prefix = 'experiments/2026-11-25-taccorl-pirl-step400/tools/'
    if name.startswith(tool_prefix) and (out / 'tools' / Path(name).name).is_file():
        assert hashlib.sha256((out / 'tools' / Path(name).name).read_bytes()).hexdigest() == expected, name
model = Path(os.environ['TACCORL_PIRL_MODEL_PATH'])
for path, digest in [(model / 'model.safetensors', '8fb715689772433d7a321e14d03fbfc266cbc11cc6636a84fab7fa29db9778c7'),
                     (Path(os.environ['TACCORL_NORM_STATS_PATH']), 'c944fb4951d7968b6602edc4cf7ce0bbe8f098e40fed8bcbfb29358dbd25da61')]:
    with path.open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, path
dataset = Path(os.environ['TACCORL_REAL_SFT_DATA_PATH'])
episodes = [json.loads(line) for line in (dataset / 'meta/episodes.jsonl').read_text().splitlines()]
selected = [row for row in episodes if row['episode_index'] < 60]
assert len(selected) == 60 and sum(row['length'] for row in selected) == 8676
assert all('cookie' not in task.lower() for row in selected for task in row['tasks'])
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.ppo_multi_task import validate_config

stages = {}
with initialize_config_dir(version_base='1.1', config_dir=str(out / 'config')):
    for stage, target in [('stage1', 200), ('stage2', 400)]:
        cfg = compose(config_name=f'2026-11-25-taccorl-pirl-step400/{stage}')
        OmegaConf.resolve(cfg)
        validate_config(cfg)
        assert cfg.actor.fsdp_config.gradient_checkpointing
        assert cfg.actor.fsdp_config.use_orig_params
        assert not cfg.actor.model.openpi.discrete_state_input
        assert cfg.actor.enable_sft_co_train and cfg.algorithm.multi_task.enabled
        assert not cfg.runner.use_training_pipeline
        assert cfg.runner.max_steps == cfg.runner.max_epochs == target
        assert cfg.actor.micro_batch_size * 8 == cfg.actor.global_batch_size == 384
        assert dict(cfg.cluster.component_placement) == {'env': '0-7', 'actor': '0-7', 'rollout': '0-7'}
        assert cfg.env.train.total_num_envs == 128
        assert cfg.env.eval is None
        assert cfg.runner.val_check_interval == -1 and not cfg.runner.only_eval
        assert [t.name for t in cfg.env.train.multi_task.tasks] == ['vitasoy', 'coca_cola']
        assert all(t.init_params.success.normalize_effort_reward.enabled == (stage == 'stage2') for t in cfg.env.train.multi_task.tasks)
        data = get_openpi_config(cfg.actor.config_name, model_path=str(model), batch_size=384,
                                repo_id=str(dataset), data_kwargs=cfg.actor.model.openpi_data)
        assert list(data.data.create(data.assets_dirs, data.model).episodes) == list(range(60))
        OmegaConf.save(cfg, out / f'validation/{stage}_resolved.yaml')
        stages[stage] = {'target': target, 'placement': OmegaConf.to_container(cfg.cluster.component_placement)}
provenance = {name: importlib.util.find_spec(name).origin for name in ['openpi', 'rlinf', 'tac_manip']}
assert provenance['openpi'].startswith(str(root / 'T2-VLA/src'))
assert provenance['rlinf'].startswith(str(root / 'RLinf/rlinf'))
result = {'boundary': 'source/assets/config preflight; no rollout or update', 'stages': stages,
          'source_files': len(manifest), 'provenance': provenance, 'sft_episodes': 60, 'sft_frames': 8676,
          'cuda_devices': torch.cuda.device_count(), 'gradient_checkpointing': True}
assert result['cuda_devices'] == 8, result
(out / 'validation/preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
