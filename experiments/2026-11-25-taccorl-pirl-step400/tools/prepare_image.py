"""Freeze current training sources for a 0.11.0 incremental build."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).absolute().parents[3]
out = root / 'results/2026-11-25-taccorl-pirl-step400/8gpu'
(out / 'validation').mkdir(parents=True, exist_ok=True)
(out / 'tools').mkdir(parents=True, exist_ok=True)
for name in ('preflight.py', 'audit_checkpoint.py', 'audit_cloud.py', 'audit_export.py', 'poll.py'):
    shutil.copy2(Path(__file__).absolute().parent / name, out / 'tools' / name)
context = out / 'image/context'
workspace = context / 'workspace'
if workspace.exists():
    shutil.rmtree(workspace)
scope = {
    'RLinf': ('rlinf/', 'examples/', 'tests/', 'toolkits/'),
    'T2-VLA': ('src/', 'scripts/', 'packages/openpi-client/src/'),
    'Tabero_X': ('source/tac_manip/', 'benchmarks/openpi/', 'benchmarks/datasets/realworld/config/'),
    '.': ('docker/env_setup/', 'experiments/2026-11-25-taccorl-pirl-step400/'),
}
suffixes = {'.py', '.sh', '.yaml', '.yml', '.json', '.toml', '.md', '.txt'}
manifest = {}
for repo, prefixes in scope.items():
    names = subprocess.check_output(['git', '-C', str(root / repo), 'ls-files', '-co', '--exclude-standard', '-z']).decode().split('\0')
    for name in sorted(set(names)):
        if not name.startswith(prefixes) or Path(name).suffix not in suffixes:
            continue
        source = root / repo / name
        if not source.is_file() or source.is_symlink() or source.name.startswith('.env'):
            continue
        relative = source.relative_to(root)
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
base = json.loads((root / 'results/docker-0.11.0-20261124/source-manifest.json').read_text())
removed = []
for entry in base['files']:
    path = entry['path']
    if Path(path).suffix not in suffixes or path in manifest:
        continue
    for repo, prefixes in scope.items():
        full_prefixes = prefixes if repo == '.' else tuple(f'{repo}/{p}' for p in prefixes)
        if path.startswith(full_prefixes) and not (root / path).exists():
            removed.append(path)
(out / 'image/source_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
(context / 'source_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
revisions = {
    repo: subprocess.check_output(['git', '-C', str(root / repo), 'rev-parse', 'HEAD'], text=True).strip()
    for repo in ('.', 'RLinf', 'T2-VLA', 'Tabero_X', 'IsaacLab')
}
(context / 'source_revisions.json').write_text(json.dumps(revisions, indent=2))
(out / 'image/source_revisions.json').write_text(json.dumps(revisions, indent=2))
(context / 'removed_sources.json').write_text(json.dumps(removed))
(context / 'Dockerfile').write_text('''FROM ccr.ccs.tencentyun.com/vtla/vtla@sha256:d58ed1129006bd43d4fdf78bf89d349b8b4266e2dcfcb669b0ccfb9cc08031e4
COPY workspace/ /root/VTLA-RL/
COPY source_manifest.json source_revisions.json /opt/vtla/
COPY removed_sources.json /tmp/vtla-removed-sources.json
RUN /root/VTLA-RL/RLinf/.venv/bin/python -c "import json,pathlib; root=pathlib.Path('/root/VTLA-RL'); [(root/p).unlink(missing_ok=True) for p in json.load(open('/tmp/vtla-removed-sources.json'))]" \\
    && cp -a /root/VTLA-RL/T2-VLA/src/openpi/models_pytorch/transformers_replace/. /root/VTLA-RL/RLinf/.venv/lib/python3.11/site-packages/transformers/ \\
    && cp -a /root/VTLA-RL/T2-VLA/src/openpi/models_pytorch/transformers_replace/. /root/VTLA-RL/T2-VLA/.venv/lib/python3.11/site-packages/transformers/ \\
    && rm /tmp/vtla-removed-sources.json \\
    && mkdir -p /isaac-sim/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default
LABEL org.vtla.experiment="2026-11-25-taccorl-pirl-step400" org.vtla.base="0.11.0"
''')
config = out / 'config'
if config.exists():
    shutil.rmtree(config)
shutil.copytree(root / 'RLinf/examples/embodiment/config', config, ignore=shutil.ignore_patterns('__pycache__'))
print(json.dumps({'source_files': len(manifest), 'removed': removed, 'bytes': sum((workspace / p).stat().st_size for p in manifest)}))
