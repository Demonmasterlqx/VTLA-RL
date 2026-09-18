# Repository-relative Tabero_X and T2-VLA runtime protocol

Read this reference when a policy test needs detailed preflight, launch, compatibility, reset, or artifact procedures. Angle-bracket values are resolved from the user request and current checkout; never replace them with a historical model, dataset, host path, or result directory.

## Parameter resolution

Resolve WORKSPACE_ROOT and record:

- user-selected policy/checkpoint directory, config, normalization assets, provenance, and SHA-256;
- registered environment, domain, suite, Task IDs, control mode, prompt/adverb, reset source, and episode protocol;
- current T2-VLA, Tabero_X, and IsaacLab commits and the runtime selected for each component;
- physical GPU assignments, process-local mapping, ports, output directory, and cleanup owner.

Never default a Task ID, model, dataset, prompt, or reset file. If task scope is missing, inspect the selected suite configuration and request it before allocating Isaac. Treat unspecified episode/step values as a connectivity smoke.

List valid tasks after substituting the selected domain and suite:

    jq '.tasks[] | {task_id, task_name, language_instruction}' \
      "$WORKSPACE_ROOT/Tabero_X/benchmarks/datasets/<domain>/config/<task_suite>.json"

Inspect the current batch evaluator under Tabero_X/scripts/tools/evaluation before building a manual loop. Use it only when its task selection, interpreter, reset semantics, and artifact layout match the resolved contract.

## Runtime and environment isolation

Use the environments required by the user-provided AGENTS.md:

    uv run --project "$WORKSPACE_ROOT/T2-VLA" ...
    conda run --no-capture-output -n tabero ...

If the checked-out project documents another invocation, follow that source and record sys.executable, package locations, and the command used. Do not embed a host-specific interpreter path. Clear inherited PYTHONPATH for isolation checks, then add only current-checkout source roots.

Map server and client GPU resources according to the selected runtime. Keep physical IDs, process-local IDs, and renderer IDs distinct. RLinf training GPU allocation remains a YAML configuration concern and must not be replaced with CUDA_VISIBLE_DEVICES.

Resolve any Vulkan ICD value from the current environment or explicit user input; do not prescribe a system path.

## Model and compatibility gate

Before allocating Isaac:

1. Resolve a checkpoint directory containing the expected weight files and inspect its parent if the user names a file.
2. Hash actual weights. If export metadata exists, require its hash and inspect source checkpoint, method, step/finality, precision, dataset provenance, action dimensions, and tactile metadata.
3. Inspect the selected T2-VLA config for model type, transforms, action horizon/dimension, normalization asset ID, tactile history, force history, and marker-motion shape.
4. Resolve and hash the exact normalization file the loader will use. Stop if absent or unproven.
5. Inspect the selected Tabero_X observation builder and environment sensors. Compare camera keys/sizes, state fields, tactile/force/marker history, and history lengths with the policy transform.
6. Compare policy output with the selected control mode and record executed versus unused dimensions.
7. Do not send inference while any shape, semantic, normalization, gripper, or control mapping remains unproved.

Inspect current source rather than relying on this reference when it has changed:

    git -C "$WORKSPACE_ROOT/Tabero_X" status --short
    git -C "$WORKSPACE_ROOT/T2-VLA" status --short
    rg -n 'name=|tactile|norm_stats|action_horizon' \
      "$WORKSPACE_ROOT/T2-VLA/src/openpi/training/config.py"
    rg -n 'resolve_openpi_runtime|control_mode|debug_mode' \
      "$WORKSPACE_ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py"

## HDF5 and reset gate

Keep these concepts separate:

- HDF5_TRAJ_SOURCE_DIR resolves task/environment assets.
- --hdf5_folder supplies episode reset data.

Before using a reset HDF5, verify that it opens, contains /data, has usable demo groups, has parseable environment arguments, matches the selected domain/suite/task/environment, and contains compatible initial states. File-name matching alone is insufficient.

If the protocol permits default reset and no compatible HDF5 is proven, create a run-local empty directory and record task_config_default_reset. If the user requires HDF5 resets, stop instead of substituting default reset.

## Launch templates

Create a run directory under the user-selected output root:

    <run-dir>/
    ├── server.log
    ├── server.pid
    ├── empty_hdf5/
    ├── raw/task_<id>/
    │   ├── client.log
    │   ├── client.pid
    │   └── artifacts/
    │       ├── run_meta.json
    │       └── exp_000/{exp_meta.json,forces.jsonl,preview.mp4}
    └── summary.json

Server template after resolving placeholders:

    env CUDA_VISIBLE_DEVICES="<server-physical-gpu>" \
      JAX_PLATFORMS=cuda \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONUNBUFFERED=1 \
      uv run --project "$WORKSPACE_ROOT/T2-VLA" \
      "$WORKSPACE_ROOT/T2-VLA/scripts/serve_policy.py" \
      --port <port> \
      policy:checkpoint \
      --policy.config=<t2-vla-config> \
      --policy.dir=<checkpoint-directory> \
      > <run-dir>/server.log 2>&1 &

Store the PID, poll the exact listening signal, and confirm the process is alive. On early exit, preserve the log and diagnose compatibility before starting Isaac.

Client template after resolving placeholders and selecting the current client entrypoint:

    env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY \
      CUDA_VISIBLE_DEVICES=<client-physical-gpu> \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      PYTHONUNBUFFERED=1 \
      HDF5_TRAJ_SOURCE_DIR="$WORKSPACE_ROOT/Tabero_X/benchmarks/datasets/<domain>/assembled_hdf5" \
      PYTHONPATH="$WORKSPACE_ROOT/Tabero_X/source/tac_manip:$WORKSPACE_ROOT/IsaacLab/source/isaaclab:$WORKSPACE_ROOT/IsaacLab/source/isaaclab_tasks" \
      conda run --no-capture-output -n tabero python -u \
      "$WORKSPACE_ROOT/Tabero_X/benchmarks/openpi/openpi_inference_client.py" \
      --server_host 127.0.0.1 \
      --server_port <port> \
      --task <environment-id> \
      --control_mode <control-mode> \
      --task_domain <domain> \
      --task_suite <task-suite> \
      --task_id <task-id> \
      --hdf5_folder <audited-hdf5-folder-or-run-empty-folder> \
      --num_total_experiments <episodes> \
      --max_inference_steps <chunks> \
      --replan_steps <steps-per-chunk> \
      --num_success_steps <success-hold> \
      --seed <seed> \
      --prompt_seed <prompt-seed> \
      --debug_mode <capture-mode> \
      --debug_path <run-dir>/raw/task_<task-id> \
      --debug_run_dir <run-dir>/raw/task_<task-id>/artifacts \
      --device cuda:0 \
      --headless \
      > <run-dir>/raw/task_<task-id>/client.log 2>&1

Verify supported flags with the current client help or dataclass before launch. Stop the stored server PID gracefully after clients finish, then verify port and GPU cleanup.

## Artifact validation and reporting

Run from the resolved workspace root:

    PYTHONDONTWRITEBYTECODE=1 python3 \
      "$WORKSPACE_ROOT/.codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py" \
      <run-dir>
    PYTHONDONTWRITEBYTECODE=1 python3 \
      "$WORKSPACE_ROOT/.codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py" \
      --json <run-dir> > <run-dir>/artifact_validation.json

Require finalized metadata, contiguous frame/action/replan fields, equal force/environment/video counts, decoded video agreement, consistent task identity, and run-level totals. Reconcile summary.json from per-episode metadata and state the sample-size boundary.

Create the required Record/experiments report with user-selected task/model parameters, result paths, exact weight hashes, commands, logs, duration, reset provenance, compatibility decisions, warnings, repository state, and W&B/TensorBoard status. For direct inference without training telemetry, state that those backends were not created.

## Failure handling

- Preserve every failed command and log with an attempt suffix.
- Exclude failed preflight, server load, incompatible inference, and partial episodes from success metrics.
- Stop at compatibility errors and report expected versus actual contracts.
- Treat a nonzero exit after complete artifacts as a teardown warning only when validation and client summaries pass.
- Never use broad process-kill patterns or mutate checkpoint, adapter, evaluator, or environment code to obtain a pass.
