---
name: taberox-rlinf-t2-vla-env-config
description: Configure the repository's RLinf, T2-VLA, Tabero_X, and IsaacLab environments with local uv projects, explicit user-supplied assets, and staged runtime validation.
---

# RLinf, T2-VLA, Tabero_X, and IsaacLab environment setup

Use this skill when the workspace needs environment setup or reproducibility checks. Keep the workflow repository-relative and parameterized: the skill must not choose a dataset, model, checkpoint, host path, GPU layout, or experiment for the user.

## Repository contract

Resolve the workspace root from the current checkout or from an explicit `WORKSPACE_ROOT` supplied by the user. Do not write a machine-specific absolute path into commands or reports. Expected sibling directories are:

```text
WORKSPACE_ROOT/
├── RLinf/
├── T2-VLA/
├── Tabero_X/
├── IsaacLab/
├── models/        # optional, user-managed
├── datasets/      # optional, user-managed
├── results/
└── Record/
```

Preserve the case-sensitive `IsaacLab` name. Inspect the current checkout, branch/commit, existing links, and dirty state before changing anything.

## Preconditions

- `uv` is available and the required Isaac Sim/runtime prerequisites are documented by the checked-out repositories.
- If Hugging Face access is needed, run `source .env` from the relevant workspace and use the supplied `$HF_TOKEN`. Never request or record a private key.
- The user supplies any `DATASET_ID`, `MODEL_ID`, local checkpoint, asset revision, task, or training configuration required for the requested validation.
- Do not infer that an asset, model, or dataset is valid merely because a directory exists. Record the source, revision, hashes, and the exact validation performed.

## Environment setup

Use the local uv project for each component:

- `T2-VLA/.venv` for T2-VLA server work;
- `RLinf/.venv` for RLinf work;
- the user-provided conda environment `tabero` for Tabero_X/IsaacLab work, following the checked-out `IsaacLab` and `Tabero_X` installation instructions.

Prefer `uv run --project <component>` or the component's local interpreter for T2-VLA and RLinf. Run Tabero_X/IsaacLab through the conda environment named `tabero`; do not hard-code its installation path or invoke an unrelated global Python. Clear inherited `PYTHONPATH` for isolation checks, then add only source roots explicitly required by IsaacLab.

For RLinf, configure GPU placement in the selected RLinf YAML (`cluster.component_placement` and related fields). Do not use `CUDA_VISIBLE_DEVICES` as a replacement for RLinf placement.

## Staged acceptance

Report each stage separately; a later stage must not be claimed from an earlier one:

1. Verify `sys.executable`, Python, package locations, Torch/CUDA, and uv dependency health in each required project.
2. Verify the IsaacLab/Isaac Sim interpreter and a minimal import/startup path using the repository's current instructions.
3. Run the repository-provided asset checker when assets are in scope. Record the requested asset ID/revision and result.
4. If a user-supplied policy is in scope, resolve its config, weights, normalization, observation, action, and tactile contracts before starting a server.
5. Distinguish server load, client connectivity, one valid action, and closed-loop task execution. Add a finite timeout to connectivity checks.
6. For RLinf PIRL/DSRL/RLT smoke work, use the user's explicit configs, model/data paths, GPU placement, update limit, timeout, and success criteria. RLT Stage 2 must identify its Stage 1 input.

## Reporting and boundaries

Record commands, interpreter paths, repository commits, resolved user inputs, hashes, logs, warnings, and the exact acceptance stage in `Record` when the activity is a training or evaluation run. Use the repository naming convention from `AGENTS.md`.

Do not download a default model or dataset, start a formal training run, alter unrelated code, or create experiment-specific scripts as part of environment setup. Do not write tests that depend on local configuration, models, datasets, or hardware.
