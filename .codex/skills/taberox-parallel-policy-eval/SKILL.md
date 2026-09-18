---
name: taberox-parallel-policy-eval
description: Run and supervise parallel T2-VLA and Tabero_X policy evaluations with explicit user-selected policies, tasks, GPU assignments, and artifact validation.
---

# Parallel T2-VLA and Tabero_X policy evaluation

Use this skill for parallel policy-server/client evaluations. It provides a reusable execution and evidence protocol; it must not select a model, dataset, task, prompt, environment, asset revision, or historical result for the user.

## Resolve the run contract

Before launching anything, resolve and record these user-supplied values:

- `WORKSPACE_ROOT` and the checked-out `T2-VLA`, `Tabero_X`, and `IsaacLab` directories;
- `POLICY_DIR`, policy config, normalization assets, and their SHA-256/provenance;
- `ENV_ID`, domain, suite, Task IDs, prompts/adverbs, control mode, and reset source;
- server/client GPU assignments, ports, worker count, episode count, seed policy, and step/replan limits;
- output `RUN_DIR`, logging paths, and the required Record report name.

Do not infer a task from a numeric ID or a model from a directory name. Inspect the selected suite configuration, client arguments, policy metadata, observation/action/tactile contracts, and reset semantics first. If task scope is missing, stop before allocating Isaac and request it.

## Repository-relative runtime

Run commands from `WORKSPACE_ROOT` or the relevant component directory. Use the environments required by the user-provided `AGENTS.md`:

- T2-VLA server: `uv run --project T2-VLA ...`;
- Tabero_X/IsaacLab client: `conda run --no-capture-output -n tabero ...`, following the checked-out client instructions.

Use variables such as `$WORKSPACE_ROOT`, `$POLICY_DIR`, `$RUN_DIR`, `$SERVER_GPU`, `$CLIENT_GPU`, and `$PORT`; never embed a host-specific absolute path. If a Vulkan ICD, tactile backend, or launcher helper is required, discover it from the current runtime or accept it as an explicit user value and record it. Do not prescribe a machine-specific system path.

For evaluation clients, process-local GPU selection may be needed by the current Isaac runtime. Keep physical GPU assignment, process-local `cuda:0`, and renderer selection distinct, and verify the actual mapping before launch. This rule does not override the RLinf training rule that GPU placement belongs in the RLinf configuration.

## Parallel execution

Each worker owns one policy server, one client, one port, and one result subtree. Use unique ports and result directories. Different workers may run concurrently; tasks sharing a server run sequentially unless the selected client explicitly supports another schedule.

Before launch:

1. Check free GPU/port ownership and available disk capacity without evicting unrelated processes.
2. Verify the policy directory, metadata, normalization files, environment registration, selected assets, and reset source.
3. Create the run and shard directories. Preserve failed attempts under distinct attempt names.
4. Start the server and wait for its exact listening signal and live PID.
5. Start clients only after the server and compatibility gates pass.

Do not use a fixed episode budget or frame limit unless it is part of the resolved user protocol. Report configured limits separately from actual environment steps and wall-clock duration.

## Supervision and artifact validation

Track each worker's PID, port, GPU ownership/utilization, latest log progress, completed episodes, current action steps, and disk state. A listening server, partial video, or clean process exit is not evaluation success.

After every shard, run the repository validator with a repository-relative command such as:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 .codex/skills/testing-tabero-x-policies/scripts/validate_mode6_artifacts.py \
  "$SHARD_ARTIFACTS" --json > "$SHARD_VALIDATION"
```

Require complete metadata, contiguous frame/action/replan records, matching force/video/environment counts, and consistent task identity. Apply any additional protocol-specific checks from the resolved contract; do not assume the validator covers them.

At run completion, reconcile per-episode artifacts into `summary.json`, separate infrastructure failures from policy failures, and record model/config hashes, task scope, prompts, seeds, GPU/port assignments, commands, logs, duration, warnings, cleanup evidence, and TensorBoard/W&B status in `Record`. Stop only processes created for this run and verify that their ports and GPUs are released.

Do not change checkpoint, adapter, evaluator, environment, or reset semantics merely to obtain a successful launch.
