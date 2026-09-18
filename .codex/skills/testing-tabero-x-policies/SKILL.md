---
name: testing-tabero-x-policies
description: Run, audit, and document T2-VLA policy inference in Tabero_X and IsaacLab with explicit checkpoint, environment, task, reset, runtime, and artifact contracts.
---

# Testing Tabero_X policies

Use this skill for policy smoke tests or formal evaluations against a user-selected local model/export and Tabero_X task suite. It is evaluation-only; do not start training or change model, adapter, evaluator, environment, or reset semantics to make a run pass.

## Establish the contract

Resolve the workspace root and record these values before launching Isaac:

1. Policy/checkpoint directory, config, normalization assets, source provenance, and SHA-256 hashes.
2. Registered environment, domain, suite, Task IDs, control mode, prompt/adverb, episode/step protocol, GPU mapping, port, and reset source.
3. Current `T2-VLA`, `Tabero_X`, and `IsaacLab` commits and the runtime selected for each component.

Never default a Task ID, model, dataset, prompt, or reset file. If the user did not provide task scope, inspect the selected suite configuration and request the missing scope before allocating Isaac. Treat unspecified episode/step values as a connectivity smoke, not a formal evaluation.

Read [references/runtime-protocol.md](references/runtime-protocol.md) when the run needs the detailed compatibility gate, launch template, or artifact protocol.

## Audit before launch

- Inspect the current Tabero_X client and existing evaluator scripts before writing a launcher. Reuse an existing script only when its task selection, environment, reset semantics, output layout, and arguments match the requested contract.
- Preserve unrelated and untracked repository changes; do not commit unless requested.
- Prove checkpoint identity from actual files and metadata. For merged exports, validate the root weights and export metadata rather than trusting a directory name.
- Resolve the T2-VLA config and normalization assets from source and checkpoint metadata. Stop on observation, action, tactile, normalization, control, or environment mismatches; do not add adapters, truncation, padding, resampling, or alternate statistics.
- Check the selected physical GPUs, target port, Python interpreters, Vulkan/runtime requirements, task config entry, assets, and HDF5 semantics with read-only commands.

## Execute and supervise

- Run the T2-VLA server with its local uv project and the Tabero_X/IsaacLab client through the conda environment named `tabero`; do not hard-code a host-specific interpreter path.
- Wait for the exact listening signal and verify that the server process is alive before starting a client.
- Keep environment/task asset resolution separate from the client's reset source. Use an explicit empty reset directory only when the resolved protocol permits default reset, and record that decision.
- Use the requested capture protocol, normally mode 6 when artifact auditing is required, but do not force it when the user selected another protocol.
- Stop only processes created for this run and verify port/GPU cleanup.

## Validate and report

Run `scripts/validate_mode6_artifacts.py <run-dir>` after clients finish. Require frame continuity, matching environment/force/video counts, complete metadata, consistent task identity, and reconciled per-episode totals. Derive success from episode artifacts, not connection counts or server logs.

Distinguish clean exit, post-artifact teardown warning, incomplete episode, compatibility failure, and policy failure. Write `summary.json` and a `Record/experiments/` report using the repository naming/content requirements: exact user-selected model/config/hashes, task scope, protocol, commands, logs, duration, reset provenance, per-episode evidence, warnings, repository state, and W&B/TensorBoard status. Do not create tests that depend on local models, datasets, or configuration files.
