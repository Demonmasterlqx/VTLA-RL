---
name: training-rlinf-t2vla-models
description: Govern RLinf or T2-VLA model training, including initial runs, strong startup supervision, persistent goal-tracked monitoring, continuation, restart, and failure recovery. Use when preparing, starting, supervising, resuming, restarting, or reporting training in these repositories; do not use for inference-only evaluation.
---

# Training RLinf and T2-VLA Models

This is a constraint and review skill. It intentionally contains no training command. Inspect the current repository entrypoints and configs before proposing how to launch a run, and prefer an existing launcher over creating a new one.

Use the local uv project required by `AGENTS.md`: `RLinf/.venv` for RLinf and `T2-VLA/.venv` for T2-VLA. Resolve paths from the current checkout or a user-supplied workspace variable; never embed a host-specific absolute path, model, dataset, or checkpoint.

## Classify the run

Before changing configuration or starting a process, classify the request:

- **Initial training**: no approved run contract exists for this experiment.
- **Continuation**: a complete, compatible checkpoint continues the same logical training run.
- **Restart or recovery**: a stopped or failed process is restarted as the same logical training run.
- **New experiment**: the model, data, task scope, method or loss, trainable modules, GPU count or distributed topology, effective batch size, optimizer or schedule, precision, seed, target-step semantics, or another result-affecting parameter changes materially.

Continuation and restart do not become new experiments merely because the process, host session, or launcher was restarted. A material contract change does become a new experiment and must pass the initial-training review gate.

## Gate initial training on review

For initial training, first inspect the actual entrypoint, resolved config, model/checkpoint, dataset, and repository state with read-only checks. Then present one explicit training contract to the trainer for review. Include:

- framework and method; task and dataset scope; base model or initialization checkpoint and their provenance;
- requested GPU count and mapping, distributed topology, and expected resource ownership;
- per-device or micro batch size, gradient accumulation, world size, and the resulting effective global batch size;
- target optimizer steps, epochs or sample budget, and the exact meaning of the configured step limit;
- optimizer, learning rate, scheduler, warmup, weight decay, gradient clipping, precision, and seed, and do not start training while this state is implicit or unresolved;
- trainable and frozen modules, LoRA or other adapter settings, action or sequence horizon, and relevant observation or tactile contracts;
- checkpoint, export, evaluation, logging, and retention intervals; output, checkpoint, log, TensorBoard, and W&B locations;
- one proposed W&B project, run name, and run ID strategy;
- completion criteria, required downstream evaluation, stopping conditions, and recovery boundary;
- the source file or metadata used to resolve every value, plus all missing, conflicting, or unverified fields.

Do not start formal training until the trainer explicitly approves this contract. Approval applies only to the reviewed values; resolve and re-submit any material change before launch. A connectivity or capacity smoke that consumes training resources also requires explicit scope and must not be presented as formal training.

For RLinf, configure GPU allocation through the applicable RLinf training config. Do not use `CUDA_VISIBLE_DEVICES` as a substitute for that configuration.

## Preserve the contract on continuation and restart

Before continuing or restarting, inspect the current launcher and worker state, GPU owners, last trustworthy log step, checkpoint metadata and completeness, output directories, and W&B/TensorBoard state. Do not infer resumability from a checkpoint directory name or a stale monitor.

Compare the proposed recovery against the last approved contract. Preserve the framework, model, dataset, task scope, trainable modules, GPU count, batch arithmetic, optimizer and scheduler state, precision, seed, target-step semantics, output lineage, and evaluation target. Confirm whether the framework interprets the target as a total step or additional steps.

A continuation requires a checkpoint that contains the state the selected framework needs to resume correctly. If optimizer, scheduler, global-step, RNG, or other required state is absent, report the exact boundary; do not call a weights-only load a faithful continuation. Diagnose failures before restarting, retry the same approved specification first when recovery is safe, and never silently reduce batch size, change precision, switch data, or alter the model to make a run proceed. Stop and report a hard or repeated blocker.

## Use one W&B identity per logical training run

Choose exactly one W&B run name before initial launch and include it in the approved contract. The same logical training run must keep that name through every continuation, restart, retry, and checkpoint phase.

- Do not create suffix variants such as `resume`, `restart`, a new timestamp, or an attempt number for the same logical run.
- Reuse the original W&B run ID and resume that run when the integration supports it; do not silently create another W&B run with the same or a different name.
- Ensure only the designated logging process initializes the top-level run. Distributed ranks, Ray workers, and restarted launchers must not create independent W&B names or runs.
- If the original run ID cannot be recovered or the backend cannot resume it safely, stop before creating a replacement and ask the trainer how to preserve the experiment lineage.
- A materially changed experiment may use a new name only after it passes the initial-training review gate.

The Record report must contain the single W&B name, run ID, and link for the logical training run. Restarts and continuations belong under that same entry, not separate experiment links.

## Supervise an approved run

Once authorized, launch every training process inside a named `tmux` session so that it remains attached to a durable, inspectable supervisor context. Record the `tmux` session name and pane, verify the launcher is actually running under that session, and use the same session or clearly documented replacement session for same-spec recovery. Do not launch formal training only in a transient foreground shell, detached shell without a durable session, or bare background process.

Before executing the launch command, restate the resolved `gradient_checkpointing` state as `enabled` or `disabled` and verify that the effective config or command matches it. If no explicit choice was approved, use `disabled`; never infer that it is enabled from framework defaults or enable it silently as a memory-saving recovery change.

Supervise training from launch to a verified terminal state. Use both TensorBoard and W&B, confirm that their identities and step axes match the approved contract, and watch process health, GPU ownership, finite loss, throughput, global-step progress, checkpoint creation, and disk state. Do not launch training and immediately return control to the trainer with only a command or log path.

### Strongly supervise the first 100 steps

From process launch until the trustworthy global step reaches at least 100, remain in a continuous poll-and-check loop:

- Poll every 5–30 seconds. Start near 5 seconds while workers initialize, after recovery, or when any signal is uncertain; widen gradually toward 30 seconds only while progress is healthy. Do not use intervals longer than 30 seconds during this phase.
- At every poll, inspect deltas rather than repeatedly dumping full logs: launcher and worker liveness, owned GPU processes and utilization, latest global step and timestamp, finite loss and key metrics, throughput or step-time trend, new warnings or tracebacks, disk capacity, and checkpoint activity when applicable.
- Treat a step as trustworthy only when it is supported by the active logs or telemetry for the current process lineage. Do not count stale W&B/TensorBoard points, pre-restart logs, or a directory name as current progress.
- Do not declare the run stable merely because the process remains alive. Stability requires reaching at least step 100 with continued step advancement, finite metrics, expected GPU ownership, no unresolved fatal or repeated error, and no evidence that logging or checkpoint state has diverged from the approved contract.
- If training stalls, becomes non-finite, loses workers, exits, or shows a resource fault, stop widening the interval and diagnose immediately under the approved recovery boundary.

### Track stable training with a goal

After the run satisfies the stability criteria, create a goal for the already authorized continuing supervision. The goal objective must identify the logical run, terminal training criterion, required final artifact checks, and downstream evaluation. Do not create a second goal for the same run or use goal creation to broaden the approved training contract.

While that goal is active, the agent must continue supervising rather than ending after a single healthy observation:

- Poll with sleep-based delta checks every 1–5 minutes. Use intervals near 1 minute after entering the stable phase, after a restart, near a checkpoint or terminal boundary, or when health is uncertain; widen toward 5 minutes only for consistently healthy long-running segments. Never exceed 5 minutes unless the trainer explicitly changes the monitoring contract.
- Keep each check lightweight but verify current step progress, process and GPU health, finite metric trends, fresh log timestamps, checkpoint or export progress, disk state, and new errors. A sleeping interval is part of supervision, not a reason to yield or declare success.
- Continue the poll, sleep, and check cycle across goal continuations until the verified terminal state is reached, the trainer explicitly pauses or stops the run, or a hard blocker requires termination. Do not mark the goal complete because the monitoring turn, shell command, launcher, or one worker ended.
- Mark the goal complete only after the training terminal criterion, final checkpoint or export integrity, TensorBoard/W&B consistency, and required downstream evaluation are all verified. If the run is explicitly stopped, record it as user-stopped rather than completed. Use a blocked outcome only when progress is genuinely impossible under the approved recovery boundary, not merely because the run is long.

On failure, preserve the command, logs, telemetry, checkpoint state, and attempt history before recovery. On an explicit pause or stop request, interrupt the owned launcher process group gracefully, preserve artifacts, and verify that its workers and GPUs are released without affecting unrelated jobs.

Do not declare completion from a launcher exit alone. Verify the target step or other terminal criterion, final checkpoint completeness and finality, expected exports, TensorBoard/W&B consistency, and the requested downstream evaluation.

## Record the run

Create or update the appropriate `Record/experiments/` report using the repository naming convention. Record the approved contract, actual launch command, repository state, all attempts, start/end time and duration, logs, checkpoints and exports, the single W&B identity/link, TensorBoard path, failures and recoveries, final status, and evaluation evidence. Keep failed or user-stopped training distinct from successfully completed training.
