# VTLA

VTLA is an RL research project focused on integrating tactile sensing into reinforcement learning models.

## Repository Structure

* `T2-VLA`: OpenPI server, mainly used for testing.
* `Tabero_X`: IsaacLab environments and evaluation scripts.
* `RLinf`: Main RL training framework, with support for RL, LoRA, RLT, etc. It already includes IsaacLab/IsaacSim integrations.
* `Record`: Experiment log repository. Every training and evaluation run must be documented.
* `results`: Training and evaluation outputs.
* `models`: Baseline model files.
* `IsaacLab`: IsaacLab installation used for VTLA evaluation.

## Python Environments

* `T2-VLA` and `RLinf`: use the local `uv` environment inside each repository.
* `Tabero_X` / `IsaacLab`: use the local `uv` environment under `IsaacLab`.

## Execution Environment

Training and evaluation may run either on the host or inside a provided container.

Before launching any training or evaluation, explicitly confirm with the user whether it should run on the **host** or in the **container**.

If running inside the container:

* Verify that the code inside the container is synchronized with the local repository.
* If the container code differs from the local code, do **not** proceed automatically.
* Clearly report the mismatch and confirm with the user which version should be used or whether synchronization should be performed first.

## Experiment Records

Each training or evaluation run must have a report under `Record`.

Naming format:

`full_time-taskid-adverb-train_method-annotation.md`

Each report should include:

* Training parameters
* Results and detailed result paths
* Model checkpoint / weight paths
* Launch command
* Training logs
* Important events and observations during the run
* Total training duration
* W&B links for every experiment when multiple runs are involved

## Training and Evaluation

* Before running an evaluation, check whether an existing script already supports the task. Prefer existing scripts whenever possible.
* Training runs must be monitored throughout execution, with logs properly recorded.

## Code Changes

* Use both TensorBoard and Weights & Biases as logging backends.
* Minimize code changes. Check existing implementations before adding new code.
* Avoid adding unnecessary scripts to repositories.
* If a script is needed specifically for a training experiment or result analysis, place it under that experiment's corresponding directory in `results`.
* Temporary analysis scripts should also stay inside the relevant experiment directory instead of being added to the main repository structure.
* Do not create documentation for highly specialized or one-off changes.
* Do **not** write tests for configuration files, local models, or datasets.
* Tests should not depend on local configuration, model files, or datasets.
* code in IsaacLab is not allowed to modified

## Notes

* Do **not** use `CUDA_VISIBLE_DEVICES` to control GPU allocation in `RLinf`. Configure GPU usage through the corresponding RLinf training configuration.
* When downloading models or assets from Hugging Face, run `source .env` and use the provided `$HF_TOKEN`.
