# CS:Source Aim-Trajectory Data Collection And Analysis Pipeline

BSc Critical Project 6: transformer-based classification of human, smooth-assist,
and humanised-assist aim trajectories from Counter-Strike: Source telemetry.

The final dissertation analysis uses participant-held-out evaluation on
AK-47-only pre-fire windows. Questionnaire data is processed separately and is
not used as model input.

## Architecture

```
CS:Source dedicated server (srcds_linux + Tickrate_Enabler → 100 Hz)
    ├── Tickrate_Enabler.so              Patches the engine 66.7 Hz cap
    ├── cs_aim_telemetry.smx             HTTP POST every tick → Flask :3000
    └── cs_aim_live_controller.smx       Primary aimbot — runs in-process
                                         on OnPlayerRunCmd, zero round-trip

Python pipeline (capture/)
    ├── telemetry_server.py              Flask receiver → sort/dedupe → JSON
    ├── session_orchestrator.py          Session runner (labels, console hints)
    └── bot_profiles/                    YAML parameter sets (kept for the
                                         dissertation — documents the bot
                                         design space and the legacy
                                         Python-TCP aim path)

Data
    └── sessions/                        Per-session events JSON

Analysis (analysis/)
    ├── audit_tickrate.py                Capture-quality gate (--active-only)
    ├── plot_trajectories.py             Visual inspection (--fire-window)
    ├── features.py                      33-dim handcrafted feature extractor
    ├── preprocess_sequences.py          Events JSON → model-ready windows
    ├── run_baselines.py                 Classical baseline models
    ├── train_transformer.py             PyTorch Transformer Encoder training
    ├── evaluate_transformer.py          Metrics and plots from checkpoints
    ├── questionnaire_summary.py         Subjective questionnaire summaries
    └── labels.py                        Filename → SessionMeta parser
```

The **SourceMod-native controller** (`cs_aim_live_controller.smx`) is the active
aimbot for the study. It runs inside the server's `OnPlayerRunCmd` hook, so the
aim correction lands on the same tick as the position read — no Python
round-trip latency. The legacy Python TCP loop (`cs_aim_override.smx` +
`bot_aim_generator.py`) is kept in the repo for reference and parameter
documentation but is **not** the path used for participant recordings.

## Fresh-Machine Setup

The canonical replication guide is **[SETUP.md](SETUP.md)**. It is structured
as numbered steps with a **Verify** block after each one — if your output
doesn't match the expected output, stop and fix it before continuing.

A clean run produces sessions that pass:

```bash
python -m analysis.audit_tickrate --sessions sessions --active-only
```

with no flags (mean ≥ 95 Hz, zero duplicates, zero non-monotonic).

For analysis-only replication on a machine that already has the completed
`sessions/` folder, the CS:Source server is not required. Use the steps below.

## Analysis-Only Replication

### 1. Create the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want GPU training, install a PyTorch build that matches your CUDA driver.
The scripts automatically choose CUDA when available and fall back to CPU unless
`--require-cuda` is supplied.

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

### 2. Required local data

The analysis expects the final participant folders under:

```text
sessions/P01 ... sessions/P12
```

Each participant folder should contain its telemetry `*_events.json`, manifest
JSONs, demographics JSON, and questionnaire JSONs. Raw `sessions/` data is not
modified by the scripts and may be ignored by git for privacy/size reasons.

### 3. Preprocess AK-47 sequence windows

```bash
.venv/bin/python -m analysis.preprocess_sequences \
  --sessions-dir sessions \
  --out-dir analysis/processed \
  --results-dir analysis/results \
  --window-start -2.0 \
  --window-end 0.0 \
  --seq-len 200 \
  --feature-set aim_plus_movement \
  --weapon ak47 \
  --config-name prefire_2s_ak47_aim_movement
```

Expected main outputs:

```text
analysis/processed/windows_prefire_2s_ak47_aim_movement.npz
analysis/processed/windows_prefire_2s_ak47_aim_movement_metadata.csv
analysis/processed/preprocess_report_prefire_2s_ak47_aim_movement.json
analysis/results/schema_summary.json
analysis/results/session_inventory.csv
analysis/results/sensitivity_report.csv
analysis/results/rejected_windows.csv
analysis/results/excluded_sessions.csv
```

### 4. Train classical baselines

```bash
.venv/bin/python -m analysis.run_baselines \
  --data analysis/processed/windows_prefire_2s_ak47_aim_movement.npz \
  --metadata analysis/processed/windows_prefire_2s_ak47_aim_movement_metadata.csv \
  --task binary_smooth \
  --out-dir analysis/results/Baselines+Binaries/ak47_baselines_binary_smooth
```

Repeat with `binary_humanised`, `binary_all`, and `multiclass`, changing the
output directory name to match the task.

### 5. Train the Transformer

```bash
.venv/bin/python -m analysis.train_transformer \
  --data analysis/processed/windows_prefire_2s_ak47_aim_movement.npz \
  --metadata analysis/processed/windows_prefire_2s_ak47_aim_movement_metadata.csv \
  --task binary_smooth \
  --splitter lopo \
  --epochs 60 \
  --batch-size 64 \
  --lr 3e-4 \
  --require-cuda \
  --amp \
  --out-dir analysis/results/Baselines+Binaries/ak47_transformer_binary_smooth
```

Repeat with `binary_humanised`, `binary_all`, and `multiclass`, changing the
output directory name to match the task. Remove `--require-cuda` if you need CPU
fallback.

Each Transformer run writes:

```text
metrics.json
per_fold_metrics.csv
fold_predictions.csv
fold_splits.json
confusion_matrix.png
classification_report.txt
training_curves.png
config_used.json
checkpoints/
```

To regenerate evaluation artefacts for a completed Transformer run:

```bash
.venv/bin/python -m analysis.evaluate_transformer \
  analysis/results/Baselines+Binaries/ak47_transformer_binary_smooth
```

Repeat for the other Transformer output directories as needed.

### 6. Process questionnaires

```bash
.venv/bin/python -m analysis.questionnaire_summary \
  --sessions-dir sessions \
  --out-dir analysis/results/questionnaires
```

Expected outputs include:

```text
analysis/results/questionnaires/questionnaire_summary.csv
analysis/results/questionnaires/questionnaire_descriptives.csv
analysis/results/questionnaires/questionnaire_stats.json
analysis/results/questionnaires/questionnaire_condition_summary.png
analysis/results/questionnaires/questionnaire_igeq_summary.png
```

### 7. Report build

The LaTeX report source is in:

```text
UAL_Undergrad_Thesis_Template__1____1_/
```

Build it with:

```bash
cd UAL_Undergrad_Thesis_Template__1____1_
latexmk -pdf -interaction=nonstopmode thesis.tex
```

This step requires a local LaTeX distribution with `latexmk` installed.

The generated PDF is:

```text
UAL_Undergrad_Thesis_Template__1____1_/thesis.pdf
```

## Current Dissertation Result Locations

```text
analysis/results/Baselines+Binaries/ak47_baselines_binary_smooth/
analysis/results/Baselines+Binaries/ak47_baselines_binary_humanised/
analysis/results/Baselines+Binaries/ak47_baselines_binary_all/
analysis/results/Baselines+Binaries/ak47_baselines_multiclass/
analysis/results/Baselines+Binaries/ak47_transformer_binary_smooth/
analysis/results/Baselines+Binaries/ak47_transformer_binary_humanised/
analysis/results/Baselines+Binaries/ak47_transformer_binary_all/
analysis/results/Baselines+Binaries/ak47_transformer_multiclass/
analysis/results/questionnaires/
```

The detailed training notes are in
**[docs/transformer-training-pipeline.md](docs/transformer-training-pipeline.md)**.

## Quick Start (already set up)

```bash
# 1. Start server
./start_css_server.sh

# 2. Start telemetry receiver
source .venv/bin/activate
python -m capture.session_orchestrator

# 3. In-game console (per round)
sm_nativeaim_me                          // register the participant
sm_nativeaim_mode humanised_high         // pick a mode
sm_nativeaim_active 0                    // bind to MOUSE4 (see SETUP.md §8)
```

## Conditions

| Label                          | Native mode       | Notes                                                  |
|--------------------------------|-------------------|--------------------------------------------------------|
| `human`                        | (assist off)      | Subject plays manually, no assist active               |
| `sm_native_raw`                | raw               | Zero-latency instant snap                              |
| `sm_native_smooth`             | smooth            | Distance-scaled gain, linear settle                    |
| `sm_native_humanised_med`      | humanised         | Reaction delay + light jitter                          |
| `sm_native_humanised_high`     | humanised\_high   | Distance-scaled jitter, drift, stochastic overshoot    |

Legacy Python-TCP labels (`bot_raw`, `bot_smooth`, `bot_humanised_*`) remain
recognised by `session_orchestrator.py` but are not the default. Their
parameters in `capture/bot_profiles/` document the bot design space used for
the dissertation discussion.

## Repository Structure

```
sourcemod/              SourceMod plugins (.sp + .smx) and Tickrate_Enabler
capture/                Python telemetry receiver, orchestrator, bot profiles
analysis/               Audit + plot tooling (audit_tickrate, plot_trajectories)
docs/                   Training and report documentation
cssource_server/        Server install (gitignored — install via SteamCMD)
start_css_server.sh     Server launch script (forces 100 Hz, restricts bots)
requirements.txt        Python dependencies
SETUP.md                Full replication guide
```
