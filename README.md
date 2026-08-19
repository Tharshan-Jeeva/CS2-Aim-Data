# CS:Source Aim-Trajectory Data Collection and Analysis Pipeline

BSc Critical Project 6: a pipeline for collecting and analysing aim trajectories from a local **Counter-Strike: Source** server.

The repository contains the SourceMod plugins, Python telemetry/capture code, analysis pipeline, server configuration, and setup documentation used for the project.

## Overview

The project uses a local CS:Source dedicated server to collect player telemetry at a target rate of 100 Hz. SourceMod plugins provide the telemetry and aim-assist implementations, while the Python pipeline receives and stores the resulting event data.

The analysis pipeline preprocesses recorded sessions into fixed-length trajectory windows and supports both classical baseline models and a PyTorch Transformer model.

## Repository Structure

```text
CS2-Aim-Data/
├── analysis/
│   ├── audit_tickrate.py
│   ├── baselines.py
│   ├── configs/
│   ├── dataset.py
│   ├── evaluate_transformer.py
│   ├── features.py
│   ├── labels.py
│   ├── make_splits.py
│   ├── plot_trajectories.py
│   ├── preprocess_sequences.py
│   ├── run_baselines.py
│   ├── sequence_dataset.py
│   ├── train_transformer.py
│   ├── transformer_model.py
│   └── __init__.py
│
├── capture/
│   ├── bot_aim_generator.py
│   ├── bot_profiles/
│   ├── session_orchestrator.py
│   ├── telemetry_server.py
│   └── __init__.py
│
├── server_config/
│   └── server.cfg
│
├── sourcemod/
│   ├── Tickrate_Enabler.so
│   ├── Tickrate_Enabler.vdf
│   ├── cs_aim_live_controller.sp
│   ├── cs_aim_live_controller.smx
│   ├── cs_aim_override.sp
│   ├── cs_aim_override.smx
│   ├── cs_aim_telemetry.sp
│   ├── cs_aim_telemetry.smx
│   └── tickrate_enabler_src/
│
├── start_css_server.sh
├── requirements.txt
├── SETUP.md
└── README.md
```

The repository does **not** include the CS:Source server installation or participant session data. These are created locally during setup and data collection.

## Architecture

```text
Counter-Strike: Source dedicated server
        │
        ├── Tickrate Enabler
        │       └── enables 100 Hz server operation
        │
        ├── SourceMod telemetry plugin
        │       └── sends telemetry to Python
        │
        └── SourceMod aim controller
                └── provides the native aim-assist conditions

Python capture pipeline
        │
        ├── telemetry_server.py
        │       └── receives and stores telemetry
        │
        ├── session_orchestrator.py
        │       └── manages capture sessions and labels
        │
        └── bot_aim_generator.py
                └── legacy Python-TCP aim path

Analysis pipeline
        │
        ├── audit_tickrate.py
        ├── plot_trajectories.py
        ├── preprocess_sequences.py
        ├── run_baselines.py
        └── train/evaluate Transformer
```

## Aim Conditions

The session orchestrator supports the following SourceMod-native conditions:

| Label | Native mode |
|---|---|
| `human` | Assist disabled |
| `sm_native_raw` | `raw` |
| `sm_native_smooth` | `smooth` |
| `sm_native_humanised_med` | `humanised` |
| `sm_native_humanised_high` | `humanised_high` |

The repository also contains a **legacy Python-TCP aim path** with the following labels:

```text
bot_raw
bot_smooth
bot_humanised_low
bot_humanised_med
bot_humanised_high
```

These use the YAML profiles in `capture/bot_profiles/` and are retained in the repository as the earlier aim-control implementation.

The SourceMod-native controller is implemented in:

```text
sourcemod/cs_aim_live_controller.sp
```

The legacy Python-TCP implementation is provided by:

```text
sourcemod/cs_aim_override.sp
capture/bot_aim_generator.py
```

## Data Collection

`capture/session_orchestrator.py` starts the local telemetry receiver and manages individual capture sessions.

For a standard session:

```bash
source .venv/bin/activate
python -m capture.session_orchestrator
```

The orchestrator supports participant IDs, condition labels, session duration, automatic stopping, and session manifests.

Captured session data is written to a local `sessions/` directory when the pipeline is run. This directory is not part of the repository because participant data is not included in the source distribution.

## Analysis

### Environment

Create the Python environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Audit telemetry

Use the tickrate audit to check captured sessions:

```bash
python -m analysis.audit_tickrate --sessions sessions --active-only
```

### Inspect trajectories

Trajectory plots can be generated with:

```bash
python -m analysis.plot_trajectories --sessions sessions
```

See the script's `--help` output for the available plotting options.

### Preprocess sequences

The preprocessing pipeline converts recorded event JSON files into model-ready sequence data.

The exact preprocessing arguments used for a particular experiment should be kept with the corresponding experiment configuration and results.

### Classical baselines

Classical baseline models can be run using:

```bash
python -m analysis.run_baselines --help
```

The baseline implementation is contained in:

```text
analysis/baselines.py
```

### Transformer

The project includes a PyTorch Transformer implementation for sequence classification.

Training:

```bash
python -m analysis.train_transformer --help
```

Evaluation of an existing run:

```bash
python -m analysis.evaluate_transformer --help
```

The Transformer configuration is also provided in:

```text
analysis/configs/transformer_default.json
```

## Server Setup

The repository includes the files required to configure the CS:Source server:

```text
server_config/server.cfg
sourcemod/
start_css_server.sh
```

The full fresh-machine installation and replication procedure is documented in **[SETUP.md](SETUP.md)**.

The setup guide covers:

- CS:Source dedicated server installation
- Metamod:Source and SourceMod
- SteamWorks
- Tickrate Enabler
- SourceMod plugin deployment
- Server and client configuration
- Running a capture session
- Auditing captured telemetry
- Inspecting trajectories
- Rebuilding Tickrate Enabler
- The legacy Python-TCP aim path

## Tickrate Enabler

`Tickrate_Enabler.so` and its VDF file are included in the repository.

The source used to build the Tickrate Enabler is located in:

```text
sourcemod/tickrate_enabler_src/
```

See `sourcemod/tickrate_enabler_src/BUILD.md` for build information.

## Requirements

Python dependencies are listed in:

```text
requirements.txt
```

The server itself requires additional CS:Source, Metamod:Source, SourceMod, and SteamWorks components. Their installation is described in `SETUP.md`.

## Reproduction

For a complete fresh-machine setup, follow:

```text
SETUP.md
```

For an already configured environment, the main components are:

```bash
./start_css_server.sh
```

and, in another terminal:

```bash
source .venv/bin/activate
python -m capture.session_orchestrator
```

The resulting session data can then be audited, visualised, preprocessed, and used by the baseline and Transformer analysis scripts.

## Data and Results

Participant session data, generated analysis outputs, model checkpoints, figures, and other experiment artefacts are **not included in this repository**.

They must be generated locally from the capture and analysis pipelines.
