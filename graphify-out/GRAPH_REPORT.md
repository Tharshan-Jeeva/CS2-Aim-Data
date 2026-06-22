# Graph Report - .  (2026-06-22)

## Corpus Check
- Corpus is ~19,887 words - fits in a single context window. You may not need a graph.

## Summary
- 309 nodes · 466 edges · 16 communities (14 shown, 2 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 45 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Transformer Training & Eval|Transformer Training & Eval]]
- [[_COMMUNITY_Feature Extraction & Plots|Feature Extraction & Plots]]
- [[_COMMUNITY_Transformer Config|Transformer Config]]
- [[_COMMUNITY_Sequence Preprocessing|Sequence Preprocessing]]
- [[_COMMUNITY_Server Plugin Interfaces|Server Plugin Interfaces]]
- [[_COMMUNITY_Baseline CV & Dataset|Baseline CV & Dataset]]
- [[_COMMUNITY_Bot Aim Generator|Bot Aim Generator]]
- [[_COMMUNITY_Session Orchestration & Telemetry|Session Orchestration & Telemetry]]
- [[_COMMUNITY_Tickrate Audit & Labels|Tickrate Audit & Labels]]
- [[_COMMUNITY_Baseline Runner & Splits|Baseline Runner & Splits]]
- [[_COMMUNITY_Sequence Dataset & Labels|Sequence Dataset & Labels]]
- [[_COMMUNITY_Analysis Package Init|Analysis Package Init]]
- [[_COMMUNITY_Label Definition|Label Definition]]

## God Nodes (most connected - your core abstractions)
1. `CEmptyServerPlugin` - 26 edges
2. `preprocess()` - 19 edges
3. `run_training()` - 15 edges
4. `BotAimGenerator` - 14 edges
5. `run_session()` - 14 edges
6. `preprocess` - 13 edges
7. `run()` - 11 edges
8. `train_fold()` - 11 edges
9. `plot_trajectories()` - 10 edges
10. `load_sequence_data()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `run_session()` --calls--> `load_config()`  [INFERRED]
  capture/session_orchestrator.py → capture/bot_aim_generator.py
- `run_training()` --calls--> `make_lopo_splits()`  [INFERRED]
  analysis/train_transformer.py → analysis/make_splits.py
- `run_training()` --calls--> `save_splits()`  [INFERRED]
  analysis/train_transformer.py → analysis/make_splits.py
- `main()` --calls--> `load_sequence_data()`  [INFERRED]
  analysis/make_splits.py → analysis/sequence_dataset.py
- `PreprocessConfig` --uses--> `SessionMeta`  [INFERRED]
  analysis/preprocess_sequences.py → analysis/labels.py

## Communities (16 total, 2 thin omitted)

### Community 0 - "Transformer Training & Eval"
Cohesion: 0.08
Nodes (32): main(), Regenerate summary plots/text from a transformer run directory., _read_predictions(), AimSequenceDataset, Torch dataset for (sequence, class label)., aggregate_metrics(), ChannelScaler, class_weights() (+24 more)

### Community 1 - "Feature Extraction & Plots"
Cohesion: 0.09
Nodes (32): _acf(), _direction_changes(), Per-window feature extraction from CS:Source telemetry tick streams.  Inputs are, Return (spectral_centroid_hz, spectral_entropy_normalised).      The signal is m, Lag-`lag` autocorrelation of `signal`. Robust to short windows., Count sign flips in `diff`, ignoring near-zero noise., Compute the feature vector for samples [i0:i1] of a tick stream.      Caller gua, Yield (window_start_tick, feature_vector) for one events file. (+24 more)

### Community 2 - "Transformer Config"
Cohesion: 0.06
Nodes (32): model, conv_kernel, conv_stride, d_model, dim_feedforward, dropout, n_heads, n_layers (+24 more)

### Community 3 - "Sequence Preprocessing"
Cohesion: 0.16
Nodes (27): angle_delta(), _append_sensitivity(), _build_config(), _enemy_position(), _events_between(), _extract_ticks(), _find_scalar(), _interpolate_window() (+19 more)

### Community 4 - "Server Plugin Interfaces"
Cohesion: 0.08
Nodes (25): IGameEventListener, IServerPluginCallbacks, CEmptyServerPlugin, ClientActive, ClientCommand, ClientConnect, ClientDisconnect, ClientPutInServer (+17 more)

### Community 6 - "Baseline CV & Dataset"
Cohesion: 0.13
Nodes (15): _choose_splitter(), cross_validate(), CVReport, fit_final_model(), FoldResult, make_model(), Baseline classifiers for human-vs-aimbot trajectory classification.  Two models:, Run CV and return per-fold metrics.      Returns an empty report (with a descrip (+7 more)

### Community 7 - "Bot Aim Generator"
Cohesion: 0.18
Nodes (9): angle_delta(), angle_to_target(), BotAimGenerator, clamp(), is_in_fov(), load_config(), normalize_yaw(), Return the newest available tick, discarding older ones. (+1 more)

### Community 8 - "Session Orchestration & Telemetry"
Cohesion: 0.16
Nodes (18): build_session_name(), count_heartbeats(), count_ticks(), generate_manifest(), is_bot_session(), is_sm_native_session(), native_mode_from_label(), parse_args() (+10 more)

### Community 9 - "Tickrate Audit & Labels"
Cohesion: 0.15
Nodes (15): _active_intervals(), audit_directory(), audit_one(), AuditRow, main(), _print_table(), Audit the tick stream from each recorded session.  Reports, per session: tick co, Return raw stats for one session's events list.      Uses the engine `tick` coun (+7 more)

### Community 10 - "Baseline Runner & Splits"
Cohesion: 0.22
Nodes (14): main(), make_lopo_splits(), Create Leave-One-Participant-Out splits for preprocessed windows., save_splits(), _col(), _direction_changes(), engineered_window_features(), _feature_index() (+6 more)

### Community 11 - "Sequence Dataset & Labels"
Cohesion: 0.17
Nodes (11): label_mapping(), labels_for_task(), Return {label_string: class_index} for the requested task.      Supported task n, Condition labels included in a task., load_sequence_data(), Utilities for loading preprocessed sequence windows., Load a ``windows_*.npz`` plus metadata CSV and filter for ``task``., read_metadata() (+3 more)

## Knowledge Gaps
- **51 isolated node(s):** `sessions_dir`, `out_dir`, `results_dir`, `config_name`, `window_start` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_sequence_data()` connect `Sequence Dataset & Labels` to `Transformer Training & Eval`, `Baseline Runner & Splits`?**
  _High betweenness centrality (0.239) - this node is a cross-community bridge._
- **Why does `discover_sessions()` connect `Tickrate Audit & Labels` to `Feature Extraction & Plots`, `Sequence Preprocessing`, `Baseline CV & Dataset`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Why does `run_training()` connect `Transformer Training & Eval` to `Baseline Runner & Splits`, `Sequence Dataset & Labels`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `run_training()` (e.g. with `load_sequence_data()` and `make_lopo_splits()`) actually correct?**
  _`run_training()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `run_session()` (e.g. with `create_app()` and `load_config()`) actually correct?**
  _`run_session()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Feature extraction and baseline classifiers for aim-trajectory data.`, `Create Leave-One-Participant-Out splits for preprocessed windows.`, `Regenerate summary plots/text from a transformer run directory.` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Transformer Training & Eval` be split into smaller, more focused modules?**
  _Cohesion score 0.080338266384778 - nodes in this community are weakly interconnected._