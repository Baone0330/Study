# Reproduction and private input schema

1. Use trusted copies of the ORIGINAL selected-best checkpoints and original
   CLIMATv2 runner/helper/model source tree. Do not substitute later clean or
   Strict checkpoints. Restore the original import layout on your own secured
   server; the external snapshots are intentionally not bundled here.
2. Complete all six branch definitions in a private YAML outside the clone.
   Required branch keys: `branch_id`, `internal_name`, `seeds`, `root`,
   `val_manifest`, `cohort_manifest`, `eval_script`, `metric_function`,
   `metric_call`, `matmul_allow_tf32`, `upstream_variant`, `upstream_config`.
   Optional provenance-only keys: `upstream_checkpoint`, `upstream_training_seed`.
3. The original run folder convention is `runs/<internal>/seed_<seed>/` and
   contains `best_checkpoint.pt`, `best_checkpoint_metadata.json`,
   `RUN_COMPLETED.json`, `resolved_config.yaml`, `val_metrics_by_epoch.csv`.
   The checkpoint must contain `seed`, `epoch`, `config.branch` and the complete
   `model_state_dict`. Safe `weights_only=True` loading has no pickle fallback.
4. Original Val cohort CSV has unique patient keys, `target_year`,
   `target_label`, `history_count`. The shared slot CSV has six rows per patient,
   `split`, `slot_index`, `slot_date`, `source_type`, `exam_date`,
   `image_paths_json`, `causal_source_k`. Actual clinical examples are not public.
   All six branch slot manifests must exactly match the historical shared Val.
5. Run help and dry-run, then replay, then aggregate using the README commands.
   Do not commit the completed configuration, source index, patient predictions,
   checkpoint states, metadata/private source paths, or console logs.
6. Original native model construction tries to configure optimizers. The replay
   adapter disables only this constructor callback in memory; model layers and
   parameter names are untouched, then the complete selected state loads strict.
   Runtime guards prohibit training mode, optimizer construction, backward and
   model downloads. No ImageNet weight download/initialization substitutes for
   the full historical classifier state.
7. Dry-run and aggregation run on CPU. Replay uses the validated CUDA route;
   GPU absence is an explicit failure, not a silent precision/device fallback.
   Batch/precision/view sets are not automatically tuned. No other user's job
   is stopped. For OOM, inspect the recorded failure before an explicitly
   verified mathematics-preserving retry.
8. Inspect private status/QA after any failure. A completed branch requires all
   three original seed outputs; incomplete branches stay clearly marked partial.

Validated environment versions are recorded separately; requirements list
libraries but does not authorize installing/upgrading the shared server.
The validated driver reports CUDA 13 maximum compatibility, while the actual
PyTorch runtime used here is CUDA 12.8 with sm_120 support. Those are distinct.
