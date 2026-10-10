# Scope and known limitations

- This task is historical internal Val reproduction, not a new training,
  strict temporal data reconstruction, new leakage audit or independent Test.
- The old synthetic Train support includes nominal target-year slots. No Train
  images are forwarded in this replay; their historical provenance is not
  reinterpreted as a new Strict Pre-Target protocol.
- The same internal Val selected historical checkpoints and thresholds. Do not
  market these numbers as external clinical generalization evidence.
- Full B16v1 and Plain-SD are historical names, not proof that every contrast is
  a single-variable mechanistic intervention.
- Clinical data, patient IDs/labels/probabilities, original classifier and
  generator states, private source mappings and logs remain server-private.
  The public code needs those legally accessible external assets; no complete
  standalone public clinical dataset/model download is provided.
- Exact six-decimal metric reproduction is weaker than bit-exact equality of
  all patient probabilities. Raw floating differences remain privately recorded.
- No global package, driver, Git/SSH configuration or repository protection
  setting is altered by the replay tools. There is no force push or file digest
  workflow. Git commit IDs are provenance references, not model-file checksums.
