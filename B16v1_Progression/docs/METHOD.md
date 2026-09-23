# Method and source correspondence

This package is a portable extraction of the formal B16v1 Progression runtime. `longitudinal_condition.py` preserves the audited historical feature, exam set, temporal, and bridge definitions. `progression_condition.py` preserves the formal `ProgressionCondition` class. `b14_cpfm.py` and `b14_hooks.py` preserve the B14 class implementations from the selected CPFM training source, without the unrelated historical training experiment, fixed case identifiers, or data manifests. `pipeline.py` ports the active loader, pre query condition construction, and generation call to explicit external paths.

The Progression training entry preserves the original feature extraction, patient level inner split, batch construction, objective, and optimizer settings. Its input and output paths are command line arguments. It must be run only with data the user is authorized to access; no clinical data are distributed here.

Source lineage and strict checkpoint load evidence are recorded in the release audit outside this public repository. Synthetic checkpoint parity tests compared original and released model outputs with maximum absolute difference 0 for the checked forwards.
