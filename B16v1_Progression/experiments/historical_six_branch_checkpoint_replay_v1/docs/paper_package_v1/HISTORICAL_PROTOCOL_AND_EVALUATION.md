# Historical protocol and evaluation identity

The preceding stage actually performed18 independent selected-checkpoint forwards and six patient-matched three-seed ensembles. All42 six-decimal display fields matched the archived historical reference. Patient probabilities were not necessarily bit-exact. This paper stage uses only the existing aggregate table and performs zero new clinical forwards.

Seeds42,3407,2026 are downstream training seeds. Upstream Original Progression training.seed=20260921 is verified from the original encoder configuration; the checkpoint itself does not record that seed at top level. The selected Original P16 checkpoint epoch was120. Clean/S2/Strict revisions are out of scope.

Classification input: native fixed six slots, all native views, RGB256/ImageNet normalization, zero report/time content with masks/learned positions active. Exam view aggregation preserves512×8×8 maps; standard ResNet GAP/FC is not called. Patient CLS is read through the original prognosis Transformer and five-to-two logit adapter.

ROC-AUC is threshold-free. AP uses average_precision_score, not trapezoidal PR area. Accuracy, Precision, Sensitivity, Specificity and F1 import the original native branch threshold results and confusion cells. The ensemble is a positive-probability mean, not a mean of seed AUCs. No new inferential statistics are added.

These are historical internal-Val comparisons, not independent external clinical validation or Official Test. Internal Val was used for historical checkpoint/threshold selection. P16 changes fusion and supervision, and Plain-SD removes several inputs; neither comparison alone isolates a causal module effect. Existing later protocol outcomes are separate and are not overwritten or pooled here.
