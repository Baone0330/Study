# Input contract example

The history JSON input must contain a `history` array. Each element has `exam_date` in ISO date form and `image_paths` pointing to locally authorized images. This repository intentionally contains no sample clinical paths or patient data. A user can create a synthetic image and point the array at it for private integration testing. Only exams strictly before `--query_date` enter condition construction.

The released `scripts/smoke_test.py` is the supported data free check of the custom model components.
