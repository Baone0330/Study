# Privacy and data policy

Clinical ultrasound data used in this project are not included in this repository. The release contains no raw patient images, patient identifiers, patient level prediction tables, report text, or patient split manifests. The released checkpoints retain model tensors and non private module labels only; unnecessary training state and source metadata were removed.

The provided training and generation scripts can read authorized clinical input supplied by a user. Keep that input, generated images, and patient level training output outside this repository. No credentials or private server paths are required by the public package.
