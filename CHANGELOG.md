# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [0.1.0] - 2025-12-29
### Added
- Local document indexing (TF-IDF / SBERT)
- Query API with `include_text`, `max_chars`, `top_docs_k`
- Hybrid retrieval: SBERT candidate docs + TF-IDF rerank
- Dataset generator (`tools/generate_samples_v2.py`)
- Eval scripts including hybrid eval and failure export

### Changed
- Improved JSON response encoding (UTF-8)
- Added safe wrapper for SentenceTransformer.encode kwargs compatibility

### Fixed
- Eval argparse bug (`args.dataset`)

## [Unreleased]
### Added
- API endpoint `GET /check-data` to detect whether `data_dir` content has changed compared to indexed fingerprints (no auto re-index).
- One-click Windows script `start_server.cmd` to (re)build missing indexes and start the API server.

### Changed
- Unified data fingerprint calculation between CLI indexing and API freshness checks.

### Fixed
- Resolved fingerprint function shadowing caused by duplicate `compute_data_fingerprint` definitions.
- Fixed CLI indexing fingerprint call to match the updated `compute_data_fingerprint(data_dir, allowed_exts, deep=False)` signature.
