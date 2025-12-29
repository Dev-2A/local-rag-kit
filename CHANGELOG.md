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
