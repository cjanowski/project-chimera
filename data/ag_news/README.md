# AG News Dataset (cached)

This directory contains a cached copy of the AG News dataset retrieved via Hugging Face `datasets`.

Layout:
- train/*.parquet
- validation/*.parquet
- test/*.parquet

Each row contains:
- text: str  (news title + description)
- label: int (class id in [0..3])
- label_text: str (class name among {World, Sports, Business, Sci/Tech})

Regeneration:
  python scripts/download_ag_news.py

Notes:
- We store parquet shards for fast reload.
- Original source: https://huggingface.co/datasets/ag_news
