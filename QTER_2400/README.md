# QTER v7 question-answer dataset

This directory contains the public question-answer export for the QTER v7 formal400 evaluation set.

- 2,400 question-answer pairs
- 400 circuit-diagram images
- Five primary types: Count, Value, Position, Path, and Connectivity
- English and reviewed Chinese questions

## Files

- `data/questions_answers.jsonl`: canonical machine-readable data
- `data/questions_answers.csv`: UTF-8 CSV for inspection
- `images/*.jpg`: raw circuit images referenced by `image_path`
- `dataset_info.json`: counts and type distribution
- `SHA256SUMS`: file checksums

Each JSONL row contains `question_id`, `case_id`, `image_id`, `image_path`, `primary_type`, `subtype`, `difficulty`, `question_en`, `question_zh`, `answer`, and `answer_type`.

This export contains no model predictions, baseline results, API credentials, local absolute paths, or server logs.
