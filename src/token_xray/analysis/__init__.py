"""Analysis layer: normalization, aggregates, and duplicate detection.

Nothing in this package retains raw prompt text. Prompts enter only as already
normalized strings and leave only as hashes, bottom-k sketches, or counts.
"""
