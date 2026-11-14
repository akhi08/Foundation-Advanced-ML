"""Stage 1 - Get and validate data.
Constants at the top (paths, column names, rating range).
Promises: missing file raises FileNotFoundError; missing column raises ValueError;
empty file raises ValueError; timestamp becomes a real UTC datetime.
"""
