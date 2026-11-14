"""Section B - Valid evaluation setup.
Promises: 80/20 outer split, then 80/20 fit/validation (random_state=42);
rows with users/movies unseen in the fitting data are excluded;
user-movie matrix built from fitting data only, missing = NaN.
"""
