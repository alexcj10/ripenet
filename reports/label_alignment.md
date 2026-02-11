To ensure a fair comparison between V1 and V2, all time predictions are
converted into a common positive-day scale during evaluation.

If a model outputs a zero or negative value, it is transformed using:
normalized_days = abs(value) + 1

This normalization ensures both models are evaluated under the same
interpretable timeline without modifying training data.
