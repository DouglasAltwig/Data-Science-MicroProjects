import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ThresholdClassifier(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that applies a probability threshold
    to a classifier's predictions. This allows us to include thresholding
    as a final step in an sklearn.pipeline.Pipeline.
    """
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y=None):
        # The model is already fitted, so this does nothing.
        return self

    def transform(self, X):
        # This is for pipeline compatibility, but we'll use predict.
        return self.predict(X)

    def predict(self, X):
        X_float32 = X.astype(np.float32)
        # Get probabilities from the wrapped model
        probabilities = self.model.predict_proba(X_float32)[:, 1]
        # Apply the custom threshold
        return (probabilities > self.threshold).astype(int)


def build_feature_pipeline(preprocessor):
    """Creates the preprocessing pipeline."""
    return Pipeline([
        ('preprocessor', preprocessor),
    ])