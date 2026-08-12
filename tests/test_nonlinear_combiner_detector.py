import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detectors.nonlinear_combiner_detector import is_flagged, score
from src.sae.qwen_scope import TopKSAE


def _make_sae(k: int) -> TopKSAE:
    # d_model=2, d_sae=2. Feature 0 reads dim0, feature 1 reads dim1.
    W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    W_dec = torch.zeros(2, 2)
    b_enc = torch.zeros(2)
    b_dec = torch.zeros(2)
    return TopKSAE(W_enc, W_dec, b_enc, b_dec, k=k)


def _fitted_pipeline() -> Pipeline:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline


def test_score_matches_pipeline_predict_proba_directly():
    sae = _make_sae(k=2)
    activations_by_layer = {0: torch.tensor([[5.0, -3.0], [-2.0, 4.0]])}
    features = [(0, 0), (0, 1)]
    pipeline = _fitted_pipeline()

    s = score(activations_by_layer, {0: sae}, features, pipeline)

    expected = pipeline.predict_proba(np.array([[5.0, -3.0], [-2.0, 4.0]]))[:, 1]
    assert torch.allclose(s, torch.tensor(expected), atol=1e-6)


def test_is_flagged_thresholding():
    assert is_flagged(0.7, threshold=0.5)
    assert not is_flagged(0.3, threshold=0.5)
