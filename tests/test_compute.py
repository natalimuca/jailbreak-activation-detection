import torch

from src.direction.compute import (
    compute_directions,
    compute_raw_directions,
    select_candidate_layers,
    separation_score,
)


def _synthetic_acts(n_layers=4, n_prompts=10, d_model=8, seed=0):
    """Harmful acts centered at +3 along dim 0, harmless at -3, same dim,
    with small noise elsewhere -- gives a known, checkable ground-truth
    direction (unit vector along dim 0) at every layer."""
    g = torch.Generator().manual_seed(seed)
    harmful = torch.randn(n_layers, n_prompts, d_model, generator=g) * 0.1
    harmless = torch.randn(n_layers, n_prompts, d_model, generator=g) * 0.1
    harmful[:, :, 0] += 3.0
    harmless[:, :, 0] -= 3.0
    return harmful, harmless


def test_compute_directions_are_unit_norm():
    harmful, harmless = _synthetic_acts()
    directions = compute_directions(harmful, harmless)
    norms = directions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_compute_directions_points_toward_known_axis():
    harmful, harmless = _synthetic_acts()
    directions = compute_directions(harmful, harmless)
    # Ground truth: harmful - harmless is concentrated on dim 0, positive.
    assert (directions[:, 0] > 0.9).all()


def test_compute_raw_directions_not_normalized():
    harmful, harmless = _synthetic_acts()
    raw = compute_raw_directions(harmful, harmless)
    # harmful mean ~= +3, harmless mean ~= -3 on dim 0 -> raw diff ~= 6
    assert raw[:, 0].mean().item() > 5.0
    # unit-normalizing raw should match compute_directions
    unit = compute_directions(harmful, harmless)
    normalized_raw = raw / raw.norm(dim=-1, keepdim=True)
    assert torch.allclose(normalized_raw, unit, atol=1e-5)


def test_separation_score_high_for_separated_classes():
    harmful, harmless = _synthetic_acts(n_layers=1, n_prompts=50)
    directions = compute_directions(harmful, harmless)
    acts = torch.cat([harmful, harmless], dim=1)
    labels = torch.cat([torch.ones(50, dtype=torch.bool), torch.zeros(50, dtype=torch.bool)])
    scores = separation_score(acts, directions, labels)
    # Note: "pooled std" mixes both classes, so for two well-separated clusters
    # it's dominated by the between-class spread, not the small within-class
    # noise -- for a clean 50/50 mixture ~6 units apart this converges to
    # score ~= 2 (mean_diff / pooled_std), not some much larger number.
    assert scores[0].item() > 1.5


def test_separation_score_near_zero_for_identical_distributions():
    g = torch.Generator().manual_seed(1)
    acts = torch.randn(1, 100, 8, generator=g)
    directions = compute_directions(acts[:, :50], acts[:, 50:])
    labels = torch.cat([torch.ones(50, dtype=torch.bool), torch.zeros(50, dtype=torch.bool)])
    # Same underlying distribution for both halves -- direction is just noise,
    # so held-out separation should be small (not the inflated in-sample score).
    other_acts = torch.randn(1, 100, 8, generator=torch.Generator().manual_seed(2))
    scores = separation_score(other_acts, directions, labels)
    assert abs(scores[0].item()) < 2.0


def test_select_candidate_layers_skips_edges_and_ranks_by_score():
    # 20 layers, score peaks at layer 10 (middle) -- should be selected.
    # Score also peaks at layer 0 and 19 (edges) -- should be excluded by skip_frac.
    scores = torch.zeros(20)
    scores[0] = 100.0
    scores[19] = 100.0
    scores[10] = 50.0
    scores[11] = 40.0
    candidates = select_candidate_layers(scores, k=2, skip_frac=0.25)
    assert candidates == [10, 11]
