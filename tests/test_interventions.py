import torch

from src.direction.interventions import _project_out


def test_project_out_removes_component_along_direction():
    direction = torch.tensor([1.0, 0.0, 0.0])
    act = torch.tensor([[5.0, 3.0, -2.0]])
    result = _project_out(act, direction)
    assert torch.allclose(result, torch.tensor([[0.0, 3.0, -2.0]]))


def test_project_out_leaves_orthogonal_components_untouched():
    direction = torch.tensor([0.0, 1.0, 0.0])
    act = torch.tensor([[5.0, 0.0, -2.0]])  # nothing along `direction`
    result = _project_out(act, direction)
    assert torch.allclose(result, act)


def test_project_out_is_idempotent():
    # Projecting out twice should equal projecting out once -- the component
    # along the direction is already zero after the first pass.
    direction = torch.tensor([0.6, 0.8, 0.0])  # unit vector
    act = torch.randn(4, 3)
    once = _project_out(act, direction)
    twice = _project_out(once, direction)
    assert torch.allclose(once, twice, atol=1e-6)


def test_project_out_handles_batch_and_sequence_dims():
    direction = torch.tensor([1.0, 0.0])
    act = torch.randn(2, 5, 2)  # [batch, seq, d_model]
    result = _project_out(act, direction)
    # every position's component along `direction` should now be ~0
    assert torch.allclose(result[..., 0], torch.zeros(2, 5), atol=1e-6)
