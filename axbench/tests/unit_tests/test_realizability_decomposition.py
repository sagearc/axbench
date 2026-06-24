import torch

from axbench.utils.realizability_decomposition import score_realizability_decomposition


def test_realizability_decomposition_matches_covariance_identity():
    logits0 = torch.tensor([0.2, -0.1, 0.4, -0.3])
    readout = torch.tensor([1.0, -0.5, 0.25, -1.0])
    jv = torch.tensor([0.7, -0.2, 0.1, -0.4])

    score = score_realizability_decomposition(logits0, jv, readout)

    p = torch.softmax(logits0, dim=-1)
    centered_readout = readout - (p * readout).sum()
    centered_jv = jv - (p * jv).sum()
    expected_r = (p * centered_readout * centered_jv).sum()
    expected_c = (p * centered_jv.square()).sum().sqrt()
    expected_b = (p * centered_readout.square()).sum().sqrt()
    expected_a = expected_r / (expected_c * expected_b)

    assert torch.allclose(score["realizability_R"], expected_r.reshape(1))
    assert torch.allclose(score["capacity_C"], expected_c.reshape(1))
    assert torch.allclose(score["availability_B"], expected_b.reshape(1))
    assert torch.allclose(score["alignment_A"], expected_a.reshape(1))


def test_constant_logit_velocity_has_zero_capacity_and_realizability():
    logits0 = torch.tensor([0.0, 0.5, -0.25])
    readout = torch.tensor([1.0, -1.0, 0.25])
    jv = torch.tensor([3.0, 3.0, 3.0])

    score = score_realizability_decomposition(logits0, jv, readout)

    assert score["capacity_C"].item() == 0.0
    assert score["alignment_A"].item() == 0.0
    assert abs(score["realizability_R"].item()) < 1e-7


def test_batched_decomposition_scores_share_availability():
    logits0 = torch.tensor([0.0, 0.2, -0.4])
    readout = torch.tensor([1.0, 0.0, -1.0])
    jv = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])

    score = score_realizability_decomposition(logits0, jv, readout)

    assert score["capacity_C"].shape == (2,)
    assert torch.allclose(score["availability_B"][0], score["availability_B"][1])
    assert score["realizability_R"][0] * score["realizability_R"][1] < 0
