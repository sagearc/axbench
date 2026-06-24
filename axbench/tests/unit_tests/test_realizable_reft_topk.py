import torch

from axbench.models.interventions import TopKReLUIntervention


def make_intervention(weight):
    """Creates a one-direction TopK ReLU intervention with a fixed direction."""

    intervention = TopKReLUIntervention(embed_dim=len(weight), low_rank_dimension=1)
    with torch.no_grad():
        intervention.proj.weight.copy_(torch.tensor([weight], dtype=torch.float32))
        intervention.proj.bias.zero_()
    return intervention


def test_realizable_basis_topk_ranks_positions_in_basis_coordinates():
    intervention = make_intervention([1.0, 1.0, 0.0])
    base = torch.tensor([[[0.0, 10.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])

    outputs = intervention(
        base,
        subspaces={
            "k": 1,
            "topk_metric": "realizable_basis",
            "topk_basis": torch.tensor([[1.0], [0.0], [0.0]]),
        },
    )
    latent, non_topk_latent = outputs.latent

    assert latent.tolist() == [[10.0, 2.0, 2.0]]
    assert non_topk_latent.tolist() == [[10.0, 0.0, 2.0]]


def test_identity_basis_topk_matches_activation_topk():
    intervention = make_intervention([1.0, 2.0, 0.0])
    base = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]])

    raw_outputs = intervention(base, subspaces={"k": 2, "topk_metric": "activation"})
    basis_outputs = intervention(
        base,
        subspaces={
            "k": 2,
            "topk_metric": "realizable_basis",
            "topk_basis": torch.eye(3),
        },
    )

    assert torch.allclose(raw_outputs.output, basis_outputs.output)
    assert torch.allclose(raw_outputs.latent[1], basis_outputs.latent[1])


def test_topk_is_bounded_by_sequence_length():
    intervention = make_intervention([1.0, 0.0])
    base = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])

    outputs = intervention(base, subspaces={"k": 8, "topk_metric": "activation"})

    assert outputs.latent[1].tolist() == [[0.0, 0.0]]


def test_topk_accepts_bfloat16_activations_with_float32_weights():
    intervention = make_intervention([1.0, 0.0])
    base = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]], dtype=torch.bfloat16)

    outputs = intervention(base, subspaces={"k": 1, "topk_metric": "activation"})

    assert outputs.output.dtype == torch.bfloat16
    assert outputs.latent[0].dtype == torch.float32
