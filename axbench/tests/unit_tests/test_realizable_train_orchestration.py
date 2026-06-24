from types import SimpleNamespace

import pandas as pd

from axbench.scripts import train as train_module


def model_params(**overrides):
    """Creates model params with the realizability defaults used by training.

    Args:
        **overrides: Field values to override on the returned namespace.

    Returns:
        A namespace with transform fields expected by train orchestration.
    """
    defaults = {
        "direction_transform": "none",
        "probe_transform": "none",
        "diffmean_transform": "none",
        "reft_transform": "none",
        "topk_metric": "activation",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolved_direction_transform_prefers_global_transform():
    params = model_params(
        direction_transform="projected_gradient",
        probe_transform="projected_activations",
    )

    assert train_module.resolved_direction_transform("LinearProbe", params) == "projected_gradient"


def test_resolved_direction_transform_uses_method_transform_when_global_is_none():
    params = model_params(diffmean_transform="projected_activations")

    assert train_module.resolved_direction_transform("DiffMean", params) == "projected_activations"


def test_first_realizable_model_params_finds_transformed_method():
    raw = model_params()
    projected = model_params(diffmean_transform="projected_activations")

    model_name, params = train_module.first_realizable_model_params(
        {"LinearProbe": raw, "DiffMean": projected}
    )

    assert model_name == "DiffMean"
    assert params is projected


def test_first_realizable_model_params_finds_realizable_topk_method():
    raw = model_params()
    basis_topk = model_params(topk_metric="realizable_basis")

    model_name, params = train_module.first_realizable_model_params(
        {"DiffMean": raw, "LsReFT": basis_topk}
    )

    assert model_name == "LsReFT"
    assert params is basis_topk


def test_add_realizable_projector_kwargs_only_for_transformed_methods():
    projector = object()
    kwargs = {}

    train_module.add_realizable_projector_kwargs(kwargs, projector, "projected_gradient")

    assert kwargs["projector"] is projector
    assert kwargs["realizable_projector"] is projector

    raw_kwargs = {}
    train_module.add_realizable_projector_kwargs(raw_kwargs, projector, "none")
    assert raw_kwargs == {}


def test_add_realizable_projector_kwargs_for_realizable_topk():
    projector = object()
    kwargs = {}

    train_module.add_realizable_projector_kwargs(
        kwargs,
        projector,
        "none",
        needs_projector=True,
    )

    assert kwargs["projector"] is projector
    assert kwargs["realizable_projector"] is projector


def test_prepare_realizable_basis_df_preserves_prefix_completion_boundary():
    concept_df = pd.DataFrame(
        [
            {
                "input": "The explanation becomes simpler if ",
                "output": "you track momentum.",
                "output_concept": "physics",
                "category": "positive",
            },
            {
                "input": "Ignore this ",
                "output": "other concept.",
                "output_concept": "chemistry",
                "category": "positive",
            },
        ]
    )
    negative_df = pd.DataFrame(
        [
            {
                "input": "The note says ",
                "output": "nothing technical.",
                "concept_genre": "science",
            },
            {
                "input": "Wrong genre ",
                "output": "negative.",
                "concept_genre": "art",
            },
        ]
    )
    metadata = {"concept_genres_map": {"physics": ["science"]}}

    rows = train_module.prepare_realizable_basis_df(
        concept_df,
        negative_df,
        "physics",
        metadata,
    )

    assert rows[["prefix", "completion", "text", "labels", "label"]].to_dict("records") == [
        {
            "prefix": "The explanation becomes simpler if ",
            "completion": "you track momentum.",
            "text": "The explanation becomes simpler if you track momentum.",
            "labels": 1,
            "label": 1,
        },
        {
            "prefix": "The note says ",
            "completion": "nothing technical.",
            "text": "The note says nothing technical.",
            "labels": 0,
            "label": 0,
        },
    ]
