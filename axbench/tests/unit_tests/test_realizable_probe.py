import torch
import pandas as pd
import pytest

from axbench.utils.realizable_basis import (
    _calibration_text_column,
    calibration_start_positions,
    completion_prediction_mask,
    validate_realizable_basis_examples,
)
from axbench.utils.probe_training import train_probe_direction
from axbench.utils.subspace_training import (
    ProjectedOptimizer,
    SubspaceProjector,
)


class OffsetTokenizer:
    """Tokenizes characters and exposes offsets for prefix-boundary tests."""

    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    all_special_ids = [0, 1, 2]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        padding=False,
        truncation=False,
        max_length=None,
        **_,
    ):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for text in values:
            ids = [ord(char) % 251 + 3 for char in text]
            offsets = [(idx, idx + 1) for idx in range(len(text))]
            if add_special_tokens:
                ids = [self.bos_token_id] + ids
                offsets = [(0, 0)] + offsets
                ids = ids + [self.eos_token_id]
                offsets = offsets + [(len(text), len(text))]
            if truncation and max_length is not None:
                ids = ids[:max_length]
                offsets = offsets[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            offset_mapping.append(offsets)
        if padding and not single:
            max_len = max(len(ids) for ids in input_ids)
            for idx, ids in enumerate(input_ids):
                pad = max_len - len(ids)
                input_ids[idx] = ids + [self.pad_token_id] * pad
                attention_mask[idx] = attention_mask[idx] + [0] * pad
                offset_mapping[idx] = offset_mapping[idx] + [(0, 0)] * pad
        if single:
            out = {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}
            if return_offsets_mapping:
                out["offset_mapping"] = offset_mapping[0]
            return out
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            out["offset_mapping"] = offset_mapping
        return out


def assert_unit_norm(vector, *, atol=1e-6):
    """Asserts that a vector has unit Euclidean norm.

    Args:
        vector: Tensor whose norm should be one.
        atol: Absolute tolerance for the norm check.
    """
    assert torch.isclose(torch.linalg.vector_norm(vector), torch.tensor(1.0), atol=atol)


def assert_in_projector_subspace(vector, projector, *, atol=1e-6):
    """Asserts that a vector is unchanged by a projector.

    Args:
        vector: Tensor expected to lie inside the projector subspace.
        projector: Projector used to compute the residual.
        atol: Absolute tolerance for the residual norm.
    """
    residual = vector - projector.project(vector)
    assert torch.linalg.vector_norm(residual) <= atol


def test_projected_optimizer_projects_parameter_after_step():
    weight = torch.nn.Parameter(torch.tensor([1.0, 0.0, 1.0]))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    projector = SubspaceProjector.from_basis(torch.eye(3, 2))
    optimizer = ProjectedOptimizer(optimizer, [weight], projector)

    loss = -weight.sum()
    loss.backward()
    optimizer.step()

    assert weight.detach()[2].item() == 0.0


def test_projected_gradient_probe_stays_in_subspace():
    torch.manual_seed(0)
    x = torch.tensor(
        [
            [2.0, 0.0, 10.0],
            [1.5, 0.2, 9.0],
            [-2.0, 0.0, -10.0],
            [-1.5, -0.2, -9.0],
        ]
    )
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    projector = SubspaceProjector.from_basis(torch.eye(3, 2))

    direction = train_probe_direction(
        x,
        y,
        {
            "device": "cpu",
            "seed": 0,
            "epochs": 3,
            "batch_size": 2,
            "lr": 0.1,
            "weight_decay": 0.0,
        },
        projector=projector,
    )

    assert_unit_norm(direction)
    assert_in_projector_subspace(direction, projector)


def test_plain_probe_returns_unit_direction():
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 0.0], [0.8, 0.1], [-1.0, 0.0], [-0.8, -0.1]])
    y = torch.tensor([1.0, 1.0, 0.0, 0.0])

    direction = train_probe_direction(
        x,
        y,
        {
            "device": "cpu",
            "seed": 0,
            "epochs": 2,
            "batch_size": 2,
            "lr": 0.1,
            "weight_decay": 0.0,
        },
    )

    assert_unit_norm(direction)


def test_realizable_basis_calibration_prefers_text_and_prefix_boundary():
    rows = pd.DataFrame(
        [
            {
                "prefix": "abc ",
                "completion": "def",
                "text": "abc def",
                "input": "collapsed text should not be used",
            }
        ]
    )

    text_col = _calibration_text_column(rows)
    starts = calibration_start_positions(
        OffsetTokenizer(),
        rows,
        text_col,
        max_length=None,
    )

    assert text_col == "text"
    assert starts == [4]


def test_completion_prediction_mask_matches_or_position_selection():
    tokenizer = OffsetTokenizer()
    text = "abc def"
    encoded = tokenizer(
        [text],
        add_special_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )

    mask = completion_prediction_mask(
        tokenizer,
        torch.tensor(encoded["input_ids"]),
        torch.tensor(encoded["attention_mask"]),
        [text],
        ["abc "],
        max_length=None,
    )

    assert torch.where(mask[0])[0].tolist() == [4, 5, 6]


def test_realizable_basis_examples_reject_collapsed_input_rows():
    rows = pd.DataFrame(
        [
            {
                "input": "prefix completion",
                "labels": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="Or-style rows"):
        validate_realizable_basis_examples(rows)
