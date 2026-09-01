from __future__ import annotations

import torch

from models.wecan import (
    LDDGraphAttention,
    NEG_INF,
    POS_INF,
    WeCAN,
    WeCANConfig,
    fold_ldd_distance,
    ldd_attention_mask,
    ldd_bias_indices,
)
from tests.fixtures import handcrafted_instances


def test_ldd_folding_and_bias_indices_have_confirmed_boundary_semantics() -> None:
    values = torch.tensor([-501, -500, -499, 499, 500, 501, POS_INF, NEG_INF])
    assert [fold_ldd_distance(int(value)) for value in values] == [-499, -499, -499, 499, 499, 499, 500, -500]
    assert ldd_bias_indices(values).tolist() == [1, 1, 1, 999, 999, 999, 1000, 0]


def test_paper_profile_has_required_shapes_and_single_forward() -> None:
    instance = handcrafted_instances()["diamond"]
    config = WeCANConfig(
        profile="paper", high_dim=512, low_dim=128, weca_heads=8, ldd_heads=16,
        ldd_layers=8, dmax=500, skip_parameterization="paper_sigmoid",
    )
    model = WeCAN(instance.resource_dims, config)
    output = model(instance)
    assert len(model.ldd_layers) == 8
    assert model.ldd_layers[0].heads == 16
    assert model.task_projection.in_features == 512
    assert model.task_projection.out_features == 128
    assert output.task_pool_scores.shape == (instance.num_tasks, instance.num_pools)
    assert torch.all((output.skip_parameters > 0) & (output.skip_parameters < 1))
    assert model.forward_calls == 1


def test_per_head_mask_matches_eight_groups_for_sixteen_heads() -> None:
    values = torch.tensor([[0, 1, -1, 2, -2, 3, -3, POS_INF, NEG_INF]])
    mask = ldd_attention_mask(values, 16)
    assert mask.shape == (16, 1, 9)
    assert mask[0].all() and mask[1].all()  # N1
    assert mask[2, 0, 1] and mask[3, 0, 1]  # N2
    assert mask[4, 0, 2] and mask[5, 0, 2]  # N3
    assert mask[6, 0, 3] and mask[7, 0, 3]  # N4
    assert mask[8, 0, 4] and mask[9, 0, 4]  # N5
    assert mask[10, 0, 5] and mask[11, 0, 5]  # N6
    assert mask[12, 0, 6] and mask[13, 0, 6]  # N7
    assert mask[14, 0, 7] and mask[15, 0, 7]  # N8
    assert not mask[14, 0, 0]


def test_empty_specialized_head_outputs_are_zero_before_output_projection() -> None:
    layer = LDDGraphAttention(hidden_dim=8, heads=8, dmax=500)
    nodes = torch.randn(2, 8)
    distances = torch.tensor([[0, NEG_INF], [NEG_INF, 0]])
    mask = ldd_attention_mask(distances, 8)
    assert not mask[1:].any()  # only N1 sees nodes; every specialized head is empty.
    output = layer(nodes, distances)
    assert torch.isfinite(output).all()


def test_softplus_skip_ablation_stays_strictly_positive() -> None:
    instance = handcrafted_instances()["single_chain"]
    model = WeCAN(instance.resource_dims, WeCANConfig(profile="smoke", high_dim=32, low_dim=32, weca_heads=4, ldd_heads=4, ldd_layers=1, skip_parameterization="softplus"))
    output = model(instance)
    assert torch.all(output.skip_parameters > 0)
