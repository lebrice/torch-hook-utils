from __future__ import annotations

import torch
import torch.distributions as dist
from torch import nn
from torch_hook_utils import named_modules_of_type, replace_layer_inputs
from torch_hook_utils.layers import Lambda, Sample, Sequential


def test_replace_inputs():
    model = Sequential(
        nn.Flatten(),
        nn.LazyLinear(out_features=1),
        Lambda(dist.Normal, scale=0.1),
        Sample(),
    )

    input = torch.rand(1, 4)
    sample_layers = named_modules_of_type(model, Sample)

    # NOTE: This as rich typing information: the inputs should be `Distribution` objects!
    with replace_layer_inputs(sample_layers, {"3": dist.Normal(0, 0.0001)}):
        output = model(input)
    assert output.abs
    print(output)
