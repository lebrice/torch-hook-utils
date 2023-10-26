import torch

from hook_utils.types import named_modules_of_type
from hook_utils import get_layer_inputs
from hook_utils.layers import Sequential, Lambda, Sample
from torch import nn
import torch.distributions as dist

model = Sequential(
    nn.Flatten(),
    nn.LazyLinear(out_features=1),
    Lambda(dist.Normal, scale=0.1),
    Sample(),
)

input = torch.rand(1, 4)
sample_layers = named_modules_of_type(model, Sample)
with get_layer_inputs(sample_layers) as layer_inputs:
    output = model(input)
print(layer_inputs)
