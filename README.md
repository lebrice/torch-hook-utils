# torch-hook-utils

This small module contains a few utility functions to make some complicated ML code much easier to write.
This is made possible by the hook system of PyTorch.

## Extracting/replacing layer inputs and outputs
Here are the functions:
- `get_layer_inputs`: Context manager that collects the layer's inputs during the forward pass.
- `get_layer_outputs`: Context manager that collects the outputs of the given layers during a forward pass.
- `replace_layer_inputs`: Context manager that replaces the inputs to the given layers during the forward pass.
- `replace_layer_output`: Context manager that replaces the outputs of the layers after the forward pass.


```python
import torch
import torch.distributions as dist
from torch import nn

from torch_hook_utils import get_layer_inputs, named_modules_of_type
from torch_hook_utils.layers import Lambda, Sample

# Some network
model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(out_features=1),
    Lambda(dist.Normal, scale=0.1),
    Sample(),
)

input = torch.rand(1, 4)
sample_layers = named_modules_of_type(model, Sample)

# Capture the inputs to all the `Sample` layers 
with get_layer_inputs(sample_layers) as layer_inputs:
    output = model(input)
print(layer_inputs)  # dict[str, Normal] that contains all the inputs to the Sample layers
```


This also contains some functions that make it much easier to use `nn.MaxPool` - `nn.MaxUnpool` layers to build encoder/decoder networks:
- `save_maxpool_indices`: Saves the max indices of the `nn.MaxPool` layers during the forward pass.
- `use_indices_in_maxunpool` : Use the given indices as the `indices` argument to the `nn.MaxUnpool` layers, 

For example:
```python

encoder = nn.Sequential(
    OrderedDict(pool=nn.MaxPool2d(kernel_size=2)),
)
with save_maxpool_indices(encoder) as maxpool_indices:
    encoder_output = encoder(
        torch.as_tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        )
    )
print(maxpool_indices)  # {"pool": tensor([[[3]]])}
print(encoder_output)  # tensor([[[4.0]]]))

decoder = nn.Sequential(
    OrderedDict(pool=nn.MaxUnpool2d(kernel_size=2)),
)
with use_indices_in_maxunpool(decoder, maxpool_indices):
    decoder_output = decoder(encoder_output)

print(decoder_output)
# tensor([[[0., 0.],
#          [0., 4.]]])
```

## Implementation

The general pattern goes like this:

```python
def some_context_manager(some_layers: dict[str, nn.Module]):
    # Add some PyTorch hooks to the layers 

    yield  # yield something (e.g. a dictionary)

    # remove the hooks.
```

Feel free to contibute some more ideas if you have some.