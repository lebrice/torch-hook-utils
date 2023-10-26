from __future__ import annotations

import contextlib
from functools import partial
from typing import Any, Generator, Iterable

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from typing_extensions import Concatenate

from .types import Module, OutT, T


@contextlib.contextmanager
def get_layer_inputs(
    layers: Module[[T], OutT] | Iterable[tuple[str, Module[[T], OutT]]]
) -> Generator[dict[str, T], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, nn.Module) else layers
    layer_inputs: dict[str, T] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_input_hook,
            name,
            layer_inputs,
        )
        hook_handle = layer.register_forward_pre_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_inputs

    for hook_handle in hook_handles:
        hook_handle.remove()


@contextlib.contextmanager
def get_layer_outputs(
    layers: Module[..., OutT] | Iterable[tuple[str, Module[..., OutT]]]
) -> Generator[dict[str, OutT], None, None]:
    named_layers = layers.named_modules() if isinstance(layers, nn.Module) else layers
    layer_outputs: dict[str, OutT] = {}
    hook_handles: list[RemovableHandle] = []

    for name, layer in named_layers:
        assert isinstance(layer, nn.Module)
        layer_hook = partial(
            _save_output_hook,
            name,
            layer_outputs,
        )
        hook_handle = layer.register_forward_hook(layer_hook, with_kwargs=True)
        hook_handles.append(hook_handle)

    yield layer_outputs

    for hook_handle in hook_handles:
        hook_handle.remove()


def _save_input_hook(
    _layer_name: str,
    _inputs: dict[str, T],
    /,
    module: Module[Concatenate[T, ...], OutT],
    args: tuple[T, ...],
    kwargs: dict[str, Any],
):
    """Saves the layer inputs and outputs in the given dictionaries.

    This should only really be used with a `functools.partial` that pre-binds the layer name,
    as well as the input and output dictionaries.

    NOTE: This only saves the first input to the network.
    """
    assert len(args) == 1, "TODO: Assumes a single input per layer atm."
    _inputs[_layer_name] = args[0]


def _save_output_hook(
    _layer_name: str,
    _outputs: dict[str, OutT],
    /,
    module: Module[Concatenate[T, ...], OutT],
    args: tuple[T, ...],
    kwargs: dict[str, Any],
    output: OutT,
):
    """Saves the layer inputs and outputs in the given dictionaries.

    This should only really be used with a `functools.partial` that pre-binds the layer name,
    as well as the input and output dictionaries.

    NOTE: This only saves the first input to the network.
    """
    _outputs[_layer_name] = output


@contextlib.contextmanager
def replace_layer_inputs(
    network_layers: Iterable[tuple[str, nn.Module]],
    inputs_to_use: dict[str, Tensor],
):
    """Context that temporarily makes these layers use these inputs in their forward pass."""
    handles: list[RemovableHandle] = []
    for name, layer in network_layers:
        handle = layer.register_forward_pre_hook(
            partial(_replace_layer_input_hook, name, inputs_to_use)
        )
        handles.append(handle)

    yield

    for handle in handles:
        handle.remove()


def _replace_layer_input_hook(
    _name: str,
    _inputs_to_use: dict[str, Tensor],
    module: nn.Module,
    args: tuple[Tensor, ...],
):
    assert len(args) == 1
    replaced_inputs = _inputs_to_use[_name]
    if not isinstance(replaced_inputs, tuple):
        replaced_inputs = (replaced_inputs,)
    assert len(replaced_inputs) == len(args)
    assert all(
        replaced_input.shape == arg.shape for replaced_input, arg in zip(replaced_inputs, args)
    )
    return replaced_inputs
