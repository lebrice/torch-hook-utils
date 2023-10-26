from __future__ import annotations

import contextlib
from functools import partial
import inspect
from typing import Any, Generator, Iterable, TypeVar

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from typing_extensions import Concatenate, ParamSpec, TypeVarTuple, Unpack

from .types import Module, OutT, T

P = ParamSpec("P")
Input = TypeVar("Input")
T_co = TypeVar("T_co", covariant=True)
T_cot = TypeVar("T_cot", contravariant=True)
# Ts = TypeVarTuple("Ts")


@contextlib.contextmanager
def get_layer_inputs(
    named_layers: Iterable[tuple[str, Module[Concatenate[Input, P], Any]]]
) -> Generator[dict[str, Input], None, None]:
    layer_inputs: dict[str, Input] = {}
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
    named_layers: Iterable[tuple[str, Module[..., OutT]]]
) -> Generator[dict[str, OutT], None, None]:
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
    module: Module[Concatenate[T, P], Any],
    args: tuple[T, ...],  # Concatenate[T, Unpack[P.args]] (Can't do this with Python typing yet.)
    kwargs: dict[str, Any],  # Unpack[P.kwargs] (Can't do this with Python typing yet.)
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
    network_layers: Iterable[tuple[str, Module[Concatenate[T, P], Any]]],
    inputs_to_use: dict[str, T],
):
    """Context that temporarily makes these layers use these inputs in their forward pass."""
    handles: list[RemovableHandle] = []
    for name, layer in network_layers:
        assert isinstance(layer, nn.Module)
        handle = layer.register_forward_pre_hook(
            partial(_replace_layer_input_hook, inputs_to_use[name])
        )
        handles.append(handle)

    yield

    for handle in handles:
        handle.remove()


def _replace_layer_input_hook(
    _inputs_to_use: T,
    /,
    module: Module[Concatenate[T, P], Any],
    # args: tuple[T, Unpack[P.args]]  # (can't do this with Python typing yet.)
    args: tuple[T, Unpack[tuple[Any, ...]]],
):
    return (_inputs_to_use, *args[1:])


@contextlib.contextmanager
def replace_layer_outputs(
    network_layers: Iterable[tuple[str, Module[..., OutT]]],
    layer_outputs: dict[str, OutT],
):
    """Context that temporarily makes these layers return these values in their forward pass.

    NOTE: The forward pass still happens with the original arguments, but this value is returned
    instead.
    """
    handles: list[RemovableHandle] = []
    for name, layer in network_layers:
        assert isinstance(layer, nn.Module)
        handle = layer.register_forward_hook(
            partial(_replace_layer_output_hook, layer_outputs[name])
        )
        handles.append(handle)

    yield

    for handle in handles:
        handle.remove()


def _replace_layer_output_hook(
    _output_to_use: OutT,
    /,
    module: Module[P, OutT],
    args: tuple[Tensor, ...],  # Unpack[P.args] (Can't do this with Python typing yet.)
    kwargs: dict[str, Any],  # Unpack[P.kwargs] (Can't do this with Python typing yet.)
    output: OutT,
):
    return _output_to_use


@contextlib.contextmanager
def replace_layer_input_arguments(
    network_layers: Iterable[tuple[str, Module[Concatenate[Tensor, P]]]],
    *input_args_to_use: P.args,
    **input_kwargs_to_use: P.kwargs,
):
    handles: list[RemovableHandle] = []
    for name, layer in network_layers:
        assert isinstance(layer, nn.Module)
        handle = layer.register_forward_pre_hook(
            partial(
                _replace_layer_input_arguments_hook, name, input_args_to_use, input_kwargs_to_use
            )
        )
        handles.append(handle)

    yield

    for handle in handles:
        handle.remove()


Ts = TypeVarTuple("Ts")


def _replace_layer_input_arguments_hook(
    _name: str,
    _inputs_args_to_use: P.args,  # type: ignore
    _inputs_kwargs_to_use: P.kwargs,  # type: ignore
    /,
    module: nn.Module,  # Module[Concatenate[Unpack[Ts], P], OutT],
    args: Ts,  # type: ignore
):
    assert len(args) == 1
    forward_sig = inspect.signature(module.forward)

    # TODO: This is definitely not the right way to override an arbitrary arg.
    bound_args = forward_sig.bind_partial(
        *args,  # not correct, for sure.
        *_inputs_args_to_use,
        **_inputs_kwargs_to_use,
    )
    # bound_args.apply_defaults()
    return bound_args.args, bound_args.kwargs
