from __future__ import annotations

import functools
import operator
import typing
from collections import OrderedDict
from typing import Callable, Generic, Iterator, Self, Sequence, Tuple, overload

import torch
from torch import Tensor, nn
from torch._jit_internal import _copy_to_script_wrapper
from typing_extensions import Concatenate, ParamSpec, TypeVar, TypeVarTuple, Unpack

from .types import Module, ModuleType

P = ParamSpec("P")
R = ParamSpec("R")
OutT = TypeVar("OutT", default=Tensor)
Ts = TypeVarTuple("Ts", default=Unpack[Tuple[Tensor, ...]])
T = TypeVar("T", default=Tensor)


class Sequential(nn.Sequential, Sequence[ModuleType]):
    # Small typing fixes for torch.nn.Sequential
    @overload
    def __init__(self, *args: ModuleType) -> None:
        ...

    @overload
    def __init__(self, **kwargs: ModuleType) -> None:
        ...

    @overload
    def __init__(self, arg: dict[str, ModuleType]) -> None:
        ...

    def __init__(self, *args, **kwargs):
        if args:
            assert not kwargs, "can only use *args or **kwargs, not both"

            new_args = []
            for arg in args:
                if not isinstance(arg, nn.Module) and callable(arg):
                    arg = Lambda(arg)
                new_args.append(arg)
            args = new_args

        if kwargs:
            assert not args, "can only use *args or **kwargs, not both"
            new_kwargs = {}
            for name, module in kwargs.items():
                if not isinstance(module, nn.Module) and callable(module):
                    module = Lambda(module)
                new_kwargs[name] = module
            kwargs = new_kwargs

            args = (OrderedDict(kwargs),)

        super().__init__(*args)

    @overload
    def __getitem__(self, idx: int) -> ModuleType:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Self:
        ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int | slice) -> Self | ModuleType:
        if isinstance(idx, slice):
            # NOTE: Fixing this here, subclass constructors shouldn't be called on getitem with
            # slice.
            return type(self)(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __iter__(self) -> Iterator[ModuleType]:
        return super().__iter__()  # type: ignore

    # Violates the LSP, but eh.
    def __setitem__(self, idx: int, module: ModuleType) -> None:
        return super().__setitem__(idx, module)

    def forward(self, *args, **kwargs):
        # Slightly modified from torch.nn.Sequential: *args and **kwargs can now be passed and are
        # sent to the first module.
        out = None
        for i, module in enumerate(self):
            if i == 0:
                out = module(*args, **kwargs)  # type: ignore
            else:
                out = module(out)  # type: ignore
        assert out is not None
        return out


class Lambda(nn.Module, Generic[OutT]):
    """A simple nn.Module wrapping a function.

    Any positional or keyword arguments passed to the constructor are pre-bound to the function
    using a `functools.partial`.
    """

    @overload
    def __init__(
        self,
        f: Callable[Concatenate[Tensor, P], OutT],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        ...

    @overload
    def __init__(
        self,
        f: Callable[Concatenate[Tensor, Tensor, P], OutT],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        ...

    def __init__(self, f: Callable[..., OutT], *args, **kwargs):
        super().__init__()
        self.f: Callable[..., OutT]
        if args or kwargs:
            self.f = functools.partial(f, *args, **kwargs)
        else:
            self.f = f

    def forward(self, *args, **kwargs) -> OutT:
        return self.f(*args, **kwargs)

    def extra_repr(self) -> str:
        if not isinstance(self.f, functools.partial):
            return f"f={self.f.__module__}.{self.f.__name__}"
        partial_args: list[str] = []
        if not isinstance(self.f.func, nn.Module):
            partial_args.append(f"{self.f.func.__module__}.{self.f.func.__name__}")
        else:
            partial_args.append(repr(self.f.func))
        partial_args.extend(repr(x) for x in self.f.args)
        partial_args.extend(f"{k}={v!r}" for (k, v) in self.f.keywords.items())

        partial_qualname = type(self.f).__qualname__
        if type(self.f).__module__ == "functools":
            partial_qualname = f"functools.{partial_qualname}"
        return f"f={partial_qualname}(" + ", ".join(partial_args) + ")"

    if typing.TYPE_CHECKING:
        __call__ = forward


class Sample(Lambda, Module[[torch.distributions.Distribution], Tensor]):
    def __init__(self, differentiable: bool = False) -> None:
        super().__init__(f=operator.methodcaller("rsample" if differentiable else "sample"))
        self.differentiable = differentiable

    def forward(self, dist: torch.distributions.Distribution) -> Tensor:
        return super().forward(dist)

    def extra_repr(self) -> str:
        return f"differentiable={self.differentiable}"

    if typing.TYPE_CHECKING:
        __call__ = forward


bob = Sample(differentiable=True).__call__("1.23")
