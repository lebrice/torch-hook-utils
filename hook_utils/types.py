from torch import nn, Tensor
import typing
from typing import ParamSpec, Iterable, Protocol
from typing_extensions import TypeVar

T = TypeVar("T", default=Tensor)
OutT = TypeVar("OutT", default=Tensor)
ModuleType = TypeVar("ModuleType", bound=nn.Module)
P = ParamSpec("P")


class Module(Protocol[P, OutT]):
    if typing.TYPE_CHECKING:

        def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT:
            ...

        __call__ = forward


def named_modules_of_type(
    module: nn.Module, module_type: type[ModuleType] | tuple[type[ModuleType], ...]
) -> Iterable[tuple[str, ModuleType]]:
    for name, mod in module.named_modules():
        if isinstance(mod, module_type):
            yield name, mod
