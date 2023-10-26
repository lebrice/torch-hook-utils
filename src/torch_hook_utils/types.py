import typing
from typing import Iterable, ParamSpec, Protocol

from torch import Tensor, nn
from typing_extensions import TypeVar

T = TypeVar("T", default=Tensor)
OutT = TypeVar("OutT", default=Tensor)
OutT_co = TypeVar("OutT_co", default=Tensor, covariant=True)
ModuleType = TypeVar("ModuleType", bound=nn.Module)
P = ParamSpec("P")


class Module(Protocol[P, OutT_co]):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT_co:
        ...

    # if typing.TYPE_CHECKING:
    #     __call__ = forward


def named_modules_of_type(
    module: nn.Module, module_type: type[ModuleType] | tuple[type[ModuleType], ...]
) -> Iterable[tuple[str, ModuleType]]:
    for name, mod in module.named_modules():
        if isinstance(mod, module_type):
            yield name, mod
