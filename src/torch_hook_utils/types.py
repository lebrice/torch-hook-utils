from typing import Iterable, Protocol, ParamSpec
import typing
from torch import nn
from typing_extensions import TypeVar

T = TypeVar("T")
OutT = TypeVar("OutT")
OutT_co = TypeVar("OutT_co", covariant=True)
P = ParamSpec("P")


class Module(Protocol[P, OutT_co]):
    def forward(self, *args: P.args, **kwargs: P.kwargs) -> OutT_co:
        ...

    if typing.TYPE_CHECKING:

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> OutT_co:
            ...


ModuleType = TypeVar("ModuleType", bound=Module, default=nn.Module)


def named_modules_of_type(
    module: nn.Module, module_type: type[ModuleType] | tuple[type[ModuleType], ...]
) -> Iterable[tuple[str, ModuleType]]:
    for name, mod in module.named_modules():
        if isinstance(mod, module_type):
            yield name, mod
