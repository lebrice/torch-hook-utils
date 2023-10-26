from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple

import pytest
import torch
import torch.testing
from torch import Tensor, nn

from torch_hook_utils.maxpool_hooks import save_maxpool_indices, use_indices_in_maxunpool


def test_maxpool_indices_example():
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
    torch.testing.assert_close(
        maxpool_indices,
        {
            "pool": torch.as_tensor([[[3]]]),
        },
    )
    torch.testing.assert_close(encoder_output, torch.as_tensor([[[4.0]]]))

    decoder = nn.Sequential(
        OrderedDict(pool=nn.MaxUnpool2d(kernel_size=2)),
    )
    with use_indices_in_maxunpool(decoder, maxpool_indices):
        decoder_output = decoder(encoder_output)

    torch.testing.assert_close(
        decoder_output,
        torch.as_tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 4.0],
                ]
            ]
        ),
    )


class Args(NamedTuple):
    net: nn.Module
    input: Tensor
    indices: dict[str, Tensor]
    expected_output: Tensor


@pytest.mark.parametrize(
    ("net", "input", "expected_indices", "expected_output"),
    [
        Args(
            net=nn.Sequential(
                OrderedDict(pool=nn.MaxPool2d(kernel_size=2, return_indices=False)),
            ),
            input=torch.as_tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                ]
            ),
            indices={
                "pool": torch.as_tensor([[[3]]]),
            },
            expected_output=torch.as_tensor([[[4.0]]]),
        ),
    ],
)
def test_save_maxpool_indices(
    net: nn.Module, input: Tensor, expected_indices: dict[str, Tensor], expected_output: Tensor
):
    with save_maxpool_indices(net) as indices:
        output = net(input)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(indices, expected_indices)


@pytest.mark.parametrize(
    ("net", "input", "indices", "expected_output"),
    [
        Args(
            net=nn.Sequential(
                OrderedDict(pool=nn.MaxUnpool2d(kernel_size=2)),
            ),
            input=torch.as_tensor([[[4.0]]]),
            indices={
                "pool": torch.as_tensor([[[3]]]),
            },
            expected_output=torch.as_tensor(
                [
                    [
                        [0.0, 0.0],
                        [0.0, 4.0],
                    ]
                ]
            ),
        ),
    ],
)
def test_use_maxpool_indices(
    net: nn.Module, input: Tensor, indices: dict[str, Tensor], expected_output: Tensor
):
    with use_indices_in_maxunpool(net, indices):
        out = net(input)
    torch.testing.assert_close(out, expected_output)


class FullArgs(NamedTuple):
    encoder: nn.Module
    decoder: nn.Module
    input: Tensor
    indices: dict[str, Tensor]
    encoder_output: Tensor
    decoder_output: Tensor


@pytest.mark.parametrize(
    (
        "encoder",
        "decoder",
        "input",
        "expected_indices",
        "expected_encoder_output",
        "expected_decoder_output",
    ),
    [
        FullArgs(
            encoder=nn.Sequential(
                OrderedDict(pool=nn.MaxPool2d(kernel_size=2)),
            ),
            decoder=nn.Sequential(
                OrderedDict(pool=nn.MaxUnpool2d(kernel_size=2)),
            ),
            input=torch.as_tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ]
                ]
            ),
            indices={
                "pool": torch.as_tensor([[[3]]]),
            },
            encoder_output=torch.as_tensor([[[4.0]]]),
            decoder_output=torch.as_tensor(
                [
                    [
                        [0.0, 0.0],
                        [0.0, 4.0],
                    ]
                ]
            ),
        ),
    ],
)
def test_full_loop(
    encoder: nn.Module,
    decoder: nn.Module,
    input: Tensor,
    expected_indices: dict[str, Tensor],
    expected_encoder_output: Tensor,
    expected_decoder_output: Tensor,
):
    with save_maxpool_indices(encoder) as maxpool_indices:
        encoder_output = encoder(input)

    torch.testing.assert_close(maxpool_indices, expected_indices)
    torch.testing.assert_close(encoder_output, expected_encoder_output)

    with use_indices_in_maxunpool(decoder, maxpool_indices):
        decoder_output = decoder(encoder_output)

    torch.testing.assert_close(decoder_output, expected_decoder_output)


@pytest.mark.parametrize("consume_indices", [True, False])
def test_using_mapping_instead_of_module(consume_indices: bool):
    encoder_layer = nn.MaxPool2d(kernel_size=2)
    decoder_layer = nn.MaxUnpool2d(kernel_size=2)

    encoder = nn.Sequential(
        OrderedDict(
            aaa=nn.Identity(),
            foobar=encoder_layer,
            bbb=nn.Identity(),
        ),
    )
    assert encoder.foobar is encoder_layer
    encoder_map = {"foobar": encoder_layer}
    decoder = nn.Sequential(
        OrderedDict(
            useless=nn.Identity(),
            baz=decoder_layer,
        ),
    )
    assert encoder[1] is encoder_layer and ("baz", decoder_layer) in decoder.named_children()
    decoder_map = {"baz": decoder_layer}

    input = torch.as_tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    expected_encoder_output = torch.as_tensor([[[4.0]]])
    expected_maxpool_indices = {
        "foobar": torch.as_tensor([[[3]]]),
    }
    expected_decoder_output = torch.as_tensor(
        [
            [
                [0.0, 0.0],
                [0.0, 4.0],
            ]
        ]
    )

    with save_maxpool_indices(encoder_map) as maxpool_indices_from_encoder:
        encoder_output = encoder(input)
    torch.testing.assert_close(encoder_output, expected_encoder_output)
    torch.testing.assert_close(maxpool_indices_from_encoder, expected_maxpool_indices)

    # NOTE: Need to translate the keys in the dict to match the layer of the decoder:
    maxpool_indices_for_decoder = {
        "baz": maxpool_indices_from_encoder["foobar"],
    }

    with use_indices_in_maxunpool(
        decoder_map, maxpool_indices_for_decoder, consume_indices=consume_indices
    ):
        decoder_output = decoder(encoder_output)
    torch.testing.assert_close(decoder_output, expected_decoder_output)
    if consume_indices:
        assert not maxpool_indices_for_decoder
