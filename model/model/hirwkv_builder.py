# models/hirwkv_builder.py
from __future__ import annotations

from typing import Any, Dict, List
import torch.nn as nn

from vrwkv_IP import HSI_RWKV


def build_rwkv_stack(
    dim: int,
    group_num: int,
    variant_spec: List[Dict[str, Any]],
) -> nn.Sequential:
    layers: List[nn.Module] = []

    for op in variant_spec:
        t = op["type"].lower()

        if t == "rwkv":
            num_blocks = op.get("num_blocks", [1])
            layers.append(HSI_RWKV(dim=dim, num_blocks=num_blocks))

        elif t == "pool":
            mode = op.get("mode", "max").lower()
            k = int(op.get("k", 2))
            s = int(op.get("s", k))
            if mode == "max":
                layers.append(nn.MaxPool2d(kernel_size=k, stride=s))
            elif mode == "avg":
                layers.append(nn.AvgPool2d(kernel_size=k, stride=s))
            else:
                raise ValueError(f"Unknown pool mode: {mode}")

        elif t == "norm":
            name = op.get("name", "groupnorm").lower()
            if name == "groupnorm":
                layers.append(nn.GroupNorm(group_num, dim))
            else:
                raise ValueError(f"Unknown norm: {name}")

        elif t == "act":
            name = op.get("name", "silu").lower()
            if name == "silu":
                layers.append(nn.SiLU())
            elif name == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif name == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {name}")

        else:
            raise ValueError(f"Unknown op type: {t}")

    return nn.Sequential(*layers)
