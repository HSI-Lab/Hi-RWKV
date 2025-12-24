import torch.nn as nn
from .model.vrwkv_IP import HSI_RWKV

def build_rwkv_stack(hidden_dim: int, group_num: int, spec: list) -> nn.Sequential:
    """
    Build nn.Sequential for rwkv trunk from a list of ops.

    Each op is a dict-like object, e.g.
      {"type": "rwkv", "num_blocks": [1]}
      {"type": "pool", "mode": "max", "k": 2, "s": 2}
      {"type": "norm", "name": "groupnorm"}
      {"type": "act", "name": "silu"}
    """
    if spec is None or len(spec) == 0:
        raise ValueError("rwkv_spec is empty. Please provide a non-empty spec list.")

    layers = []
    for op in spec:
        if not isinstance(op, dict):
            raise TypeError(f"Each rwkv op must be a dict, got: {type(op)}")

        t = str(op.get("type", "")).lower().strip()
        if t == "rwkv":
            num_blocks = op.get("num_blocks", [1])
            layers.append(HSI_RWKV(dim=hidden_dim, num_blocks=num_blocks))

        elif t == "pool":
            mode = str(op.get("mode", "max")).lower().strip()
            k = int(op.get("k", 2))
            s = int(op.get("s", k))
            if mode == "max":
                layers.append(nn.MaxPool2d(kernel_size=k, stride=s))
            elif mode == "avg":
                layers.append(nn.AvgPool2d(kernel_size=k, stride=s))
            else:
                raise ValueError(f"Unknown pool mode: {mode}")

        elif t == "norm":
            name = str(op.get("name", "groupnorm")).lower().strip()
            if name == "groupnorm":
                layers.append(nn.GroupNorm(group_num, hidden_dim))
            else:
                raise ValueError(f"Unknown norm: {name}")

        elif t == "act":
            name = str(op.get("name", "silu")).lower().strip()
            if name == "silu":
                layers.append(nn.SiLU())
            elif name == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif name == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {name}")

        else:
            raise ValueError(f"Unknown op type: {t}. Supported: rwkv/pool/norm/act")

    return nn.Sequential(*layers)


class HiRWKV(nn.Module):
    """
    HiRWKV with configurable RWKV trunk.

    Parameters
    ----------
    rwkv_spec: list[dict] | None
        If provided, build self.rwkv from this spec.
        If None, fall back to default HANCHUAN-like stack (backward compatible).
    dataset_name: str | None
        Optional tag for logging / debugging.
    """
    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
        num_classes=10,
        group_num=1,
        rwkv_spec=None,
        dataset_name: str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        # Default (backward compatible): your original HANCHUAN trunk
        default_spec = [
            {"type": "rwkv", "num_blocks": [1]},
            {"type": "pool", "mode": "max", "k": 2, "s": 2},
            {"type": "act", "name": "silu"},
            {"type": "rwkv", "num_blocks": [1]},
            {"type": "pool", "mode": "max", "k": 2, "s": 2},
            {"type": "act", "name": "silu"},
            {"type": "rwkv", "num_blocks": [1]},
        ]
        spec = default_spec if rwkv_spec is None else rwkv_spec

        self.rwkv = build_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=spec)

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.rwkv(x)
        logits = self.cls_head(x)
        return logits
