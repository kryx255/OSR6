from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model prediction. Run install.bat or use: uv sync --extra model"
        ) from exc
    return torch, nn


def resolve_torch_device(requested: str | object | None = "auto"):
    torch, _ = require_torch()
    if not isinstance(requested, str):
        return requested
    value = (requested or "auto").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch does not report an available CUDA device.")
        return torch.device("cuda")
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested} was requested, but PyTorch does not report an available CUDA device.")
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device: {requested}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index out of range: {requested}")
        return torch.device(value)
    raise RuntimeError(f"Unsupported inference device: {requested}")


def torch_runtime_info(requested: str | None = "auto") -> dict[str, object]:
    torch, _ = require_torch()
    device = resolve_torch_device(requested)
    info: dict[str, object] = {
        "torch_version": str(torch.__version__),
        "requested_device": requested or "auto",
        "execution_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if str(device).startswith("cuda"):
        index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
        info["cuda_device_name"] = str(torch.cuda.get_device_name(index))
    return info


def create_model(input_dim: int, channels: int, layers: int, kernel_size: int, dropout: float):
    torch, nn = require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (kernel_size - 1) // 2
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        def forward(self, x):  # type: ignore[no-untyped-def]
            residual = x if self.proj is None else self.proj(x)
            return self.drop(self.act(self.conv(x))) + residual

    class TinyTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = input_dim
            for index in range(layers):
                blocks.append(ConvBlock(in_channels, channels, dilation=2**index))
                in_channels = channels
            self.net = nn.Sequential(*blocks)
            self.head = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return torch.sigmoid(self.head(self.net(x))).squeeze(1)

    return TinyTCN()
