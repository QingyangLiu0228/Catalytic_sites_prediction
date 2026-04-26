import torch

_path = "/data/data3/conglab/s441865/esm2_embeddings_fasta/sp|1a0i_A|esm.pt"
# Old PyTorch: do not pass weights_only (not supported; breaks unpickler).
# New PyTorch: optional weights_only=False for full pickle loads.
try:
    d = torch.load(_path, map_location="cpu", weights_only=False)
except TypeError:
    d = torch.load(_path, map_location="cpu")

print("type:", type(d))

def _describe(x, name="obj", depth=0):
    ind = "  " * depth
    if torch.is_tensor(x):
        print(f"{ind}{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    elif isinstance(x, dict):
        print(f"{ind}{name}: dict, len={len(x)} keys (first 40): {list(x.keys())[:40]}")
        for k in list(x.keys())[:40]:
            _describe(x[k], name=f"{k!r}", depth=depth + 1)
        if len(x) > 40:
            print(f"{ind}  ... ({len(x) - 40} more keys)")
    elif isinstance(x, (list, tuple)):
        print(f"{ind}{name}: {type(x).__name__}, len={len(x)}")
        for i, v in enumerate(x[:5]):
            _describe(v, name=f"[{i}]", depth=depth + 1)
        if len(x) > 5:
            print(f"{ind}  ... ({len(x) - 5} more items)")
    else:
        s = repr(x)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"{ind}{name}: {type(x).__name__} -> {s}")

_describe(d, "root")

# 若与 Catalytic 代码一致: d["representations"][layer] -> (L, 2560)
reps = None
if isinstance(d, dict) and "representations" in d:
    reps = d["representations"]
elif isinstance(d, dict) and d:
    first = next(iter(d.values()))
    if isinstance(first, dict) and "representations" in first:
        reps = first["representations"]
if isinstance(reps, dict):
    print("representations layer indices:", sorted(reps.keys()))
    for L in (33, 0, 36):
        if L in reps and torch.is_tensor(reps[L]):
            print(f"  layer {L} shape: {tuple(reps[L].shape)}")
