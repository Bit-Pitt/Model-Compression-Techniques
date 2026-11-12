import torch.nn as nn

# Conta in automatico il numero di parametri della rete
def count_parameters(model: nn.Module, verbose: bool = False) -> dict:
    counts = {"conv": 0, "linear": 0, "bn": 0}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            kind = "conv"
        elif isinstance(mod, nn.Linear):
            kind = "linear"
        elif isinstance(mod, nn.BatchNorm2d) and mod.affine:
            kind = "bn"
        else:
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        counts[kind] += n
        if verbose:
            print(f"{name} -> {n}")
    counts["total"] = counts["conv"] + counts["linear"] + counts["bn"]
    return counts


def print_param_summary(counts: dict, title: str = "model"):
    t = counts["total"]
    print(f"--- {title} params ---")
    print(f"Total:     {t:,}")
    for k in ("conv", "linear", "bn"):
        print(f"{k:<10} {counts[k]:,} ({100 * counts[k] / t:.2f}%)")