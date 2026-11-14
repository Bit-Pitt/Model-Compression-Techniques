from copy import copy

import torch.nn as nn
import io, json, os 
import io
import json
import os
import torch
import time


# Stime di energia consumate dalle operazioni di inferenza (moltiplicazione e somma) in microjoule (uJ)
# Horowitz, ISSCC 2014 (45 nm), solo aritmetica, in pJ.
# int8: moltiplicazione a 8 bit, accumulo in int32 (come nell'inferenza int8 reale).
ENERGY_PJ = {
    "fp32": {"mul": 3.7, "add": 0.9},
    "int8": {"mul": 0.2, "add": 0.1},
}

# Semplifico la SRAM stimata necessaria come  max(input+output)*bytes   [ping-pong method]
def peak_activation_bytes(model, input_size=(1, 3, 32, 32), bytes_per_elem=4):
    """Picco di memoria delle attivazioni in inferenza: max sui layer di (input + output)."""
    peak = 0

    def hook(m, inp, out):
        nonlocal peak
        peak = max(peak, (inp[0].numel() + out.numel()) * bytes_per_elem)

    hooks = [m.register_forward_hook(hook)
             for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d))]

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(input_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return peak


# conta i MACs (Multiply-Accumulate) per un modello dato, input_size = (batch, canali, H, W)
def count_macs(model, input_size=(1, 3, 32, 32), skip_zeros=False):
    """MACs per singola immagine. Conv: pesi * H_out * W_out. Linear: pesi."""
    macs = 0

    def n_weights(m):
        return m.weight.count_nonzero().item() if skip_zeros else m.weight.numel()

    def conv_hook(m, inp, out):
        nonlocal macs
        macs += n_weights(m) * out.shape[2] * out.shape[3]

    def linear_hook(m, inp, out):
        nonlocal macs
        macs += n_weights(m)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(input_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


def estimate_energy_uj(macs, dtype="fp32"):
    """Energia stimata per inferenza in microjoule (pJ -> uJ: * 1e-6)."""
    e = ENERGY_PJ[dtype]
    return macs * (e["mul"] + e["add"]) * 1e-6

# misura la latenza di inferenza su CPU (batch 1) e restituisce mediana e std in ms
import copy
@torch.no_grad()
def measure_latency_ms(model, input_size=(1, 3, 32, 32), n_warmup=20, n_runs=100, threads=1):
    """Latenza su CPU, batch 1, mediana e std su n_runs (dopo warmup)."""
    model = copy.deepcopy(model).cpu().eval()
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    x = torch.randn(*input_size)
    for _ in range(n_warmup):
        model(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev_threads)
    t = torch.tensor(times)
    return {"latency_ms_median": t.median().item(), "latency_ms_std": t.std().item()}


BYTES_PER_ELEM = {"fp32": 4, "int8": 1}

# Salvo i risultati in un file JSON usato poi per notebook evaluation.
def profile_model(model, dtype="fp32", skip_zeros=False, input_size=(1, 3, 32, 32),
                  size_mb=None, latency=True):
    macs = count_macs(model, input_size, skip_zeros)
    peak = peak_activation_bytes(model, input_size, BYTES_PER_ELEM[dtype])
    out = {
        "params": count_parameters(model)["total"],
        "size_mb": model_size_mb(model) if size_mb is None else size_mb,
        "macs": macs,
        "energy_uj": estimate_energy_uj(macs, dtype),
        "peak_act_kb": peak / 1e3,
    }
    if latency:
        out.update(measure_latency_ms(model, input_size))
    return out

def update_result(name, results_dir="../results", **fields):
    """Aggiunge campi a un JSON già esistente senza perdere quelli vecchi."""
    path = f"{results_dir}/{name}.json"
    with open(path) as f:
        data = json.load(f)
    data.update(fields)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)



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


def model_size_mb(model):
    """Dimensione della state_dict serializzata (include i buffer dei BN)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def save_result(name, results_dir="../results", **fields):
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/{name}.json", "w") as f:
        json.dump({"name": name, **fields}, f, indent=2)