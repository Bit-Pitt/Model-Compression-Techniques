import copy
import torch
import torch.nn as nn
from src.train import evaluate
from src.metrics import count_parameters, count_macs


def get_targets(model):
    """Layer fattorizzabili, in ordine: tutti i conv + le Linear nascoste (non l'ultima)."""
    convs = [m for m in model.conv_backbone if isinstance(m, nn.Conv2d)]
    linears = [m for m in model.classifier if isinstance(m, nn.Linear)][:-1]
    return convs + linears


def as_matrix(layer):
    """Pesi come matrice [out, in*kh*kw] (per le Linear è già così)."""
    return layer.weight.data.reshape(layer.weight.shape[0], -1)


def compute_svds(model):
    """SVD di ogni layer target: si calcola UNA volta e si riusa per tutti i tau."""
    return [torch.linalg.svd(as_matrix(l), full_matrices=False) for l in get_targets(model)]


def rank_for_energy(S, tau):
    """Il più piccolo k che conserva almeno la frazione `tau` dell'energia (somma dei sigma^2)."""
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int((energy < tau).sum().item()) + 1
    return min(k, S.numel())


def breakeven(layer):
    """Sotto questo rango la fattorizzazione riduce i parametri: out*in / (out+in)."""
    out, inn = as_matrix(layer).shape
    return out * inn / (out + inn)


def factorize_linear(layer, U, S, Vh, k):
    """Linear(in, out) -> Linear(in, k, senza bias) + Linear(k, out, con bias)."""
    sq = S[:k].sqrt()
    A = U[:, :k] * sq                 # [out, k]
    B = sq.unsqueeze(1) * Vh[:k]      # [k, in]
    first = nn.Linear(layer.in_features, k, bias=False)
    second = nn.Linear(k, layer.out_features, bias=True)
    first.weight.data = B.clone()
    second.weight.data = A.clone()
    second.bias.data = layer.bias.data.clone()
    return first, second


def factorize_conv(layer, U, S, Vh, r):
    """Conv(Cin->Cout, kxk) -> Conv(Cin->r, kxk, senza bias) + Conv(r->Cout, 1x1, con bias)."""
    sq = S[:r].sqrt()
    A = U[:, :r] * sq                 # [Cout, r]
    B = sq.unsqueeze(1) * Vh[:r]      # [r, Cin*kh*kw]
    first = nn.Conv2d(layer.in_channels, r, layer.kernel_size,
                      stride=layer.stride, padding=layer.padding, bias=False)
    second = nn.Conv2d(r, layer.out_channels, kernel_size=1, bias=True)
    first.weight.data = B.reshape(r, layer.in_channels, *layer.kernel_size).clone()
    second.weight.data = A.reshape(layer.out_channels, r, 1, 1).clone()
    second.bias.data = layer.bias.data.clone()
    return first, second


def build_lowrank(model, svds, tau, skip_first=2, force=False):
    """Copia del modello con i layer fattorizzati (i due fattori sono inseriti 'piatti' nella Sequential).
    Un layer si fattorizza solo se il rango scelto fa risparmiare parametri (o se force=True).
    I primi `skip_first` conv non si toccano. Ritorna (modello, info per layer)."""
    model = copy.deepcopy(model).eval()
    plan, info = {}, []                               # plan: id(layer) -> (fattore1, fattore2)
    for idx, (layer, (U, S, Vh)) in enumerate(zip(get_targets(model), svds)):
        is_conv = isinstance(layer, nn.Conv2d)
        kind = "conv" if is_conv else "fc"
        k = rank_for_energy(S, tau)
        skip = is_conv and idx < skip_first
        if skip or not (force or k < breakeven(layer)):
            info.append(dict(layer=idx, kind=kind, full_rank=S.numel(), k=None))
            continue
        fact = factorize_conv if is_conv else factorize_linear
        plan[id(layer)] = fact(layer, U, S, Vh, k)
        info.append(dict(layer=idx, kind=kind, full_rank=S.numel(), k=k))

    def rebuild(seq):
        out = []
        for l in seq:
            out.extend(plan[id(l)] if id(l) in plan else [l])
        return nn.Sequential(*out)

    model.conv_backbone = rebuild(model.conv_backbone)
    model.classifier = rebuild(model.classifier)
    return model, info


def sweep_lowrank(model, loader, device, taus, skip_first=2):
    """Per ogni soglia di energia: accuracy sulla val, parametri e MACs del modello fattorizzato."""
    svds = compute_svds(model)
    _, base_acc = evaluate(model, loader, device)
    rows = []
    for tau in taus:
        lr_model, info = build_lowrank(model, svds, tau, skip_first)
        _, acc = evaluate(lr_model, loader, device)
        n_fact = sum(i["k"] is not None for i in info)
        rows.append(dict(tau=tau, val_acc=acc, drop=(base_acc - acc) * 100,
                         params=count_parameters(lr_model)["total"],
                         macs=count_macs(lr_model), n_factorized=n_fact))
        print(f"tau {tau:.2f} | val {acc * 100:.2f} (Δ {-(base_acc - acc) * 100:+.2f}) | "
              f"{rows[-1]['params']:,} params | {rows[-1]['macs'] / 1e6:.1f}M MACs | "
              f"{n_fact} layer fattorizzati")
    return base_acc, rows, svds