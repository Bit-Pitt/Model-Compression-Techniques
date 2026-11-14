import copy
import torch
import torch.nn as nn
from src.train import evaluate
from src.metrics import count_macs
from src.models import VGGNet
from src.metrics import count_macs, count_parameters


''' 

    UNSTRUCTURED PRUNING 
    Crea modelli sparse (con pesi azzerati) e calcola la loro accuratezza sulla val/test set.
    --> decisa ad esempio la sparsità 30% --> allora si usa la soglia 30% percentile dei valori assoluti dei pesi per azzerare i più piccoli.
    - compute_masks: calcola le maschere di pruning (soglia globale o per layer)
    - apply_masks: applica le maschere al modello, azzerando i pesi corrispondenti
    - count_nonzero_params: conta i pesi non nulli

    STRUCTURED PRUNING
    - stesso framework ma solo locale (per layer, altrimenti non avrebbe senso) e rimuovi interi filtri / neuroni sulla base della somma L1 dei pesi 
    - produce una rete effettivamente più piccola 
'''





def _prunable(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def _threshold(abs_w, sparsity):
    """Soglia tale che circa "sparsity" dei valori sia <= soglia (k-esimo valore più piccolo)."""
    k = max(1, int(sparsity * abs_w.numel()))
    return abs_w.kthvalue(k).values.item()


def compute_masks(model, sparsity, mode="global"):
    """mask = |w| > soglia. mode='global': soglia unica; 'layer': stessa sparsità in ogni layer."""
    layers = _prunable(model)
    if sparsity <= 0:
        return {n: torch.ones_like(m.weight.data, dtype=torch.bool) for n, m in layers}
    if mode == "global":
        thr = _threshold(torch.cat([m.weight.data.abs().flatten() for _, m in layers]), sparsity)
        return {n: m.weight.data.abs() > thr for n, m in layers}
    return {n: m.weight.data.abs() > _threshold(m.weight.data.abs().flatten(), sparsity)
            for n, m in layers}


def apply_masks(model, masks):
    """Copia del modello con i pesi mascherati azzerati (i bias non si toccano)."""
    pruned = copy.deepcopy(model)
    mods = dict(pruned.named_modules())
    for n, mask in masks.items():
        mods[n].weight.data.mul_(mask.float())
    return pruned


def count_nonzero_params(model):
    return sum(p.count_nonzero().item() for p in model.parameters())


def sparse_size_mb(model, value_bytes=4, index_bytes=2):
    """STIMA dello spazio in formato sparso: ogni peso non nullo = valore + indice. Bias densi fp32.
    Conviene rispetto al denso solo oltre ~67% di sparsità (6 byte per non nullo contro 4)."""
    nz = sum(m.weight.count_nonzero().item() for _, m in _prunable(model))
    biases = sum(m.bias.numel() for _, m in _prunable(model) if m.bias is not None)
    return (nz * (value_bytes + index_bytes) + biases * 4) / 1e6


def sweep(model, loader, device, sparsities, mode="global"):
    """Accuracy sulla val per ogni livello di sparsità, sempre partendo dal modello originale."""
    _, base_acc = evaluate(model, loader, device)
    rows = []
    for s in sparsities:
        pruned = apply_masks(model, compute_masks(model, s, mode))
        _, acc = evaluate(pruned, loader, device)
        rows.append(dict(mode=mode, sparsity=s, val_acc=acc, drop=(base_acc - acc) * 100,
                         nonzero=count_nonzero_params(pruned),
                         macs_sparse=count_macs(pruned, skip_zeros=True)))
        print(f"{mode:6s} sparsity {s:.2f} | val {acc * 100:.2f} (Δ {-(base_acc - acc) * 100:+.2f})")
    return base_acc, rows




'''
    Ora parte di quantizzazion per Structured Pruning 
'''


def _convs(model):
    return [m for m in model.conv_backbone if isinstance(m, nn.Conv2d)]


def _linears(model):
    return [m for m in model.classifier if isinstance(m, nn.Linear)]

def _keep_idx(scores, ratio):
    """Indici (ordinati) delle unità da TENERE: scarta la frazione `ratio` con punteggio minore."""
    assert 0.0 <= ratio < 1.0, f"ratio deve stare in [0, 1), trovato {ratio}"
    n = scores.numel()
    n_keep = max(1, n - int(round(ratio * n)))
    return scores.topk(n_keep).indices.sort().values


def _per_layer(ratio, n):
    """Accetta un numero (uguale per tutti i layer) o una lista con un valore per layer."""
    return list(ratio) if isinstance(ratio, (list, tuple)) else [ratio] * n


def structured_keep(model, conv_ratio=0.0, fc_ratio=0.0):
    """Modello GIÀ con BN fuso. Punteggio = norma L1 dei pesi del filtro/neurone, per layer.
    L'ultimo Linear (logits) non si pota."""
    convs, hidden = _convs(model), _linears(model)[:-1]
    cr, fr = _per_layer(conv_ratio, len(convs)), _per_layer(fc_ratio, len(hidden))
    keep_conv = [_keep_idx(c.weight.data.abs().sum(dim=(1, 2, 3)), r) for c, r in zip(convs, cr)]
    keep_fc = [_keep_idx(f.weight.data.abs().sum(dim=1), r) for f, r in zip(hidden, fr)]
    return keep_conv, keep_fc




def prune_structured(model, conv_ratio=0.0, fc_ratio=0.0):
    """Ritorna un VGGNet davvero più piccolo (BN-free): canali/neuroni rimossi, pesi copiati."""
    convs, fcs = _convs(model), _linears(model)
    keep_conv, keep_fc = structured_keep(model, conv_ratio, fc_ratio)

    # nuova architettura: stessa sequenza di conv/max_pool, meno canali
    it = iter(keep_conv)
    new_cfg = [len(next(it)) if isinstance(l, nn.Conv2d) else "max_pool"
               for l in model.conv_backbone if isinstance(l, (nn.Conv2d, nn.MaxPool2d))]
    new = VGGNet(new_cfg, hidden_dims=tuple(len(k) for k in keep_fc),
                 num_classes=fcs[-1].out_features, batch_norm=False, dropout=0.0)

    # conv: tengo i filtri `keep` (righe) e i canali di ingresso sopravvissuti al layer precedente
    prev = torch.arange(convs[0].in_channels)
    for old, new_m, keep in zip(convs, _convs(new), keep_conv):
        new_m.weight.data = old.weight.data[keep][:, prev].clone()
        new_m.bias.data = old.bias.data[keep].clone()
        prev = keep

    # ultimo conv -> prima FC: ogni canale vale hw colonne dopo il flatten (indice = canale*hw + pos)
    hw = fcs[0].in_features // convs[-1].out_channels
    prev = (prev.unsqueeze(1) * hw + torch.arange(hw)).flatten()

    # FC: righe = neuroni tenuti, colonne = uscite sopravvissute del layer precedente
    keeps = keep_fc + [torch.arange(fcs[-1].out_features)]
    for old, new_m, keep in zip(fcs, _linears(new), keeps):
        new_m.weight.data = old.weight.data[keep][:, prev].clone()
        new_m.bias.data = old.bias.data[keep].clone()
        prev = keep
    return new.eval()


def zero_out_structured(model, keep_conv, keep_fc):
    """Copia con filtri/neuroni scartati azzerati (peso e bias). Equivale alla rimozione perché
    non c'è BN e ReLU(0) = 0: serve solo come test di prune_structured."""
    out = copy.deepcopy(model)
    for layer, keep in list(zip(_convs(out), keep_conv)) + list(zip(_linears(out)[:-1], keep_fc)):
        drop = torch.ones(layer.weight.shape[0], dtype=torch.bool)
        drop[keep] = False
        layer.weight.data[drop] = 0
        layer.bias.data[drop] = 0
    return out


def skip_first_ratios(model, ratio, skip_first):
    """Lista di rapporti per layer: 0 per i primi `skip_first` conv, `ratio` per gli altri."""
    n = len(_convs(model))
    return [0.0] * skip_first + [ratio] * (n - skip_first)


def sweep_structured(model, loader, device, conv_ratios, fc_ratios, skip_first=0):
    """Griglia (conv_ratio x fc_ratio) sulla val, con il modello VERAMENTE potato.
    I primi `skip_first` conv non vengono potati."""
    _, base_acc = evaluate(model, loader, device)
    rows = []
    for cr in conv_ratios:
        conv_r = skip_first_ratios(model, cr, skip_first)
        for fr in fc_ratios:
            pruned = prune_structured(model, conv_r, fr)
            _, acc = evaluate(pruned, loader, device)
            rows.append(dict(conv_ratio=cr, fc_ratio=fr, val_acc=acc, drop=(base_acc - acc) * 100,
                             params=count_parameters(pruned)["total"], macs=count_macs(pruned)))
            print(f"conv {cr:.2f} | fc {fr:.2f} | val {acc * 100:.2f} (Δ {-(base_acc - acc) * 100:+.2f}) | "
                  f"{rows[-1]['params']:,} params | {rows[-1]['macs'] / 1e6:.1f}M MACs")
    return base_acc, rows

def layer_sensitivity(model, loader, device, ratios):
    """Pota UN solo conv alla volta (gli altri intatti): calo di accuracy per layer e frazione."""
    _, base_acc = evaluate(model, loader, device)
    n = len(_convs(model))
    rows = []
    for l in range(n):
        for r in ratios:
            cr = [0.0] * n
            cr[l] = r
            _, acc = evaluate(prune_structured(model, cr, 0.0), loader, device)
            rows.append(dict(layer=l, ratio=r, val_acc=acc, drop=(base_acc - acc) * 100))
            print(f"conv {l} | ratio {r:.2f} | Δ {-(base_acc - acc) * 100:+.2f}")
    return base_acc, rows