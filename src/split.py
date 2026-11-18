import copy
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.quant import FakeQuant
from src.metrics import estimate_energy_uj
from src.train import evaluate


FLASH_KB = 4096   # flash tipica di un ESP32 (ipotesi)
SRAM_KB = 300     # SRAM utilizzabile per le attivazioni (ipotesi)


@torch.no_grad()
def op_profile(model, input_size=(1, 3, 32, 32)):
    """Una riga per ogni Conv/Linear/MaxPool, in ordine di esecuzione: elementi in/out, MACs, parametri
    e indice del layer di calcolo a cui appartiene (un max-pool appartiene al conv che lo precede)."""
    model = copy.deepcopy(model).cpu().eval()
    ops, state = [], {"layer": -1}

    def hook(m, inp, out):
        if isinstance(m, nn.MaxPool2d):
            macs, params = 0, 0
        else:
            state["layer"] += 1
            spatial = out.shape[2] * out.shape[3] if isinstance(m, nn.Conv2d) else 1
            macs = m.weight.numel() * spatial
            params = m.weight.numel() + m.bias.numel()
    
        ops.append(dict(kind=type(m).__name__, layer=state["layer"], in_elems=inp[0].numel(),
                        out_elems=out.numel(), out_shape=tuple(out.shape[1:]),
                        macs=macs, params=params))

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d))]
    model(torch.zeros(input_size))
    for h in hooks:
        h.remove()
    return ops


def cut_labels(ops):
    """Etichetta leggibile per ogni n_head (0 = tutto sul fog)."""
    n_layers = max(o["layer"] for o in ops) + 1
    labels, n_conv, n_fc = ["input (tutto sul fog)"], 0, 0
    for i in range(n_layers):
        group = [o for o in ops if o["layer"] == i]
        if group[0]["kind"] == "Conv2d":
            n_conv += 1
            name = f"conv{n_conv}"
        else:
            n_fc += 1
            name = f"fc{n_fc}"
        if any(o["kind"] == "MaxPool2d" for o in group):
            name += "+pool"
        labels.append(f"dopo {name}" + (" (tutto sul device)" if i == n_layers - 1 else ""))
    return labels

# Costruisce il modelli splittato  (primi n layer quantizzati)
def build_split_model(folded, qmodel, n_head):
    """Modello ibrido: i primi n_head layer di calcolo sono W8A8, gli altri fp32.
    Il FakeQuant all'uscita del head rappresenta il tensore int8 che viaggia verso il fog."""
    hybrid = copy.deepcopy(qmodel).eval()
    fp_layers = [m for m in folded.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    hy_layers = [m for m in hybrid.model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for i, (hy, fp) in enumerate(zip(hy_layers, fp_layers)):
        if i >= n_head:                                   # tail: pesi e bias fp32 originali
            hy.weight.data = fp.weight.data.clone()
            hy.bias.data = fp.bias.data.clone()

    # quantizzatori: 0 = input, j = dopo la ReLU del layer j-1. Si tengono solo quelli con j <= n_head
    if n_head == 0:
        hybrid.input_q = nn.Identity()
    j = 1
    for seq in (hybrid.model.conv_backbone, hybrid.model.classifier):
        for idx, m in list(enumerate(seq)):
            if isinstance(m, FakeQuant):
                if j > n_head:
                    seq[idx] = nn.Identity()
                j += 1
    return hybrid


def split_metrics(ops, qparams, bias_q, n_head, n_layers, input_shape):
    """Metriche strutturali del taglio: device (int8) e fog (fp32)."""
    head = [o for o in ops if o["layer"] < n_head]
    tail = [o for o in ops if o["layer"] >= n_head]
    names = list(qparams.keys())[:n_head]
    input_elems = math.prod(input_shape)

    dev_macs, fog_macs = sum(o["macs"] for o in head), sum(o["macs"] for o in tail)
    dev_params, fog_params = sum(o["params"] for o in head), sum(o["params"] for o in tail)
    n_quantizers = min(n_head + 1, n_layers) if n_head > 0 else 0
    # device: pesi int8 + scale fp32 per canale + bias int32 + scala/zero-point delle attivazioni
    dev_bytes = sum(qparams[n][0].numel() + 4 * qparams[n][1].numel() + 4 * bias_q[n].numel()
                    for n in names) + 5 * n_quantizers
    fog_bytes = 4 * fog_params                              # fog: fp32

    if n_head == 0:
        payload, shape = input_elems, input_shape           # immagine grezza, 1 byte per valore (uint8)
    elif n_head == n_layers:
        payload, shape = 0, None                            # tutto sul device: nessuna trasmissione
    else:
        payload, shape = head[-1]["out_elems"], head[-1]["out_shape"]   # tensore int8 all'uscita del head

    peak_dev = max([o["in_elems"] + o["out_elems"] for o in head], default=input_elems)        # int8
    peak_fog = 4 * max([o["in_elems"] + o["out_elems"] for o in tail], default=0)              # fp32

    row = dict(n_head=n_head, dev_params=dev_params, fog_params=fog_params,
               dev_kb=dev_bytes / 1e3, fog_kb=fog_bytes / 1e3,
               dev_macs=dev_macs, fog_macs=fog_macs,
               payload_b=payload, payload_shape="×".join(map(str, shape)) if shape else "-",
               payload_vs_input=payload / input_elems,
               sram_dev_kb=peak_dev / 1e3, sram_fog_kb=peak_fog / 1e3,
               e_dev_arith_uj=estimate_energy_uj(dev_macs, "int8"),
               e_fog_arith_uj=estimate_energy_uj(fog_macs, "fp32"))
    row["fits"] = bool(row["dev_kb"] <= FLASH_KB and row["sram_dev_kb"] <= SRAM_KB)
    return row

# "sweep" sui tagli possibili
def run_split_study(folded, qmodel, qparams, bias_q, val_loader, test_loader=None,
                    device=torch.device("cpu"), input_size=(1, 3, 32, 32)):
    """Per ogni n_head: accuracy del modello ibrido + metriche strutturali del taglio."""
    ops = op_profile(folded, input_size)
    n_layers = max(o["layer"] for o in ops) + 1
    input_shape = tuple(input_size[1:])
    labels = cut_labels(ops)
    _, base_val = evaluate(folded, val_loader, device)

    rows = []
    for n_head in range(n_layers + 1):
        hybrid = build_split_model(folded, qmodel, n_head)
        _, val_acc = evaluate(hybrid, val_loader, device)
        test_acc = evaluate(hybrid, test_loader, device)[1] if test_loader is not None else float("nan")
        row = split_metrics(ops, qparams, bias_q, n_head, n_layers, input_shape)
        row.update(cut=labels[n_head], val_acc=val_acc, test_acc=test_acc,
                   val_drop=(base_val - val_acc) * 100)
        rows.append(row)
        print(f"n_head {n_head:2d} | {labels[n_head]:<30s} | val {val_acc * 100:.2f} | "
              f"payload {row['payload_b']:>6,} B | device {row['dev_kb']:>8.1f} KB | entra: {row['fits']}")
    return pd.DataFrame(rows), base_val



def split_tables(df):
    """Due tabelle: cosa passa al taglio e quanta rete sta su device/fog; accuracy ed energia aritmetica."""
    n_layers = len(df) - 1
    structure = pd.DataFrame({
        "taglio": df["cut"],
        "tensore trasmesso": df["payload_shape"],
        "payload (B)": df["payload_b"],
        "payload / immagine": df["payload_vs_input"].round(2),
        "layer su device (%)": (100 * df["n_head"] / n_layers).round(0),
        "params su device (%)": (100 * df["dev_params"] / (df["dev_params"] + df["fog_params"])).round(1),
        "MACs su device (%)": (100 * df["dev_macs"] / (df["dev_macs"] + df["fog_macs"])).round(1),
        "device (KB, int8)": df["dev_kb"].round(1),
        "fog (KB, fp32)": df["fog_kb"].round(1),
        "SRAM device (KB)": df["sram_dev_kb"].round(1),
        "entra nel device": df["fits"].map({True: "sì", False: "no"}),
    })
    quality = pd.DataFrame({
        "taglio": df["cut"],
        "val acc (%)": (df["val_acc"] * 100).round(2),
        "test acc (%)": (df["test_acc"] * 100).round(2),
        "E device aritm. (µJ)": df["e_dev_arith_uj"].round(2),
        "E fog aritm. (µJ)": df["e_fog_arith_uj"].round(2),
    })
    structure.index = quality.index = df["n_head"].values
    structure.index.name = quality.index.name = "n_head"
    return structure, quality


def plot_split(df, title):
    x = df["n_head"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    ax = ax.ravel()

    ax[0].bar(x, df["payload_b"].clip(lower=1))
    ax[0].axhline(df["payload_b"].iloc[0], color="red", linestyle="--", label="immagine grezza")
    ax[0].set_yscale("log"); ax[0].set_title("Byte del tensore trasmesso"); ax[0].legend()

    ax[1].plot(x, 100 * df["dev_params"] / (df["dev_params"] + df["fog_params"]), marker="o", label="parametri")
    ax[1].plot(x, 100 * df["dev_macs"] / (df["dev_macs"] + df["fog_macs"]), marker="o", label="MACs")
    ax[1].set_title("Quota della rete sul device (%)"); ax[1].legend()

    dev_mb, fog_mb = df["dev_kb"] / 1e3, df["fog_kb"] / 1e3
    ax[2].bar(x, dev_mb, label="device (int8)")
    ax[2].bar(x, fog_mb, bottom=dev_mb, label="fog (fp32)")
    ax[2].axhline(FLASH_KB / 1e3, color="red", linestyle="--", label="flash device")
    ax[2].set_title("Dimensione dei pesi (MB)"); ax[2].legend()

    ax[3].bar(x, df["sram_dev_kb"])
    ax[3].axhline(SRAM_KB, color="red", linestyle="--", label="SRAM device")
    ax[3].set_title("Picco SRAM sul device (KB)"); ax[3].legend()

    ax[4].plot(x, df["e_dev_arith_uj"].clip(lower=1e-3), marker="o", label="device (int8)")
    ax[4].plot(x, df["e_fog_arith_uj"].clip(lower=1e-3), marker="o", label="fog (fp32)")
    ax[4].set_yscale("log"); ax[4].set_title("Energia aritmetica stimata (µJ)"); ax[4].legend()

    ax[5].plot(x, df["val_acc"] * 100, marker="o", label="val")
    if df["test_acc"].notna().any():
        ax[5].plot(x, df["test_acc"] * 100, marker="o", label="test")
    ax[5].set_title("Accuracy (%)"); ax[5].legend()

    for a in ax:
        a.set_xticks(x); a.set_xlabel("n_head (layer sul device)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def min_payload_split(df, max_drop=1.0):
    """Tra i tagli INTERMEDI (almeno un layer per lato) che entrano nel device e perdono <= max_drop
    punti sulla val, quello che trasmette meno byte. None se non ce ne sono."""
    n = len(df) - 1
    ok = df[(df["n_head"] > 0) & (df["n_head"] < n) & df["fits"] & (df["val_drop"] <= max_drop)]
    return ok.sort_values("payload_b").iloc[0] if len(ok) else None


def save_split(name, df, results_dir="../results/split"):
    os.makedirs(results_dir, exist_ok=True)
    df.to_json(f"{results_dir}/split_{name}.json", orient="records", indent=2, force_ascii=False)