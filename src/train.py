import copy
import time
import torch
import torch.nn as nn


def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}" + (f" ({torch.cuda.get_device_name()})" if dev.type == "cuda" else ""))
    return dev

#per inizializzare i pesi dei modelli (conv, batchnorm, linear) in modo appropriato
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)
    # ultimo layer (logits): pesi piccoli -> loss iniziale ~ ln(10)
    nn.init.normal_(model.classifier[-1].weight, 0, 0.01)


def train_one_epoch(model, loader, crit, opt, device):
    model.train()  # BatchNorm usa le statistiche del batch, Dropout attivo
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:         #il loader restituisce batch di dati (x) e label (y)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        logits = model(x)                 # 1. forward
        loss = crit(logits, y)            # 2. loss

        opt.zero_grad(set_to_none=True)   # 3. azzera i gradienti del batch precedente
        loss.backward()                   # 4. backprop: riempie per ogni parametro
        opt.step()                        # 5. update rule (SGD + momentum + weight decay):
                                          #    g = p.grad + wd * w
                                          #    v = momentum * v + g
                                          #    w = w - lr * v

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

# funzione di sola evaluation (senza update dei pesi) su un loader (val)
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()  # BatchNorm usa le running stats, Dropout spento
    crit = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += crit(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

# Funzione principale per addestrare il modello  
# (uso di SGD + cosine annealing, salvataggio del miglior checkpoint sul validation set)
def fit(model, train_loader, val_loader, device, epochs=60, lr=0.05,
        momentum=0.9, weight_decay=5e-4, ckpt_path=None):
    """Ricetta unica per tutti i modelli: SGD + cosine annealing, best ckpt sulla val."""
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}
    best_acc, t0 = 0.0, time.time()

    for e in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, crit, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()  # una volta per epoca
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc"],
                        [tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)

        if va_acc > best_acc:       #salvi il miglior modello (checkpoint) se la val acc migliora
            best_acc = va_acc
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)

        print(f"ep {e + 1:3d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc * 100:5.2f}"
              f" | val loss {va_loss:.3f} acc {va_acc * 100:5.2f} (best {best_acc * 100:5.2f})"
              f" | lr {history['lr'][-1]:.4f}")

    if ckpt_path:  # torna ai pesi migliori (per val), non agli ultimi
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history, time.time() - t0


def overfit_one_batch(model, loader, device, steps=100, lr=0.01):
    """Sanity check: su un solo batch la loss deve scendere. Lavora su una COPIA."""
    model = copy.deepcopy(model).to(device).train()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    for s in range(steps):
        loss = crit(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 20 == 0 or s == steps - 1:
            print(f"step {s:3d} | loss {loss.item():.4f}")