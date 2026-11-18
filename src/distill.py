import time
import torch
import torch.nn.functional as F
from src.train import evaluate

# Formulazione della loss di distillazione (Hinton et al. 2015)
def distillation_loss(s_logits, t_logits, y, T=4.0, alpha=0.1):
    """alpha * CE(hard label) + (1 - alpha) * T^2 * KL(teacher_T || student_T)."""
    hard = F.cross_entropy(s_logits, y)
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),     # log-prob dello student (a temperatura T)
                    F.softmax(t_logits / T, dim=1),         # prob del teacher (a temperatura T)
                    reduction="batchmean") * (T * T)
    return alpha * hard + (1 - alpha) * soft

# Nuovo flusso di train di un epoca con distillazione: il teacher è fisso, lo student impara da hard + soft label
def train_one_epoch_kd(student, teacher, loader, opt, device, T, alpha):
    student.train()      # BN e Dropout dello student in modalità training
    teacher.eval()       # il teacher è fisso: BN con running stats, niente dropout
    total_loss, correct, t_correct, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.no_grad():
            t_logits = teacher(x)                              # 1. il teacher guarda lo stesso batch
        s_logits = student(x)                                  # 2. forward dello student
        loss = distillation_loss(s_logits, t_logits, y, T, alpha)   # 3. loss mista hard + soft

        opt.zero_grad(set_to_none=True)                        # 4. come nel training normale
        loss.backward()                                        # 5. backprop solo sullo student
        opt.step()                                             # 6. update rule (SGD + momentum + wd)

        total_loss += loss.item() * x.size(0)
        correct += (s_logits.argmax(1) == y).sum().item()
        t_correct += (t_logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n, t_correct / n

# Training di distllation: SGD nesterov + cosine annealing, checkpoint del modello migliore sulla val   [come per fit()]
def fit_kd(student, teacher, train_loader, val_loader, device, epochs=60, lr=0.05, momentum=0.9,
           weight_decay=5e-4, T=4.0, alpha=0.1, ckpt_path=None):
    """Stessa ricetta di fit() (SGD nesterov + cosine, best ckpt sulla val) con la loss di distillation.
    Nota: train_loss è la loss KD (non confrontabile con la CE); val_loss è la CE sulle hard label."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "teacher_train_acc", "val_loss", "val_acc", "lr"]}
    best_acc, t0 = 0.0, time.time()

    for e in range(epochs):
        tr_loss, tr_acc, t_acc = train_one_epoch_kd(student, teacher, train_loader, opt, device, T, alpha)
        va_loss, va_acc = evaluate(student, val_loader, device)

        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()
        for k, v in zip(["train_loss", "train_acc", "teacher_train_acc", "val_loss", "val_acc"],
                        [tr_loss, tr_acc, t_acc, va_loss, va_acc]):
            history[k].append(v)

        if va_acc > best_acc:
            best_acc = va_acc
            if ckpt_path:
                torch.save(student.state_dict(), ckpt_path)

        print(f"ep {e + 1:3d}/{epochs} | KD loss {tr_loss:.3f} | train acc {tr_acc * 100:5.2f} "
              f"(teacher {t_acc * 100:5.2f}) | val acc {va_acc * 100:5.2f} (best {best_acc * 100:5.2f}) "
              f"| lr {history['lr'][-1]:.4f}")

    if ckpt_path:  # torna ai pesi migliori (per val)
        student.load_state_dict(torch.load(ckpt_path, map_location=device))
    return history, time.time() - t0