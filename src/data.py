import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def download_cifar10(root="./data"):
    # mirror HF al posto dell'URL originale
    torchvision.datasets.CIFAR10.url = (
        "https://huggingface.co/datasets/MIT-OL-AI-D/"
        "cifar-10-python/resolve/main/cifar-10-python.tar.gz"
    )
    # train=True scarica comunque l'archivio intero (train + test)
    return torchvision.datasets.CIFAR10(root=root, train=True, download=True)


def get_dataloaders(root="./data", batch_size=128, val_size=5000,
                    num_workers=2, seed=0):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # stesso dataset due volte: con augmentation (train) e senza (val)
    train_aug = torchvision.datasets.CIFAR10(root, train=True, transform=train_tf)
    train_eval = torchvision.datasets.CIFAR10(root, train=True, transform=eval_tf)
    test_set = torchvision.datasets.CIFAR10(root, train=False, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(Subset(train_eval, val_idx), batch_size=256, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def show_samples(root="./data", n=16, seed=0, cols=8):
    ds = torchvision.datasets.CIFAR10(root=root, train=True)  # PIL, nessun transform
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idxs):
        img, label = ds[i]
        ax.imshow(img)
        ax.set_title(ds.classes[label], fontsize=8)
    plt.tight_layout()
    plt.show()