import torch.nn as nn

# VGG11: 8 conv, 4 max_pool (32 -> 2), l'ultimo pool chiude il backbone
VGG11_CONFIG = [64, "max_pool", 128, "max_pool", 256, 256, "max_pool",
                512, 512, 512, 512, "max_pool"]

TEACHER_HIDDEN = (2048, 1024)
STUDENT_WIDTH = 1 / 8               # Student --> circa 1/64 dei parametri del Teacher      [stessa struttura]
                                        # (diminuisce con il quadrato per via dei kernel quadrati)


def scale_config(config, width_mult):
    """Scala i canali dei conv; i 'max_pool' restano invariati."""
    return [c if c == "max_pool" else max(1, round(c * width_mult)) for c in config]


class VGGNet(nn.Module):
    def __init__(self, config, hidden_dims=TEACHER_HIDDEN, in_channels=3,
                 num_classes=10, input_size=32, batch_norm=True, dropout=0.5):
        super().__init__()

        #creo la backbone conv+pool a partire dalla config (lista di int e 'max_pool')
        in_ch, n_pools, layers = in_channels, 0, []
        for layer in config:
            if layer == "max_pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                n_pools += 1
            else:
                layers.append(nn.Conv2d(in_ch, layer, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(layer))
                layers.append(nn.ReLU(inplace=True))
                in_ch = layer
        self.conv_backbone = nn.Sequential(*layers)

        final_hw = input_size // (2 ** n_pools)
        assert final_hw >= 1, "troppi max_pool per questa input_size"
        self.feature_dim = in_ch * final_hw * final_hw

        mlp, prev = [], self.feature_dim
        for h in hidden_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p=dropout)]
            prev = h
        mlp.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*mlp)

    #  conv + mlp 
    def forward(self, x):
        x = self.conv_backbone(x)
        return self.classifier(x.flatten(1))


MODEL_SPECS = {
    "teacher": dict(config=VGG11_CONFIG, hidden_dims=TEACHER_HIDDEN, dropout=0.5),
    # student = teacher con larghezza /8 (canali conv e hidden dell'MLP)
    "student": dict(
        config=scale_config(VGG11_CONFIG, STUDENT_WIDTH),
        hidden_dims=tuple(round(h * STUDENT_WIDTH) for h in TEACHER_HIDDEN),
        dropout=0.3,  # modello piccolo -> meno regolarizzazione
    ),
}


def build_model(name, num_classes=10, batch_norm=True):
    return VGGNet(num_classes=num_classes, batch_norm=batch_norm, **MODEL_SPECS[name])