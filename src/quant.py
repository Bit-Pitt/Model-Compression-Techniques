import copy, torch, torch.nn as nn

'''
    Quantizzazione W8A8 PTQ (Post Training Quantization) simulata in pyTorch:
    - pesi per canale di output  [per channel quantization], simmetrica, una scala per canale
'''



def fold_bn(model):
    """Copia del modello con ogni BatchNorm fuso nel Conv che lo precede."""
    model = copy.deepcopy(model).eval()
    layers = list(model.conv_backbone)
    new_layers, i = [], 0
    while i < len(layers):
        layer = layers[i]
        if (isinstance(layer, nn.Conv2d) and i + 1 < len(layers)
                and isinstance(layers[i + 1], nn.BatchNorm2d)):
            bn = layers[i + 1]
            factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # gamma / sqrt(var + eps)
            with torch.no_grad():
                layer.bias.copy_((layer.bias - bn.running_mean) * factor + bn.bias)
                layer.weight.mul_(factor.view(-1, 1, 1, 1))           # un fattore per filtro
            new_layers.append(layer)
            i += 2
        else:
            new_layers.append(layer)
            i += 1
    model.conv_backbone = nn.Sequential(*new_layers)
    return model

# Quantizazione dei pesi per canale di output, simmetrica, una scala per canale. Ritorna (q int8, scale [C_out]).
def _per_channel(scale, ref):
    """Reshape di [C_out] in [C_out, 1, 1, ...] per fare broadcast sul tensore dei pesi."""
    return scale.view(-1, *[1] * (ref.dim() - 1))


def quantize_weights_per_channel(w, bits=8):
    """Simmetrica, una scala per canale di output. w: [C_out, ...] (Conv o Linear).
    Ritorna (q int8, scale [C_out])."""
    qmax = 2 ** (bits - 1) - 1                      # 127 per 8 bit
    max_abs = w.reshape(w.shape[0], -1).abs().amax(dim=1)   # un massimo per filtro
    scale = (max_abs / qmax).clamp(min=1e-12)       # evita scala 0 se un filtro è tutto a zero
    q = torch.round(w / _per_channel(scale, w)).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_weights(q, scale):
    return q.float() * _per_channel(scale, q)


def quantize_model_weights(model, bits=8):
    """Modello GIÀ con BN fuso. Ritorna:
    - sim_model: copia con i pesi sostituiti da q * scale (fp32 sulla griglia int8)
    - qparams: {nome_layer: (q_int8, scale)}  -> serve per export e calcolo dello spazio"""
    sim_model = copy.deepcopy(model).eval()
    qparams = {}
    for name, m in sim_model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            q, scale = quantize_weights_per_channel(m.weight.data, bits)
            qparams[name] = (q, scale)
            m.weight.data = dequantize_weights(q, scale)
    return sim_model, qparams




class FakeQuant(nn.Module):
    """Quantizzatore di attivazioni: int8 asimmetrico, per-tensor, range da calibrazione min/max."""
    QMIN, QMAX = -128, 127

    def __init__(self):
        super().__init__()
        self.calibrating = True
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))

    def forward(self, x):
        if self.calibrating:                       # fase 1: osserva e basta
            self.min_val = torch.minimum(self.min_val, x.min())
            self.max_val = torch.maximum(self.max_val, x.max())
            return x
        q = torch.clamp(torch.round(x / self.scale) + self.zero_point, self.QMIN, self.QMAX)
        return (q - self.zero_point) * self.scale  # fase 2: quantizza e dequantizza

    def freeze(self):
        lo = min(self.min_val.item(), 0.0)         # lo zero deve essere rappresentabile esattamente
        hi = max(self.max_val.item(), 0.0)
        scale = max((hi - lo) / (self.QMAX - self.QMIN), 1e-8)
        self.scale = torch.tensor(scale)
        self.zero_point = torch.tensor(float(round(self.QMIN - lo / scale)))
        self.calibrating = False

'''
Cosa aggiungiamo

Il modello W8A8 PTQ simulato da pyTorch:

- FakeQuant all'ingresso e dopo ogni ReLU: arrotonda le attivazioni su una griglia int8 asimmetrica (con zero-point),
                                             poi le riporta in float.
- Calibrazione: {serve per quantizzare la Attivazioni}fa passare qualche centinaio di immagini di train, registra min/max di ogni punto e
                                            ne ricava scala e zero-point. 
- Bias int32: in un conv int8 reale il bias si somma all'accumulatore, quindi è un intero con scala s_in · s_w[canale]. Qui lo simuliamo.

'''
class QuantModel(nn.Module):
    """Quantizza l'input e delega al modello (che ha FakeQuant dopo ogni ReLU)."""
    def __init__(self, model):
        super().__init__()
        self.input_q = FakeQuant()
        self.model = model

    def forward(self, x):
        return self.model(self.input_q(x))

    def quantizers(self):  # ordine: input, dopo ReLU 1, 2, ...
        return [m for m in self.modules() if isinstance(m, FakeQuant)]


def add_act_quantizers(model):
    """Copia del modello con un FakeQuant dopo ogni ReLU."""
    model = copy.deepcopy(model).eval()

    def with_fq(seq):
        out = []
        for layer in seq:
            out.append(layer)
            if isinstance(layer, nn.ReLU):
                out.append(FakeQuant())
        return nn.Sequential(*out)

    model.conv_backbone = with_fq(model.conv_backbone)
    model.classifier = with_fq(model.classifier)
    return model


@torch.no_grad()
def calibrate(qmodel, loader, n_batches=4):
    qmodel.eval()
    for fq in qmodel.quantizers():
        fq.calibrating = True
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        qmodel(x)
    for fq in qmodel.quantizers():
        fq.freeze()


def quantize_biases(qmodel, qparams):
    """Bias int32 con scala s_in * s_w[canale]; nel modello simulato il bias diventa il suo valore
    dequantizzato. Ritorna {nome_layer: bias_int32} per l'export."""
    fqs = qmodel.quantizers()
    layers = [(n, m) for n, m in qmodel.model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(fqs) == len(layers), "un quantizzatore di input per ogni layer di calcolo"
    bias_q = {}
    for k, (name, m) in enumerate(layers):
        s_b = fqs[k].scale * qparams[name][1]            # s_in * s_w, un valore per canale
        b_int = torch.round(m.bias.data / s_b).to(torch.int32)
        m.bias.data = b_int.float() * s_b
        bias_q[name] = b_int
    return bias_q


def build_w8a8_model(folded, calib_loader, n_calib_batches=4):
    """BN già fuso -> modello W8A8 simulato. Ritorna (qmodel, qparams, bias_q)."""
    with_fq = add_act_quantizers(folded)
    sim, qparams = quantize_model_weights(with_fq)       # pesi per-channel int8
    qmodel = QuantModel(sim)
    calibrate(qmodel, calib_loader, n_calib_batches)     # scale delle attivazioni
    bias_q = quantize_biases(qmodel, qparams)
    return qmodel, qparams, bias_q


def int8_model_size_mb(qparams, bias_q, n_act_quantizers):
    """Spazio analitico: pesi int8 + scale fp32 per canale + bias int32 + scale/zp attivazioni."""
    w = sum(q.numel() for q, _ in qparams.values())
    s = sum(scale.numel() * 4 for _, scale in qparams.values())
    b = sum(t.numel() * 4 for t in bias_q.values())
    a = n_act_quantizers * 5
    return (w + s + b + a) / 1e6