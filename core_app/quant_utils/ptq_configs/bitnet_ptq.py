import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LlamaForCausalLM, PreTrainedModel

# ----- Quantization Functions -----
def _fake_vq(tensor, num_bits=4):
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max()
    scale = 2 ** torch.floor(torch.log2(max_val / qmax + 1e-8))
    tensor_clipped = torch.clamp(tensor, -qmax * scale, qmax * scale)
    quantized = torch.round(tensor_clipped / scale) * scale
    return quantized

def _ternary_vq(tensor, eps=1e-8):
    gamma = tensor.abs().mean()
    scaled = tensor / (gamma + eps)
    quantized = torch.round(scaled).clamp(-1, 1)
    return quantized * gamma

def _fake_aq(tensor, num_bits=8, scale=None):
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max() if scale is None else scale * qmax
    scale = 2 ** torch.floor(torch.log2(max_val / qmax + 1e-8))
    tensor_clipped = torch.clamp(tensor, -qmax * scale, qmax * scale)
    quantized = torch.round(tensor_clipped / scale) * scale
    return quantized, scale

def _calibrate_activation(self, input, percentile=0.999):
    qmax = 2 ** (8 - 1) - 1  # 8-bit
    flat = input.detach().abs().flatten()
    if flat.numel() == 0:
        return
    k = int(flat.numel() * percentile)
    max_val = torch.topk(flat, k, sorted=True).values[-1]
    scale = max(max_val.item(), 1e-5) / qmax
    self.register_buffer("act_scale", torch.tensor(scale, device=input.device))


class BitNetLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=4, use_ternary=True, act_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.use_ternary = use_ternary
        self.act_bits = act_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()
        self.quantize_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize_weights(self):
        with torch.no_grad():
            if self.use_ternary:
                self.weight.data = _ternary_vq(self.weight.data)
                if self.bias is not None:
                    self.bias.data = _ternary_vq(self.bias.data)
            else:
                self.weight.data = _fake_vq(self.weight.data, self.num_bits)
                if self.bias is not None:
                    self.bias.data = _fake_vq(self.bias.data, self.num_bits)

    def quantize_activation(self, input, scale=None):
        qmax = 2 ** (self.act_bits - 1) - 1
        if scale is None:
            max_val = input.detach().abs().max()
            scale = 2 ** torch.floor(torch.log2(max_val / qmax + 1e-8))
        input = torch.clamp(input, -qmax * scale, qmax * scale)
        return torch.round(input / scale) * scale

    def forward(self, input):
        input_q = self.quantize_activation(input)
        return F.linear(input_q, self.weight, self.bias)


def apply_bitnet_ptq(model, num_bits=4, use_ternary=True, act_bits=8):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quant_layer = BitNetLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                num_bits=num_bits,
                use_ternary=use_ternary,
                act_bits=act_bits
            )
            with torch.no_grad():
                quant_layer.weight.copy_(module.weight.data)
                if module.bias is not None:
                    quant_layer.bias.copy_(module.bias.data)
                quant_layer.quantize_weights()
            setattr(model, name, quant_layer)
        else:
            apply_bitnet_ptq(module, num_bits=num_bits, use_ternary=use_ternary, act_bits=act_bits)
    return model


