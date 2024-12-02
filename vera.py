"""
Reference: code in https://github.com/danielgrittner/nanoGPT-LoRA/blob/master/model.py
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Overwriting the methods of nn.Linear:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class VeRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

        # LoRA stuff
        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)

            self.lora_scaling = lora_alpha / lora_rank
            self.lora_A = nn.Parameter(torch.empty((lora_rank, self.in_features), device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, lora_rank), device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.vera_b = nn.Parameter(torch.empty((self.out_features), device=device, dtype=dtype))
            self.vera_d = nn.Parameter(torch.empty((lora_rank), device=device, dtype=dtype))

            self.vera_b.requires_grad = False
            self.vera_d.requires_grad = False

            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Same as nn.Linear
            torch.nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5)) # Same as nn.Linear
            torch.nn.init.zeros_(self.vera_b)
            torch.nn.init.constant_(self.vera_d, 1) 

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = nn.Linear.forward(self, input)
        if not self.has_weights_merged:
          Ax = F.linear(
                self.lora_dropout(input),
                self.lora_A
              )  * self.vera_d
          BAx = F.linear(Ax, self.lora_B) * self.vera_b
          x += self.lora_scaling * BAx
        return x

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

    def train(self, mode: bool = True) -> "VeRALinear":
        nn.Linear.train(self, mode)
        if self.has_weights_merged:
            self.weight.data -= self.lora_scaling * (self.lora_B * self.vera_b) @ (self.lora_A * self.vera_d)
            self.has_weights_merged = False
        return self

    def eval(self) -> "VeRALinear":
        nn.Linear.eval(self)
        if not self.has_weights_merged:
            self.weight.data += self.lora_scaling * (self.lora_B * self.vera_b) @ (self.lora_A * self.vera_d)
            self.has_weights_merged = True
        return self
