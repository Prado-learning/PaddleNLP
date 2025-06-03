import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SORSALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(in_features, out_features, bias_attr=bias, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.merged = False
        if self.alpha is None:
            self.scale = 1.0
        else:
            self.scale = self.alpha / self.rank
        if dropout > 0.0:
            self.sorsa_dropout = nn.Dropout(p=dropout)
        else:
            self.sorsa_dropout = lambda x: x
        if rank > 0:
            self.sorsa_A = self.create_parameter(
                shape=[rank, in_features],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.sorsa_S = self.create_parameter(
                shape=[rank],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.sorsa_B = self.create_parameter(
                shape=[out_features, rank],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.weight.stop_gradient = True

    def sorsa_init(self, weight_dtype: Optional[str] = None, adapter_dtype: Optional[str] = None):
        if not hasattr(self, "sorsa_A"):
            return
        # SVD初始化
        weight = self.weight
        if weight_dtype is not None:
            weight = paddle.cast(weight, weight_dtype)
        else:
            weight = paddle.cast(weight, "float32")
        u, s, vt = paddle.linalg.svd(weight, full_matrices=False)
        self.sorsa_A.set_value(u[:, : self.rank].transpose([1, 0]).astype(adapter_dtype or "float32"))
        self.sorsa_S.set_value(s[: self.rank].astype(adapter_dtype or "float32"))
        self.sorsa_B.set_value(vt[: self.rank, :].transpose([1, 0]).astype(adapter_dtype or "float32"))
        merge = paddle.matmul(self.sorsa_B * self.sorsa_S, self.sorsa_A)
        self.weight.set_value((weight - merge * self.scale).astype(self._dtype))

    def _merge(self, mode: bool):
        if not hasattr(self, "sorsa_A"):
            return
        if mode:
            if self.merge_weights and not self.merged:
                merge = paddle.matmul(self.sorsa_B * self.sorsa_S, self.sorsa_A)
                self.weight.set_value((self.weight + merge * self.scale).astype(self._dtype))
                self.merged = True
        else:
            if self.merge_weights and self.merged:
                merge = paddle.matmul(self.sorsa_B * self.sorsa_S, self.sorsa_A)
                self.weight.set_value((self.weight - merge * self.scale).astype(self._dtype))
                self.merged = False

    def forward(self, x: paddle.Tensor):
        if self.rank > 0 and not self.merged:
            result = F.linear(x, self.weight)
            result += (
                F.linear(
                    self.sorsa_dropout(x),
                    paddle.matmul(self.sorsa_B * self.sorsa_S, self.sorsa_A),
                )
                * self.scale
            )
            if self.bias is not None:
                result += self.bias
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

    @staticmethod
    def calc_ortho(model):
        ortho_loss = 0.0
        den = 0
        for name, param in model.named_parameters():
            if "sorsa_A" in name:
                a = param
                ia = paddle.eye(a.shape[0], dtype=a.dtype)
                a = paddle.matmul(a, a, transpose_y=True) - ia
                ortho_loss += paddle.norm(a, p="fro")
                den += 1
            elif "sorsa_B" in name:
                b = param
                ib = paddle.eye(b.shape[1], dtype=b.dtype)
                b = paddle.matmul(b, b, transpose_x=True) - ib
                ortho_loss += paddle.norm(b, p="fro")
                den += 1
        if den != 0:
            return ortho_loss / den
        else:
            return None
