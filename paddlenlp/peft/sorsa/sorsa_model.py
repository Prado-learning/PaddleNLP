import paddle
import paddle.nn as nn
from typing import Dict, Union, Optional
from .sorsa_layers import SORSALinear
from .sorsa_config import SORSAConfig

class SORSAModel(nn.Layer):
    """
    SORSAModel: 用于将SORSA算法应用到Paddle模型的包装器。
    """
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        SORSALinear: nn.Linear,
    }

    def __init__(self, model: nn.Layer, sorsa_config: SORSAConfig) -> None:
        super().__init__()
        self.sorsa_config = sorsa_config
        self.model = self.get_sorsa_model(model, sorsa_config)
        self.forward = self.model.forward
        self.mark_only_sorsa_as_trainable()

    def get_sorsa_model(self, model: Union[nn.Layer], sorsa_config: SORSAConfig):
        # 自动替换目标模块为SORSALinear
        for name, sublayer in model.named_sublayers():
            if self._is_target(name, sorsa_config.target_modules) and isinstance(sublayer, nn.Linear):
                sorsa_linear = SORSALinear(
                    in_features=sublayer.weight.shape[1],
                    out_features=sublayer.weight.shape[0],
                    rank=sorsa_config.rank,
                    alpha=sorsa_config.alpha,
                    dropout=sorsa_config.dropout,
                    merge_weights=sorsa_config.merge_weights,
                    bias=(sublayer.bias is not None),
                )
                # 拷贝原始权重和bias
                sorsa_linear.weight.set_value(sublayer.weight)
                if sublayer.bias is not None:
                    sorsa_linear.bias.set_value(sublayer.bias)
                # 替换
                self._set_submodule(model, name, sorsa_linear)
        return model

    def _is_target(self, name: str, target_modules):
        if target_modules is None:
            return False
        if isinstance(target_modules, str):
            return target_modules in name
        if isinstance(target_modules, (list, tuple)):
            return any(t in name for t in target_modules)
        return False

    def _set_submodule(self, model, target: str, module: nn.Layer):
        atoms = target.split('.')
        name = atoms.pop(-1)
        mod = model
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)

    def mark_only_sorsa_as_trainable(self):
        for name, param in self.model.named_parameters():
            if 'sorsa_' in name:
                param.stop_gradient = False
            else:
                param.stop_gradient = True

    def merge(self, mode=True):
        for sublayer in self.model.sublayers():
            if isinstance(sublayer, SORSALinear):
                sublayer._merge(mode)

    def get_trainable_state_dict(self):
        return {k: v for k, v in self.model.state_dict().items() if 'sorsa_' in k}

    def print_trainable_parameters(self):
        trainable = [k for k, v in self.model.named_parameters() if not v.stop_gradient]
        print(f"Trainable parameters: {trainable}")
