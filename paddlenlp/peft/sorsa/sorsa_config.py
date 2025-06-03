import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from ...utils.env import LORA_CONFIG_NAME
from ...utils.log import logger

@dataclass
class SORSAConfig:
    """
    SORSA配置类，存储SORSA相关超参数。
    Args:
        sorsa_rank (int): SORSA矩阵秩
        sorsa_alpha (Optional[float]): SORSA缩放因子
        sorsa_dropout (float): SORSA dropout
        sorsa_target_modules (List[str]): 需要替换为SORSA的模块名
        sorsa_base_model_name_or_path (Optional[str]): 基础模型名或路径
        sorsa_merge_weights (bool): 推理时是否merge权重
        sorsa_dtype (Optional[str]): 张量数据类型
    """
    sorsa_rank: int = field(default=4, metadata={"help": "SORSA矩阵秩"})
    sorsa_alpha: Optional[float] = field(default=None, metadata={"help": "SORSA缩放因子"})
    sorsa_dropout: float = field(default=0.0, metadata={"help": "SORSA dropout"})
    sorsa_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "需要替换为SORSA的模块名，如['query', 'key', 'value']"},
    )
    sorsa_base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "基础模型名或路径"}
    )
    sorsa_merge_weights: bool = field(default=True, metadata={"help": "推理时是否merge权重"})
    sorsa_dtype: Optional[str] = field(default=None, metadata={"help": "张量数据类型"})

    def __post_init__(self):
        pass

    @property
    def scaling(self):
        if self.sorsa_alpha is None:
            return 1.0
        return self.sorsa_alpha / self.sorsa_rank

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        output_dict = self.__dict__
        output_dict["scaling"] = self.scaling
        output_path = os.path.join(save_directory, LORA_CONFIG_NAME)
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find lora_config.json at '{pretrained_model_name_or_path}'")
        loaded_attributes = cls.from_json_file(config_file)
        loaded_attributes.pop("scaling", None)
        config = cls(**kwargs)
        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)
        return json_object
