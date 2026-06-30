import time
from typing import Any, Annotated

from pydantic import Field

from vllm.config import ModelConfig
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.pooling_params import PoolingParams
from vllm.renderers import TokenizeParams
from vllm.utils import random_uuid


class RouterClassifyRequest(OpenAIBaseModel):
    model: str
    prompts: list[str] | None = None
    messages: list[dict[str, Any]] | None = None

    truncate_prompt_tokens: Annotated[int | None, Field(ge=-1)] = None
    priority: int = 0
    use_activation: bool = False

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        encoder_config = model_config.encoder_config or {}
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            add_special_tokens=True,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            truncation_side="right",
            do_lower_case=encoder_config.get("do_lower_case", False),
            max_total_tokens_param="max_model_len",
        )

    def to_pooling_params(self):
        return PoolingParams(
            task="embed",
            use_activation=self.use_activation,
        )


class RouterClassifyResponseData(OpenAIBaseModel):
    index: int
    object: str = "router_classification"
    cls_logits: list[float]


class RouterClassifyResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cls-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[RouterClassifyResponseData]
    usage: UsageInfo