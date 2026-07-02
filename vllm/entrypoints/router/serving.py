import asyncio
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.router.protocol import (
    RouterClassifyRequest,
    RouterClassifyResponse,
    RouterClassifyResponseData,
)
from vllm.inputs import TokensPrompt, tokens_input
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.utils.async_utils import make_async, merge_async_iterators

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

def _find_safe_truncate_pos(token_ids: list[int], max_prompt_tokens: int | None, content_token_ids: Any) -> int | None:
    ids = {int(x) for x in (content_token_ids or []) if x is not None and int(x) >= 0}
    if max_prompt_tokens is None or max_prompt_tokens < 0 or len(token_ids) <= max_prompt_tokens or not ids:
        return max_prompt_tokens
    if token_ids[max_prompt_tokens - 1] not in ids and token_ids[max_prompt_tokens] not in ids:
        return max_prompt_tokens
    for pos in range(max_prompt_tokens, 0, -1):
        if token_ids[pos - 1] not in ids and (pos >= len(token_ids) or token_ids[pos] not in ids):
            return pos
    return max_prompt_tokens

def _truncate_prompt_ids(token_ids: list[int], *, max_prompt_tokens: int | None, cls_id: int | None, expects_cls: bool) -> list[int]:
    prompt_ids = [int(x) for x in token_ids]
    if max_prompt_tokens is None or max_prompt_tokens < 0 or not prompt_ids:
        return prompt_ids
    was_truncated = len(prompt_ids) >= max_prompt_tokens
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[:max_prompt_tokens]
    if expects_cls and was_truncated and cls_id is not None and cls_id >= 0 and cls_id not in prompt_ids:
        prompt_ids[-1] = int(cls_id)
    return prompt_ids

def _get_spatial_merge_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    return int(getattr(image_processor, "merge_size", None) or getattr(image_processor, "spatial_merge_size", None) or 2)

def _count_content_tokens(prompt_ids: list[int], content_token_id: int | None) -> int:
    if content_token_id is None:
        return 0
    return sum(1 for tid in prompt_ids if tid == content_token_id)

def _get_tokens_per_media(media_grid_thw: Any, processor: Any) -> list[int]:
    media_grid = _tolist_if_possible(media_grid_thw) or []
    if not media_grid:
        return []
    merge_size = _get_spatial_merge_size(processor)
    return [(int(t) * int(h) * int(w)) // (merge_size * merge_size) for t, h, w in media_grid]

def _limit_truncate_pos_by_content_units(token_ids: list[int], truncate_pos: int | None, content_token_id: int | None, kept_units: int) -> int | None:
    if truncate_pos is None or truncate_pos < 0 or content_token_id is None or kept_units < 0:
        return truncate_pos
    seen = 0
    for idx, tid in enumerate(token_ids[:truncate_pos]):
        if tid == content_token_id:
            seen += 1
            if seen > kept_units:
                return idx
    return truncate_pos

def _trim_media_inputs_by_count(media_inputs: list[Any], media_grid_thw: Any, kept_count: int) -> tuple[list[Any], Any]:
    media_grid = _tolist_if_possible(media_grid_thw) or []
    if kept_count <= 0:
        return [], []
    if kept_count > len(media_inputs) or kept_count > len(media_grid):
        raise ValueError(f"Media count exceeds available inputs: kept_count={kept_count}, media_inputs={len(media_inputs)}, media_grid={len(media_grid)}")
    return media_inputs[:kept_count], media_grid[:kept_count]

def _validate_media_unit_alignment(media_inputs: list[Any], media_grid_thw: Any, kept_count: int, media_name: str) -> None:
    media_grid = _tolist_if_possible(media_grid_thw) or []
    if kept_count != len(media_inputs) or kept_count != len(media_grid):
        raise ValueError(f"{media_name} unit mismatch after trimming: kept={kept_count}, media_inputs={len(media_inputs)}, media_grid={len(media_grid)}")

def _validate_prompt_len(prompt_ids: list[int], validation_limit: int | None, source_name: str) -> str | None:
    if validation_limit is not None and validation_limit >= 0 and len(prompt_ids) > validation_limit:
        return f"{source_name} too long after preprocessing: tokens={len(prompt_ids)}, limit={validation_limit}"
    return None

def _normalize_media_value(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("url"), str):
            return value["url"]
        if isinstance(value.get("image_url"), str):
            return value["image_url"]
        if isinstance(value.get("image"), str):
            return value["image"]
    return value

def _normalize_media_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        new_message = {"role": message.get("role"), "content": []}
        for item in message.get("content", []):
            new_item = dict(item)
            if new_item.get("type") == "image":
                if "image" not in new_item and "url" in new_item:
                    new_item["image"] = new_item.pop("url")
                new_item["image"] = _normalize_media_value(new_item.get("image"))
            elif new_item.get("type") == "image_url":
                if "image_url" not in new_item and "url" in new_item:
                    new_item["image_url"] = new_item.pop("url")
                new_item["image_url"] = _normalize_media_value(new_item.get("image_url"))
            new_message["content"].append(new_item)
        normalized.append(new_message)
    return normalized

def _restore_raw_media_fields(messages: list[dict[str, Any]], raw_messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not raw_messages:
        return messages
    restored = []
    for msg_idx, message in enumerate(messages):
        raw_message = raw_messages[msg_idx] if msg_idx < len(raw_messages) else {}
        raw_content = raw_message.get("content", []) if isinstance(raw_message, dict) else []
        new_message = {"role": message.get("role"), "content": []}
        for item_idx, item in enumerate(message.get("content", [])):
            new_item = dict(item)
            raw_item = raw_content[item_idx] if item_idx < len(raw_content) else None
            if isinstance(raw_item, dict) and raw_item.get("type") in {"image", "image_url", "video"}:
                for key in ("type", "image_url", "url", "min_pixels", "max_pixels", "fps", "nframes"):
                    if key in raw_item:
                        new_item[key] = raw_item[key]
                if raw_item.get("type") == "image_url":
                    new_item.pop("image", None)
            new_message["content"].append(new_item)
        restored.append(new_message)
    return restored

def _tolist_if_possible(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return None
    return value

class ServingRouterClassification(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None = None,
        chat_template_content_format: Any = "auto",
        default_chat_template_kwargs: dict[str, Any] | None = None,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.trust_request_chat_template = trust_request_chat_template
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)
        self.processor = None
        if AutoProcessor is not None:
            try:
                model_path = models.base_model_paths[0].model_path if getattr(models, "base_model_paths", None) else None
                if model_path:
                    self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                self.processor = None

    async def classify(
        self,
        request: RouterClassifyRequest,
        raw_request: Request | None = None,
    ) -> RouterClassifyResponse | ErrorResponse:
        error = await self._check_model(request)
        if error is not None:
            return error

        tokenizer = self.renderer.get_tokenizer()
        model_config = self.model_config
        processor = self.processor or getattr(model_config, "processor", None)
        processing_class = processor or tokenizer
        cls_id = None
        try:
            cls_id = tokenizer.convert_tokens_to_ids("<|CLS|>")
            if cls_id is not None:
                cls_id = int(cls_id)
        except Exception:
            cls_id = None
        try:
            image_content_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if image_content_token_id is not None:
                image_content_token_id = int(image_content_token_id)
        except Exception:
            image_content_token_id = None
        try:
            video_content_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            if video_content_token_id is not None:
                video_content_token_id = int(video_content_token_id)
        except Exception:
            video_content_token_id = None
        requested_truncation_limit = request.truncate_prompt_tokens
        validation_limit = getattr(model_config, "max_model_len", None)
        effective_truncation_limit = requested_truncation_limit if requested_truncation_limit is not None else validation_limit

        engine_prompts = []
        raw_json: dict[str, Any] = {}
        if raw_request is not None:
            try:
                raw_body = await raw_request.body()
                raw_json = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                raw_json = {}
        request_messages = request.messages
        source_messages = raw_json.get("messages") or request_messages
        if source_messages is not None and request_messages is not None and raw_json.get("messages"):
            request_messages = _restore_raw_media_fields(request_messages, raw_json.get("messages"))
        else:
            request_messages = source_messages
        if request_messages is not None:
            chat_template_messages = deepcopy(request_messages)
            raw_prompt = processing_class.apply_chat_template(
                chat_template_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            has_media = any(
                item.get("type") in {"image", "image_url", "video"}
                for message in request_messages
                for item in message.get("content", [])
            )
            if has_media:
                if process_vision_info is None:
                    return self.create_error_response("qwen_vl_utils.process_vision_info is required for multimodal messages classify")
                if processor is None:
                    return self.create_error_response("Multimodal classify requires a processor, but only tokenizer is available")
                materialized_messages = _normalize_media_messages(request_messages)
                materialized_messages_for_template = deepcopy(materialized_messages)
                raw_prompt = processor.apply_chat_template(
                    materialized_messages_for_template,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                image_inputs, video_inputs = process_vision_info(materialized_messages)
                processed_inputs = processor(
                    text=[raw_prompt],
                    images=image_inputs or None,
                    videos=video_inputs or None,
                    padding=False,
                    return_tensors="pt",
                )
                encode_async = make_async(tokenizer.encode, executor=self._tokenizer_executor)
                tokenization_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
                tokenized_prompts = await asyncio.gather(encode_async(raw_prompt, **tokenization_kwargs))
                raw_prompt_ids = [int(x) for x in tokenized_prompts[0]]
                image_grid_thw = processed_inputs.get("image_grid_thw")
                image_token_costs = _get_tokens_per_media(image_grid_thw, processor)
                image_placeholder_count = _count_content_tokens(raw_prompt_ids, image_content_token_id)
                non_image_token_count = len(raw_prompt_ids) - image_placeholder_count
                estimated_total_tokens = non_image_token_count + sum(int(x) for x in image_token_costs)
                if validation_limit is not None and validation_limit >= 0 and estimated_total_tokens > validation_limit:
                    return self.create_error_response(
                        f"multimodal input too long before inference: estimated_tokens={estimated_total_tokens}, limit={validation_limit}"
                    )
                kept_images = len(image_token_costs)
                if effective_truncation_limit is not None and effective_truncation_limit >= 0:
                    available_image_budget = max(0, effective_truncation_limit - non_image_token_count)
                    kept_images = 0
                    used_image_budget = 0
                    for token_num in image_token_costs:
                        if used_image_budget + int(token_num) > available_image_budget:
                            break
                        kept_images += 1
                        used_image_budget += int(token_num)
                safe_truncate_tokens = _find_safe_truncate_pos(
                    raw_prompt_ids,
                    effective_truncation_limit,
                    (image_content_token_id, video_content_token_id),
                )
                safe_truncate_tokens = _limit_truncate_pos_by_content_units(
                    raw_prompt_ids,
                    safe_truncate_tokens,
                    image_content_token_id,
                    kept_images,
                )
                
                prompt_ids = _truncate_prompt_ids(
                    raw_prompt_ids,
                    max_prompt_tokens=safe_truncate_tokens,
                    cls_id=cls_id,
                    expects_cls="<|CLS|>" in raw_prompt,
                )
                
                image_inputs, image_grid_thw = _trim_media_inputs_by_count(
                    image_inputs or [], image_grid_thw, kept_images
                )
                _validate_media_unit_alignment(image_inputs, image_grid_thw, kept_images, "image")
                err = _validate_prompt_len(prompt_ids, validation_limit, "multimodal messages")
                if err is not None:
                    return self.create_error_response(err)
                multi_modal_data: dict[str, Any] | None = None
                if image_inputs or video_inputs:
                    multi_modal_data = {}
                    if image_inputs:
                        multi_modal_data["image"] = image_inputs
                    if video_inputs:
                        multi_modal_data["video"] = video_inputs
                engine_prompts = [TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)]
            else:
                encode_async = make_async(tokenizer.encode, executor=self._tokenizer_executor)
                tokenization_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
                tokenized_prompts = await asyncio.gather(encode_async(raw_prompt, **tokenization_kwargs))
                raw_prompt_ids = [int(x) for x in tokenized_prompts[0]]
                if validation_limit is not None and validation_limit >= 0 and len(raw_prompt_ids) > validation_limit:
                    return self.create_error_response(
                        f"text messages too long before inference: tokens={len(raw_prompt_ids)}, limit={validation_limit}"
                    )
                prompt_ids = _truncate_prompt_ids(
                    raw_prompt_ids,
                    max_prompt_tokens=effective_truncation_limit,
                    cls_id=cls_id,
                    expects_cls="<|CLS|>" in raw_prompt,
                )
                err = _validate_prompt_len(prompt_ids, validation_limit, "text messages")
                if err is not None:
                    return self.create_error_response(err)
                engine_prompts = [tokens_input(prompt_ids, prompt=raw_prompt)]
        else:
            if not request.prompts:
                return self.create_error_response("Either prompts or messages must be provided")
            encode_async = make_async(tokenizer.encode, executor=self._tokenizer_executor)
            tokenization_kwargs = request.build_tok_params(model_config).get_encode_kwargs()
            tokenized_prompts = await asyncio.gather(*(encode_async(p, **tokenization_kwargs) for p in request.prompts))
            for prompt, token_ids in zip(request.prompts, tokenized_prompts):
                raw_prompt_ids = [int(x) for x in token_ids]
                prompt_ids = _truncate_prompt_ids(
                    raw_prompt_ids,
                    max_prompt_tokens=effective_truncation_limit,
                    cls_id=cls_id,
                    expects_cls="<|CLS|>" in prompt,
                )
                err = _validate_prompt_len(prompt_ids, validation_limit, "prompts")
                if err is not None:
                    return self.create_error_response(err)
                engine_prompts.append(tokens_input(prompt_ids, prompt=prompt))

        trace_headers: Mapping[str, str] | None = None
        if raw_request is not None:
            trace_headers = await self._get_trace_headers(raw_request.headers)

        pooling_params = request.to_pooling_params()

        generators = []
        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"cls-{i}"
            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=pooling_params,
                lora_request=None,
            )
            generators.append(
                self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=None,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
            )

        result_generator = merge_async_iterators(*generators)

        final_res_batch: list[PoolingRequestOutput | None] = [None] * len(engine_prompts)
        async for i, res in result_generator:
            final_res_batch[i] = res

        if None in final_res_batch:
            return self.create_error_response("Failed to generate results for all prompts")

        checked_res_batch = [res for res in final_res_batch if res is not None]

        data = []
        total_prompt_tokens = 0
        for i, res in enumerate(checked_res_batch):
            logits = res.outputs.data.detach().float().cpu().reshape(-1).tolist()
            prompt_token_ids = [int(x) for x in res.prompt_token_ids]
            total_prompt_tokens += len(prompt_token_ids)
            data.append(
                RouterClassifyResponseData(
                    index=i,
                    cls_logits=logits,
                )
            )

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            total_tokens=total_prompt_tokens,
        )

        return RouterClassifyResponse(
            model=request.model,
            data=data,
            usage=usage,
        )