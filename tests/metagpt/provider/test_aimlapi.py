#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : test_aimlapi.py
@Desc    : aimlapi.com registration and request params.
"""
from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.provider.llm_provider_registry import LLM_REGISTRY
from metagpt.provider.openai_api import OpenAILLM

AIMLAPI_BASE_URL = "https://api.aimlapi.com/v1"


def aimlapi_config(**kwargs) -> LLMConfig:
    params = {
        "api_type": LLMType.AIMLAPI,
        "base_url": AIMLAPI_BASE_URL,
        "api_key": "mock_api_key",
        "model": "openai/gpt-5-5",
    }
    params.update(kwargs)
    return LLMConfig(**params)


def test_api_type_is_registered_to_openai_provider():
    assert LLMType("aimlapi") is LLMType.AIMLAPI
    assert LLM_REGISTRY.get_provider(LLMType.AIMLAPI) is OpenAILLM


def test_request_params_omit_unset_keys():
    """aimlapi.com answers 400 for a `null` `tools`, `seed` or `response_format` on
    its stricter models, so an option that is not set has to be absent from the
    body rather than present as null. `tools` is the one that bites: a loop that
    clears it between turns succeeds on turn 1 and fails on turn 2."""
    llm = OpenAILLM(aimlapi_config())
    kwargs = llm._cons_kwargs([{"role": "user", "content": "hi"}])

    assert None not in kwargs.values()
    for key in ("tools", "tool_choice", "seed", "response_format", "parallel_tool_calls", "stop"):
        assert key not in kwargs
