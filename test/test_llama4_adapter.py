"""Tests for Llama-4 XML-to-JSON tool call adapter.

Run with:
    poetry run python -m pytest test/test_llama4_adapter.py -v -s
"""

import json
import pytest
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

from oasis.llama4_tool_adapter import (
    Llama4ToolAdapter,
    convert_llama4_response,
    is_llama4_model,
    parse_llama4_xml_tool_calls,
    remove_xml_tool_calls,
    wrap_llama4_backend,
)


class TestIsLlama4Model:
    """Tests for is_llama4_model function."""
    
    def test_scout_model(self):
        assert is_llama4_model("meta-llama/llama-4-scout-17b-16e-instruct") is True
    
    def test_maverick_model(self):
        assert is_llama4_model("meta-llama/llama-4-maverick-17b-128e-instruct") is True
    
    def test_short_names(self):
        assert is_llama4_model("llama-4-scout") is True
        assert is_llama4_model("llama-4-maverick") is True
    
    def test_case_insensitive(self):
        assert is_llama4_model("META-LLAMA/LLAMA-4-SCOUT-17B-16E-INSTRUCT") is True
    
    def test_non_llama4_models(self):
        assert is_llama4_model("llama-3.3-70b-versatile") is False
        assert is_llama4_model("gpt-4o") is False
        assert is_llama4_model("qwen/qwen3-32b") is False


class TestParseXmlToolCalls:
    """Tests for XML tool call parsing."""
    
    def test_basic_function_call(self):
        text = '<function=create_post{"content": "Hello world"}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "create_post"
        assert json.loads(calls[0]["function"]["arguments"]) == {"content": "Hello world"}
    
    def test_function_with_angle_bracket_format(self):
        text = '<function=create_post>{"content": "Hello world"}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "create_post"
    
    def test_function_with_space(self):
        text = '<function=create_post {"content": "Hello world"}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "create_post"
    
    def test_multiple_function_calls(self):
        text = '''
        <function=create_post{"content": "First post"}</function>
        Some text in between
        <function=like_post{"post_id": 123}</function>
        '''
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "create_post"
        assert calls[1]["function"]["name"] == "like_post"
    
    def test_complex_json_arguments(self):
        text = '<function=search_posts{"query": "test", "limit": 10, "filters": {"author": "user1"}}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "test"
        assert args["limit"] == 10
        assert args["filters"]["author"] == "user1"
    
    def test_invalid_json_skipped(self):
        text = '<function=bad_call{not valid json}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 0
    
    def test_no_function_calls(self):
        text = "Just regular text without any function calls"
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 0
    
    def test_tool_call_has_required_fields(self):
        text = '<function=test{"arg": "value"}</function>'
        calls = parse_llama4_xml_tool_calls(text)
        
        assert len(calls) == 1
        call = calls[0]
        
        # Check required fields
        assert "id" in call
        assert call["id"].startswith("call_")
        assert call["type"] == "function"
        assert "function" in call
        assert "name" in call["function"]
        assert "arguments" in call["function"]


class TestRemoveXmlToolCalls:
    """Tests for removing XML tool calls from text."""
    
    def test_removes_single_call(self):
        text = 'Before <function=test{"arg": "value"}</function> After'
        result = remove_xml_tool_calls(text)
        
        assert result == "Before  After"
    
    def test_removes_multiple_calls(self):
        text = '<function=a{"x": 1}</function> middle <function=b{"y": 2}</function>'
        result = remove_xml_tool_calls(text)
        
        assert result == "middle"
    
    def test_preserves_text_without_calls(self):
        text = "Just regular text"
        result = remove_xml_tool_calls(text)
        
        assert result == "Just regular text"


class TestConvertLlama4Response:
    """Tests for full response conversion."""
    
    def test_converts_with_content_and_calls(self):
        text = 'I will create a post for you. <function=create_post{"content": "Hello!"}</function>'
        result = convert_llama4_response(text)
        
        assert result["content"] == "I will create a post for you."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "create_post"
    
    def test_only_tool_call_no_content(self):
        text = '<function=create_post{"content": "Hello!"}</function>'
        result = convert_llama4_response(text)
        
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
    
    def test_only_content_no_calls(self):
        text = "Just a regular response without tool calls"
        result = convert_llama4_response(text)
        
        assert result["content"] == text
        assert result["tool_calls"] is None


class TestLlama4ToolAdapter:
    """Tests for the adapter wrapper class."""
    
    @pytest.fixture
    def mock_backend(self):
        """Create a mock model backend."""
        backend = MagicMock()
        backend.arun = AsyncMock()
        backend.run = MagicMock()
        return backend
    
    def test_wraps_llama4_model(self, mock_backend):
        adapter = Llama4ToolAdapter(mock_backend, "meta-llama/llama-4-scout-17b-16e-instruct")
        assert adapter._needs_conversion is True
    
    def test_does_not_wrap_other_models(self, mock_backend):
        adapter = Llama4ToolAdapter(mock_backend, "llama-3.3-70b-versatile")
        assert adapter._needs_conversion is False
    
    def test_proxies_attributes(self, mock_backend):
        mock_backend.some_attribute = "test_value"
        adapter = Llama4ToolAdapter(mock_backend, "llama-4-scout")
        
        assert adapter.some_attribute == "test_value"
    
    @pytest.mark.asyncio
    async def test_converts_xml_response(self, mock_backend):
        # Create a mock response with XML tool call
        @dataclass
        class MockResponse:
            content: str = '<function=create_post{"content": "Hello"}</function>'
            tool_calls: Optional[List] = None
        
        mock_backend.arun.return_value = MockResponse()
        
        adapter = Llama4ToolAdapter(mock_backend, "llama-4-scout")
        
        # Call with tools to trigger conversion
        result = await adapter.arun([], tools=[{"name": "create_post"}])
        
        # Verify conversion happened
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "create_post"
        assert adapter.conversion_count == 1
    
    @pytest.mark.asyncio
    async def test_no_conversion_without_tools(self, mock_backend):
        @dataclass
        class MockResponse:
            content: str = '<function=create_post{"content": "Hello"}</function>'
            tool_calls: Optional[List] = None
        
        mock_backend.arun.return_value = MockResponse()
        
        adapter = Llama4ToolAdapter(mock_backend, "llama-4-scout")
        
        # Call WITHOUT tools - should not convert
        result = await adapter.arun([], tools=None)
        
        # Content should be unchanged
        assert '<function=' in result.content
        assert adapter.conversion_count == 0
    
    @pytest.mark.asyncio
    async def test_no_conversion_for_non_llama4(self, mock_backend):
        @dataclass
        class MockResponse:
            content: str = '<function=create_post{"content": "Hello"}</function>'
            tool_calls: Optional[List] = None
        
        mock_backend.arun.return_value = MockResponse()
        
        adapter = Llama4ToolAdapter(mock_backend, "llama-3.3-70b")
        
        # Should not convert even with tools
        result = await adapter.arun([], tools=[{"name": "create_post"}])
        
        assert '<function=' in result.content


class TestWrapLlama4Backend:
    """Tests for the wrap_llama4_backend helper function."""
    
    def test_wraps_llama4_model(self):
        backend = MagicMock()
        result = wrap_llama4_backend(backend, "meta-llama/llama-4-scout-17b-16e-instruct")
        
        assert isinstance(result, Llama4ToolAdapter)
    
    def test_returns_original_for_other_models(self):
        backend = MagicMock()
        result = wrap_llama4_backend(backend, "llama-3.3-70b-versatile")
        
        assert result is backend
        assert not isinstance(result, Llama4ToolAdapter)


# Integration test with real-ish data
class TestRealWorldScenarios:
    """Tests with realistic Llama-4 output patterns."""
    
    def test_social_media_post_creation(self):
        # Real-ish Llama-4 output for creating a social media post
        text = '''I'll help you create a post about that topic.

<function=create_post{"content": "Hey everyone, hope you're all having a great day! What's been the highlight of your week so far? LBL:SUPPORTIVE I'm excited to catch up and hear about what's new with you all."}</function>'''
        
        result = convert_llama4_response(text)
        
        assert "I'll help you create a post" in result["content"]
        assert len(result["tool_calls"]) == 1
        
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert "LBL:SUPPORTIVE" in args["content"]
    
    def test_comment_creation(self):
        text = '<function=create_comment{"post_id": 42, "content": "Great post! I totally agree with your perspective."}</function>'
        
        result = convert_llama4_response(text)
        
        assert len(result["tool_calls"]) == 1
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args["post_id"] == 42
    
    def test_multiple_actions(self):
        text = '''Let me interact with these posts.

<function=like_post{"post_id": 1}</function>
<function=create_comment{"post_id": 1, "content": "Love this!"}</function>'''
        
        result = convert_llama4_response(text)
        
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "like_post"
        assert result["tool_calls"][1]["function"]["name"] == "create_comment"

