"""Gemini Native Adapter - Uses native google-genai SDK with safety settings disabled.

This adapter bypasses CAMEL's OpenAI-compatible Gemini endpoint and uses the native
google-genai SDK directly. This allows us to set safety_settings to BLOCK_NONE,
which is not possible through the OpenAI compatibility layer.

The adapter implements the same interface as CAMEL's BaseModelBackend so it can
be used as a drop-in replacement.

Usage:
    from oasis.gemini_native_adapter import GeminiNativeAdapter
    
    adapter = GeminiNativeAdapter(
        model_name="gemini-2.5-flash",
        api_key="your-api-key",
        temperature=0.3,
    )
    
    # Use like any CAMEL model backend
    response = await adapter.arun(messages, tools=tools)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Try to import google-genai SDK
try:
    from google import genai
    from google.genai import types as genai_types
    _HAS_GENAI = True
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    _HAS_GENAI = False
    logger.warning("google-genai SDK not installed. GeminiNativeAdapter will not work.")

# Import OpenAI types for response format compatibility
try:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )
    from openai.types.completion_usage import CompletionUsage
    _HAS_OPENAI_TYPES = True
except ImportError:
    _HAS_OPENAI_TYPES = False
    logger.warning("OpenAI types not available. Response format may be incompatible.")


@dataclass
class GeminiNativeConfig:
    """Configuration for GeminiNativeAdapter."""
    model_name: str = "gemini-2.5-flash"
    api_key: str = ""
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 64
    max_output_tokens: int = 4096
    timeout: float = 180.0


def _build_safety_settings() -> List[Any]:
    """Build safety settings with all categories set to BLOCK_NONE.
    
    This is the most permissive setting available through the Gemini API.
    Note: Some core harms (e.g., CSAM) remain hard-blocked regardless of settings.
    
    The google-genai SDK uses string literals for category and threshold,
    not enum values. See SafetySetting signature for valid values.
    """
    if not _HAS_GENAI or genai_types is None:
        return []
    
    # Categories as string literals (from SafetySetting signature)
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    
    # Use BLOCK_NONE for most permissive filtering
    # Note: "OFF" is also available but BLOCK_NONE is more widely supported
    return [
        genai_types.SafetySetting(
            category=category,
            threshold="BLOCK_NONE",
        )
        for category in categories
    ]


def _convert_openai_messages_to_gemini(
    messages: List[Dict[str, Any]]
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert OpenAI-format messages to Gemini native format.
    
    Args:
        messages: List of OpenAI-format messages with 'role' and 'content'
        
    Returns:
        Tuple of (system_instruction, contents) where:
        - system_instruction: Extracted system message content (or None)
        - contents: List of Gemini-format content dicts
    """
    system_instruction: Optional[str] = None
    contents: List[Dict[str, Any]] = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Handle system messages separately
        if role == "system":
            # Concatenate multiple system messages
            if system_instruction:
                system_instruction += "\n" + content
            else:
                system_instruction = content
            continue
        
        # Map OpenAI roles to Gemini roles
        # OpenAI: system, user, assistant, tool
        # Gemini: user, model
        if role == "assistant":
            gemini_role = "model"
        elif role == "tool":
            # Tool responses go as user messages in Gemini
            gemini_role = "user"
            # Format tool response
            tool_call_id = msg.get("tool_call_id", "")
            content = f"[Tool Response for {tool_call_id}]: {content}"
        else:
            gemini_role = "user"
        
        # Handle different content types
        if isinstance(content, str):
            parts = [{"text": content or "null"}]  # Gemini doesn't accept empty strings
        elif isinstance(content, list):
            # Multi-modal content (text + images)
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Handle image URLs if needed
                        parts.append({"text": f"[Image: {item.get('image_url', {}).get('url', '')}]"})
                else:
                    parts.append({"text": str(item)})
            if not parts:
                parts = [{"text": "null"}]
        else:
            parts = [{"text": str(content) or "null"}]
        
        # Handle tool calls in assistant messages
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # Add function call parts
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    parts.append({
                        "function_call": {
                            "name": func.get("name", ""),
                            "args": json.loads(func.get("arguments", "{}")),
                        }
                    })
        
        contents.append({
            "role": gemini_role,
            "parts": parts,
        })
    
    return system_instruction, contents


def _convert_json_schema_types(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON Schema types to Gemini's uppercase type format.
    
    Gemini expects: TYPE_UNSPECIFIED, STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT
    JSON Schema uses: string, number, integer, boolean, array, object
    """
    if not isinstance(schema, dict):
        return schema
    
    result = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            # Convert lowercase JSON Schema type to uppercase Gemini type
            result[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            # Recursively convert property types
            result[key] = {
                prop_name: _convert_json_schema_types(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            # Convert array item types
            result[key] = _convert_json_schema_types(value)
        elif isinstance(value, dict):
            # Recursively handle nested schemas
            result[key] = _convert_json_schema_types(value)
        else:
            result[key] = value
    
    return result


def _convert_openai_tools_to_gemini(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI-format tools to Gemini function declarations.
    
    Args:
        tools: List of OpenAI-format tool definitions
        
    Returns:
        List of Gemini-format function declarations, or None if no tools
    """
    if not tools:
        return None
    
    function_declarations = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        
        func = tool.get("function", {})
        declaration = {
            "name": func.get("name", ""),
            "description": func.get("description", ""),
        }
        
        # Convert parameters schema with type conversion
        params = func.get("parameters", {})
        if params:
            # Convert JSON Schema types to Gemini format
            declaration["parameters"] = _convert_json_schema_types(params)
        
        function_declarations.append(declaration)
    
    return function_declarations if function_declarations else None


def _convert_gemini_response_to_openai(
    response: Any,
    model_name: str,
) -> "ChatCompletion":
    """Convert Gemini response to OpenAI ChatCompletion format.
    
    This allows the response to work with CAMEL's agent system which expects
    OpenAI-format responses.
    """
    if not _HAS_OPENAI_TYPES:
        raise RuntimeError("OpenAI types not available for response conversion")
    
    # Extract response content
    text_content = ""
    tool_calls: List[ChatCompletionMessageToolCall] = []
    
    try:
        # Get the first candidate's content
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts if hasattr(candidate.content, 'parts') else []
                for part in parts:
                    # Text content
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text
                    # Function calls
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                type="function",
                                function=Function(
                                    name=fc.name if hasattr(fc, 'name') else "",
                                    arguments=json.dumps(dict(fc.args) if hasattr(fc, 'args') else {}),
                                ),
                            )
                        )
        elif hasattr(response, 'text'):
            text_content = response.text or ""
    except Exception as e:
        logger.warning(f"Error extracting Gemini response content: {e}")
        text_content = str(response) if response else ""
    
    # Build usage stats
    usage = CompletionUsage(
        prompt_tokens=0,  # Gemini doesn't always provide this
        completion_tokens=0,
        total_tokens=0,
    )
    
    # Try to get actual usage if available
    if hasattr(response, 'usage_metadata'):
        um = response.usage_metadata
        usage = CompletionUsage(
            prompt_tokens=getattr(um, 'prompt_token_count', 0) or 0,
            completion_tokens=getattr(um, 'candidates_token_count', 0) or 0,
            total_tokens=getattr(um, 'total_token_count', 0) or 0,
        )
    
    # Build the message
    message = ChatCompletionMessage(
        role="assistant",
        content=text_content if text_content else None,
        tool_calls=tool_calls if tool_calls else None,
    )
    
    # Build the choice
    choice = Choice(
        index=0,
        message=message,
        finish_reason="stop" if not tool_calls else "tool_calls",
    )
    
    # Build the completion
    completion = ChatCompletion(
        id=f"gemini-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[choice],
        usage=usage,
    )
    
    return completion


class GeminiNativeAdapter:
    """Adapter that uses native google-genai SDK with safety settings disabled.
    
    This adapter implements a similar interface to CAMEL's BaseModelBackend,
    allowing it to be used as a drop-in replacement for simulation agents.
    
    Key features:
    - Uses native Gemini API (not OpenAI compatibility layer)
    - Sets safety_settings to BLOCK_NONE for all categories
    - Converts OpenAI-format messages to Gemini format
    - Handles tool/function calls
    - Returns OpenAI-compatible ChatCompletion responses
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 64,
        max_output_tokens: int = 4096,
        timeout: float = 180.0,
        model_config_dict: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize the Gemini native adapter.
        
        Args:
            model_name: Gemini model to use (e.g., "gemini-2.5-flash")
            api_key: Gemini API key (falls back to GEMINI_API_KEY env var)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_output_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            model_config_dict: Additional config (for CAMEL compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        if not _HAS_GENAI:
            raise RuntimeError(
                "google-genai SDK not installed. "
                "Install with: pip install google-genai"
            )
        
        self.model_name = model_name
        self.model_type = model_name  # For CAMEL compatibility
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key.")
        
        # Apply model_config_dict overrides if provided
        if model_config_dict:
            temperature = model_config_dict.get("temperature", temperature)
            top_p = model_config_dict.get("top_p", top_p)
            top_k = model_config_dict.get("top_k", top_k)
            max_output_tokens = model_config_dict.get("max_output_tokens", max_output_tokens)
            max_output_tokens = model_config_dict.get("max_tokens", max_output_tokens)
        
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        
        # Store config dict for CAMEL compatibility
        self.model_config_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        # Initialize the client
        self._client = genai.Client(api_key=self.api_key)
        
        # Pre-build safety settings
        self._safety_settings = _build_safety_settings()
        
        logger.info(
            f"GeminiNativeAdapter initialized: model={model_name}, "
            f"temperature={temperature}, safety=BLOCK_NONE"
        )
    
    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Build GenerateContentConfig with safety settings disabled."""
        if not _HAS_GENAI or genai_types is None:
            raise RuntimeError("google-genai SDK not available")
        
        config_kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "safety_settings": self._safety_settings,
        }
        
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        
        # Add tools if provided
        if tools:
            function_declarations = _convert_openai_tools_to_gemini(tools)
            if function_declarations:
                config_kwargs["tools"] = [
                    genai_types.Tool(function_declarations=function_declarations)
                ]
        
        return genai_types.GenerateContentConfig(**config_kwargs)
    
    def run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> "ChatCompletion":
        """Synchronous inference using native Gemini API.
        
        Args:
            messages: OpenAI-format messages
            response_format: Pydantic model for structured output (not fully supported)
            tools: OpenAI-format tool definitions
            
        Returns:
            OpenAI-compatible ChatCompletion response
        """
        # Convert messages
        system_instruction, contents = _convert_openai_messages_to_gemini(messages)
        
        # Build config
        config = self._build_config(system_instruction, tools)
        
        # Make the request
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            
            # Convert to OpenAI format
            return _convert_gemini_response_to_openai(response, self.model_name)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> "ChatCompletion":
        """Asynchronous inference using native Gemini API.
        
        Note: The google-genai SDK doesn't have native async support yet,
        so we run the sync version in a thread pool.
        
        Args:
            messages: OpenAI-format messages
            response_format: Pydantic model for structured output (not fully supported)
            tools: OpenAI-format tool definitions
            
        Returns:
            OpenAI-compatible ChatCompletion response
        """
        # Convert messages
        system_instruction, contents = _convert_openai_messages_to_gemini(messages)
        
        # Build config
        config = self._build_config(system_instruction, tools)
        
        # Run in thread pool since google-genai doesn't have native async
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            return self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        
        try:
            response = await loop.run_in_executor(None, _sync_call)
            
            # Convert to OpenAI format
            return _convert_gemini_response_to_openai(response, self.model_name)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    # CAMEL compatibility methods
    def check_model_config(self) -> None:
        """Check model configuration (CAMEL compatibility)."""
        pass
    
    @property
    def token_limit(self) -> int:
        """Get token limit (CAMEL compatibility)."""
        # Gemini 2.5 Flash has 1M context window
        return 1_000_000
    
    @property
    def stream(self) -> bool:
        """Check if streaming is enabled (CAMEL compatibility)."""
        return False


def create_gemini_native_backend(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    model_config_dict: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    **kwargs: Any,
) -> GeminiNativeAdapter:
    """Factory function to create a GeminiNativeAdapter.
    
    This matches the signature expected by model_provider.py.
    
    Args:
        model_name: Gemini model to use
        api_key: API key (falls back to env var)
        model_config_dict: Configuration dictionary
        timeout: Request timeout
        **kwargs: Additional arguments
        
    Returns:
        Configured GeminiNativeAdapter instance
    """
    return GeminiNativeAdapter(
        model_name=model_name,
        api_key=api_key,
        model_config_dict=model_config_dict,
        timeout=timeout,
        **kwargs,
    )

