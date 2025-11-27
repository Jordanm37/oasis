"""Llama-4 XML-to-JSON Tool Call Adapter.

Llama-4 models on Groq sometimes output tool calls in XML format instead of
the OpenAI-compatible JSON format. This adapter intercepts responses and
converts XML-style tool calls to the expected JSON format.

Llama-4 XML format:
    <function=create_post{"content": "Hello world"}</function>

OpenAI JSON format:
    {
        "id": "call_xxx",
        "type": "function",
        "function": {
            "name": "create_post",
            "arguments": '{"content": "Hello world"}'
        }
    }

Usage:
    from oasis.llama4_tool_adapter import wrap_llama4_backend
    
    backend = ModelFactory.create(...)
    wrapped = wrap_llama4_backend(backend, model_name)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from camel.models import BaseModelBackend

logger = logging.getLogger(__name__)

# Pattern to match Llama-4's XML-style function calls
# Example: <function=create_post{"content": "Hello world"}</function>
# Also handles: <function=create_post>{"content": "Hello world"}</function>
LLAMA4_FUNCTION_PATTERNS = [
    # Pattern 1: <function=name{...}</function>
    re.compile(
        r'<function=(\w+)(\{.*?\})</function>',
        re.DOTALL
    ),
    # Pattern 2: <function=name>{...}</function>
    re.compile(
        r'<function=(\w+)>(\{.*?\})</function>',
        re.DOTALL
    ),
    # Pattern 3: <function=name {...}</function> (with space)
    re.compile(
        r'<function=(\w+)\s+(\{.*?\})</function>',
        re.DOTALL
    ),
]

# Models that may need XML-to-JSON conversion
LLAMA4_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-4-scout",
    "llama-4-maverick",
}


def is_llama4_model(model_name: str) -> bool:
    """Check if a model name is a Llama-4 model that may need conversion."""
    model_lower = model_name.lower()
    return any(
        llama4.lower() in model_lower 
        for llama4 in LLAMA4_MODELS
    )


def parse_llama4_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse all Llama-4 XML-style function calls from text.
    
    Args:
        text: Response text that may contain XML tool calls
        
    Returns:
        List of parsed tool calls in OpenAI format
    """
    tool_calls: List[Dict[str, Any]] = []
    
    for pattern in LLAMA4_FUNCTION_PATTERNS:
        for match in pattern.finditer(text):
            function_name = match.group(1)
            arguments_str = match.group(2).strip()
            
            # Validate JSON arguments
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in Llama-4 function call '{function_name}': {e}"
                )
                continue
            
            # Generate a unique call ID
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments_str
                }
            })
    
    return tool_calls


def remove_xml_tool_calls(text: str) -> str:
    """Remove XML-style tool calls from text."""
    result = text
    for pattern in LLAMA4_FUNCTION_PATTERNS:
        result = pattern.sub('', result)
    return result.strip()


def convert_llama4_response(response_content: str) -> Dict[str, Any]:
    """Convert a Llama-4 response with XML tool calls to OpenAI format.
    
    Args:
        response_content: Raw response text from Llama-4
        
    Returns:
        Dict with 'content' (text without tool calls) and 'tool_calls' (list)
    """
    tool_calls = parse_llama4_xml_tool_calls(response_content)
    clean_content = remove_xml_tool_calls(response_content)
    
    return {
        "content": clean_content if clean_content else None,
        "tool_calls": tool_calls if tool_calls else None
    }


@dataclass
class ToolCallFunction:
    """Represents a function in a tool call."""
    name: str
    arguments: str


@dataclass
class ToolCall:
    """Represents a tool call in OpenAI format."""
    id: str
    type: str
    function: ToolCallFunction


class Llama4ToolAdapter:
    """Wrapper that intercepts Llama-4 responses and converts XML tool calls to JSON.
    
    This adapter wraps a CAMEL model backend and intercepts responses that contain
    Llama-4's XML-style function calls, converting them to the OpenAI-compatible
    JSON format that CAMEL expects.
    
    Attributes:
        _backend: The underlying CAMEL model backend
        _model_name: Name of the model being wrapped
        _needs_conversion: Whether this model needs XML-to-JSON conversion
        _conversion_count: Number of successful conversions performed
    """
    
    def __init__(self, backend: "BaseModelBackend", model_name: str):
        """Initialize the adapter.
        
        Args:
            backend: The underlying CAMEL model backend to wrap
            model_name: Name of the model (used to determine if conversion is needed)
        """
        self._backend = backend
        self._model_name = model_name
        self._needs_conversion = is_llama4_model(model_name)
        self._conversion_count = 0
        
        if self._needs_conversion:
            logger.info(f"Llama4ToolAdapter: Wrapping {model_name} for XML-to-JSON conversion")
    
    @property
    def conversion_count(self) -> int:
        """Number of successful XML-to-JSON conversions performed."""
        return self._conversion_count
    
    def _convert_response(self, response: Any) -> Any:
        """Convert XML tool calls in response to JSON format.
        
        Args:
            response: The response object from the model backend
            
        Returns:
            Modified response with converted tool calls (if any)
        """
        # Get content from response
        content = getattr(response, 'content', None)
        if not content or not isinstance(content, str):
            return response
        
        # Check if response contains XML-style tool calls
        if '<function=' not in content:
            return response
        
        # Convert XML to JSON
        converted = convert_llama4_response(content)
        
        if not converted['tool_calls']:
            return response
        
        logger.info(
            f"Llama4ToolAdapter: Converted {len(converted['tool_calls'])} "
            f"XML tool call(s) for {self._model_name}"
        )
        self._conversion_count += 1
        
        # Create ToolCall objects matching CAMEL's expected format
        tool_calls = [
            ToolCall(
                id=tc['id'],
                type=tc['type'],
                function=ToolCallFunction(
                    name=tc['function']['name'],
                    arguments=tc['function']['arguments']
                )
            )
            for tc in converted['tool_calls']
        ]
        
        # Modify response object
        # Note: This modifies the response in place - CAMEL response objects
        # typically have mutable attributes
        try:
            response.content = converted['content']
            response.tool_calls = tool_calls
        except AttributeError:
            # If response is immutable, try to create a new one
            logger.warning(
                "Llama4ToolAdapter: Could not modify response object, "
                "returning original"
            )
        
        return response
    
    async def arun(
        self, 
        messages: Any, 
        response_format: Any = None, 
        tools: Any = None
    ) -> Any:
        """Async run the model and convert XML tool calls if needed.
        
        Args:
            messages: Messages to send to the model
            response_format: Optional response format specification
            tools: Optional list of tools available to the model
            
        Returns:
            Model response with converted tool calls (if applicable)
        """
        response = await self._backend.arun(messages, response_format, tools)
        
        # Only convert if we have tools and this model needs conversion
        if not self._needs_conversion or not tools:
            return response
        
        return self._convert_response(response)
    
    def run(
        self, 
        messages: Any, 
        response_format: Any = None, 
        tools: Any = None
    ) -> Any:
        """Sync run the model and convert XML tool calls if needed.
        
        Args:
            messages: Messages to send to the model
            response_format: Optional response format specification
            tools: Optional list of tools available to the model
            
        Returns:
            Model response with converted tool calls (if applicable)
        """
        response = self._backend.run(messages, response_format, tools)
        
        # Only convert if we have tools and this model needs conversion
        if not self._needs_conversion or not tools:
            return response
        
        return self._convert_response(response)
    
    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying backend.
        
        This ensures the adapter is fully compatible with the BaseModelBackend
        interface by forwarding any attribute access to the wrapped backend.
        """
        return getattr(self._backend, name)


def wrap_llama4_backend(
    backend: "BaseModelBackend", 
    model_name: str
) -> "BaseModelBackend":
    """Wrap a model backend with Llama-4 tool adapter if needed.
    
    This is the main entry point for using the adapter. It only wraps the
    backend if the model is a Llama-4 model that may need XML-to-JSON conversion.
    
    Args:
        backend: The CAMEL model backend to potentially wrap
        model_name: Name of the model
        
    Returns:
        Either the original backend (if not Llama-4) or a wrapped adapter
        
    Example:
        backend = ModelFactory.create(...)
        backend = wrap_llama4_backend(backend, "meta-llama/llama-4-scout-17b-16e-instruct")
    """
    if is_llama4_model(model_name):
        return Llama4ToolAdapter(backend, model_name)
    return backend

