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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Import OpenAI types for reconstructing responses
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai import BadRequestError

if TYPE_CHECKING:
    from camel.models import BaseModelBackend

logger = logging.getLogger(__name__)

# Known integer parameters in OASIS tool schemas that LLMs sometimes return as strings
# These will be coerced from string to int automatically
INTEGER_PARAMETERS = {
    "post_id",
    "comment_id",
    "user_id",
    "group_id",
    "message_id",
    "original_post_id",
    "target_user_id",
}

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
    # Pattern 4: <function=name {...} </function> (with trailing space before close)
    re.compile(
        r'<function=(\w+)\s+(\{.*?\})\s*</function>',
        re.DOTALL
    ),
    # Pattern 5: Malformed closing tag <function> instead of </function>
    re.compile(
        r'<function=(\w+)>(\{.*?\})<function>',
        re.DOTALL
    ),
]

# Pattern to match Qwen's <tool_call> format
# Example: <tool_call>{"name": "create_post", "arguments": {"content": "Hello"}}</tool_call>
QWEN_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL
)

# Models that may need XML-to-JSON conversion
# NOTE: This includes Llama-3.x, Llama-4, and Qwen models that sometimes output
# XML-style tool calls like <function=name>{args}</function> or <tool_call>...</tool_call>
# instead of proper JSON tool call format.
XML_TOOL_CALL_MODELS = {
    # Llama-4 models
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-4-scout",
    "llama-4-maverick",
    # Llama-3.x models (on Groq, these sometimes output XML format)
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    # Qwen models (output <tool_call>...</tool_call> format)
    "qwen/qwen3-32b",
    "qwen/qwen3-8b",
}

# Keep old name for backward compatibility
LLAMA4_MODELS = XML_TOOL_CALL_MODELS


def is_llama4_model(model_name: str) -> bool:
    """Check if a model name needs XML-to-JSON tool call conversion.
    
    Despite the name (kept for backward compatibility), this checks all models
    that may output XML-style tool calls, including Llama-3.x and Qwen models.
    """
    model_lower = model_name.lower()
    return any(
        model.lower() in model_lower 
        for model in XML_TOOL_CALL_MODELS
    )


def coerce_integer_parameters(arguments_str: str) -> str:
    """Coerce string values to integers for known integer parameters.
    
    LLMs sometimes return integer parameters as strings (e.g., "post_id": "5" 
    instead of "post_id": 5). This function parses the JSON arguments and 
    converts known integer fields from strings to actual integers.
    
    Args:
        arguments_str: JSON string of function arguments
        
    Returns:
        JSON string with integer parameters coerced to int type
        
    Example:
        >>> coerce_integer_parameters('{"post_id": "5", "content": "hello"}')
        '{"post_id": 5, "content": "hello"}'
    """
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError:
        return arguments_str  # Return original if can't parse
    
    modified = False
    for key in INTEGER_PARAMETERS:
        if key in args and isinstance(args[key], str):
            try:
                args[key] = int(args[key])
                modified = True
                logger.debug(f"Coerced {key} from string to int: {args[key]}")
            except (ValueError, TypeError):
                # Keep as string if can't convert
                pass
    
    if modified:
        return json.dumps(args)
    return arguments_str


def parse_llama4_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse all XML-style function calls from text (Llama and Qwen formats).
    
    Also coerces known integer parameters from strings to integers.
    
    Args:
        text: Response text that may contain XML tool calls
        
    Returns:
        List of parsed tool calls in OpenAI format
    """
    tool_calls: List[Dict[str, Any]] = []
    
    # Parse Llama-style <function=name>{args}</function> patterns
    for pattern in LLAMA4_FUNCTION_PATTERNS:
        for match in pattern.finditer(text):
            function_name = match.group(1)
            arguments_str = match.group(2).strip()
            
            # Validate JSON arguments
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError:
                # Attempt to repair common unescaped quote issues in "content" field
                # Llama-3 models often output: {"content": "text with "quotes" inside"}
                try:
                    if '"content": "' in arguments_str:
                        prefix = '"content": "'
                        start_idx = arguments_str.find(prefix) + len(prefix)
                        # Find the end of the string. If it's the only/last field, it ends near }
                        # We scan from right, skipping } and whitespace
                        r_idx = len(arguments_str) - 1
                        while r_idx > start_idx and arguments_str[r_idx] in ['}', ' ', '\n', '\r', '\t']:
                            r_idx -= 1
                        
                        # If the character at r_idx is '"', that's likely the closing quote
                        if arguments_str[r_idx] == '"':
                            end_idx = r_idx
                            content = arguments_str[start_idx:end_idx]
                            # Escape unescaped quotes
                            fixed_content = content.replace('"', '\\"')
                            # Reconstruct
                            repaired_str = arguments_str[:start_idx] + fixed_content + arguments_str[end_idx:]
                            # Verify
                            json.loads(repaired_str)
                            arguments_str = repaired_str
                except Exception:
                    pass # Fall through to original error logging if repair fails

            try:
                json.loads(arguments_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in Llama function call '{function_name}': {e}"
                )
                continue
            
            # Coerce string values to integers for known integer parameters
            arguments_str = coerce_integer_parameters(arguments_str)
            
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
    
    # Parse Qwen-style <tool_call>{"name": ..., "arguments": {...}}</tool_call> patterns
    for match in QWEN_TOOL_CALL_PATTERN.finditer(text):
        try:
            tool_call_json = json.loads(match.group(1).strip())
            function_name = tool_call_json.get("name", "")
            arguments = tool_call_json.get("arguments", {})
            
            if not function_name:
                logger.warning("Qwen tool_call missing 'name' field")
                continue
            
            # Arguments can be a dict or a string
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments)
            else:
                arguments_str = str(arguments)
            
            # Coerce string values to integers for known integer parameters
            arguments_str = coerce_integer_parameters(arguments_str)
            
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
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in Qwen tool_call: {e}")
            continue
    
    return tool_calls


def remove_xml_tool_calls(text: str) -> str:
    """Remove XML-style tool calls from text."""
    result = text
    for pattern in LLAMA4_FUNCTION_PATTERNS:
        result = pattern.sub('', result)
    # Also remove Qwen-style tool calls
    result = QWEN_TOOL_CALL_PATTERN.sub('', result)
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
        
        Also catches 400 BadRequestErrors caused by API-side validation of
        malformed tool calls (e.g., XML in JSON mode) and recovers by parsing
        the 'failed_generation' from the error response.
        
        Args:
            messages: Messages to send to the model
            response_format: Optional response format specification
            tools: Optional list of tools available to the model
            
        Returns:
            Model response with converted tool calls (if applicable)
        """
        try:
            response = await self._backend.arun(messages, response_format, tools)
        except BadRequestError as e:
            # CRITICAL FIX: If the API rejects the generation (400), check if we can salvage it.
            # Groq returns the bad output in 'failed_generation'.
            if not self._needs_conversion or not tools:
                raise e
                
            # Robust extraction of failed_generation
            failed_gen = None
            if hasattr(e, "body") and isinstance(e.body, dict):
                failed_gen = e.body.get("error", {}).get("failed_generation")
            
            if not failed_gen and hasattr(e, "response"):
                try:
                    # Try getting from response json if body attribute didn't work
                    import json
                    if hasattr(e.response, "json"):
                        # It might be a method or property depending on version
                        data = e.response.json() if callable(e.response.json) else e.response.json
                        if isinstance(data, dict):
                            failed_gen = data.get("error", {}).get("failed_generation")
                except Exception:
                    pass
            
            if not failed_gen:
                raise e
                
            logger.warning(f"Caught 400 error with failed_generation. Attempting recovery...")
            
            # Attempt to parse tool calls from the failed generation
            parsed_tool_calls = parse_llama4_xml_tool_calls(failed_gen)
            
            if not parsed_tool_calls:
                # Parsing failed, re-raise original error
                logger.warning("Recovery failed: No valid tool calls found in failed_generation.")
                raise e
                
            logger.info(f"RECOVERED from 400 error! Found {len(parsed_tool_calls)} tool calls.")
            self._conversion_count += 1
            
            # Clean content
            content = remove_xml_tool_calls(failed_gen)
            
            # Construct a valid ChatCompletion object manually
            # This mimics what the OpenAI SDK would return on success
            tool_calls_objects = [
                ChatCompletionMessageToolCall(
                    id=tc['id'],
                    type=tc['type'],
                    function=Function(
                        name=tc['function']['name'],
                        arguments=tc['function']['arguments']
                    )
                )
                for tc in parsed_tool_calls
            ]
            
            message = ChatCompletionMessage(
                content=content if content else None,
                role="assistant",
                tool_calls=tool_calls_objects,
            )
            
            choice = Choice(
                finish_reason="tool_calls",
                index=0,
                message=message,
                logprobs=None,
            )
            
            completion = ChatCompletion(
                id=f"chatcmpl-recovered-{uuid.uuid4().hex[:8]}",
                choices=[choice],
                created=int(time.time()),
                model=self._model_name,
                object="chat.completion",
            )
            
            return completion

        # Standard success path (200 OK)
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
        
        Also catches 400 BadRequestErrors and attempts recovery.
        
        Args:
            messages: Messages to send to the model
            response_format: Optional response format specification
            tools: Optional list of tools available to the model
            
        Returns:
            Model response with converted tool calls (if applicable)
        """
        try:
            response = self._backend.run(messages, response_format, tools)
        except BadRequestError as e:
            # Sync version of the recovery logic
            if not self._needs_conversion or not tools:
                raise e
                
            error_body = getattr(e, "body", {}) or {}
            failed_gen = error_body.get("error", {}).get("failed_generation")
            
            if not failed_gen:
                raise e
                
            logger.warning(f"Caught 400 error (sync) with failed_generation. Attempting recovery...")
            
            parsed_tool_calls = parse_llama4_xml_tool_calls(failed_gen)
            
            if not parsed_tool_calls:
                raise e
                
            logger.info(f"RECOVERED (sync) from 400 error! Found {len(parsed_tool_calls)} tool calls.")
            self._conversion_count += 1
            
            content = remove_xml_tool_calls(failed_gen)
            
            tool_calls_objects = [
                ChatCompletionMessageToolCall(
                    id=tc['id'],
                    type=tc['type'],
                    function=Function(
                        name=tc['function']['name'],
                        arguments=tc['function']['arguments']
                    )
                )
                for tc in parsed_tool_calls
            ]
            
            message = ChatCompletionMessage(
                content=content if content else None,
                role="assistant",
                tool_calls=tool_calls_objects,
            )
            
            choice = Choice(
                finish_reason="tool_calls",
                index=0,
                message=message,
                logprobs=None,
            )
            
            completion = ChatCompletion(
                id=f"chatcmpl-recovered-{uuid.uuid4().hex[:8]}",
                choices=[choice],
                created=int(time.time()),
                model=self._model_name,
                object="chat.completion",
            )
            
            return completion

        # Standard success path
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

