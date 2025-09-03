# OpenRouter Client Implementation

This module provides a complete OpenRouter API client that replicates the functionality of the OllamaClient while using OpenRouter's API for enhanced model access and capabilities.

The main class is `Client` which follows the requested structure for tool calling and agent management.

## Features

- **Full OpenRouter API Integration**: Complete implementation matching OllamaClient interface
- **Advanced Tool Calling**: Enhanced tool calling with proper message handling and agentic loops
- **Structured Outputs**: Support for JSON schema validation using OpenRouter's structured outputs
- **Streaming Support**: Real-time response streaming capabilities
- **Agent Context Management**: Built-in agent context handling and conversation history
- **Multiple Model Support**: Access to OpenRouter's extensive model catalog

## Installation

```bash
pip install requests typing pydantic
```

## Setup

Set your OpenRouter API key as an environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or pass it directly when initializing the client:
```python
client = Client(role="Your role", api_key="your-api-key")
```

## Usage

### Basic Usage (Requested Structure)

```python
from openrouter_client import Client

# Initialize with the exact structure requested
task = "You are a helpful search agent"
model2 = "openai/gpt-oss-120b:free"

re_phrase_agent = Client(role=task, model_name=model2, agent_name="SearchAgent")

# Use invoke with tools as requested
search_queries = re_phrase_agent.invoke(
    query="""Hey Man. i love coding. 
                   I wanted to ask you several questions one of which is: What is the latest version of gpt out there?""", 
    tools=[tavily_search]
)
```

### Simple Query

```python
from openrouter_client import Client

# Initialize the client
client = Client(role="You are a helpful assistant")

# Simple query
response = client.invoke(query="What is the capital of France?")
print(response["response"])
```
```

### Tool Calling

```python
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny and 25°C"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

# Use tools with the client
response = client.invoke(
    "What's the weather in Paris and what's 15 + 27?",
    tools=[get_weather, calculate]
)
print(response["response"])
```

### Agentic Loop (Automatic Tool Calling)

```python
# Automatic tool calling until completion
response = client.invoke_with_tools_loop(
    "Calculate 5 * 8, then tell me the weather in Tokyo",
    tools=[calculate, get_weather],
    max_iterations=5
)
print(f"Final response: {response['response']}")
print(f"Total iterations: {response['total_iterations']}")
```

### Structured Outputs

```python
# Define a schema
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "temperature": {"type": "number"},
        "conditions": {"type": "string"}
    },
    "required": ["location", "temperature", "conditions"]
}

# Use structured outputs
response = client.invoke(
    "Give me weather data for New York",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "weather_data",
            "strict": True,
            "schema": weather_schema
        }
    }
)
```

### Streaming Responses

```python
# Stream responses in real-time
for chunk in client.invoke_streaming("Tell me a story about AI"):
    print(chunk, end="", flush=True)
```

## API Reference

### Client Class

#### Constructor Parameters

- `role` (str): System role/instructions for the agent
- `model_name` (str, optional): Model to use (defaults to "openai/gpt-oss-120b:free")
- `agent_name` (str, optional): Name for agent context tracking
- `api_key` (str, optional): OpenRouter API key (uses env var if not provided)
- `base_url` (str, optional): Custom API base URL

#### Main Methods

##### `invoke(query, **kwargs) -> dict`

Send a query to OpenRouter and get a response.

**Parameters:**
- `query` (str): The user's query
- `response_format` (dict, optional): Response format specification
- `tools` (list, optional): List of callable functions for tool calling
- `model_name` (str, optional): Override default model
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature (0.0-2.0)
- `top_p` (float, optional): Nucleus sampling (0.0-1.0)
- `top_k` (int, optional): Top-k sampling
- `parallel_tool_calls` (bool, optional): Allow parallel tool execution
- `tool_choice` (str/dict, optional): Control tool selection
- `stop` (list, optional): Stop sequences

**Returns:**
- Dictionary with response, tool calls, and metadata

##### `invoke_streaming(query, **kwargs) -> Iterator[str]`

Stream responses from OpenRouter.

**Parameters:** Same as `invoke()` method

**Returns:**
- Iterator yielding response chunks

##### `invoke_with_tools_loop(query, tools, max_iterations=10, **kwargs) -> dict`

Automatically handle tool calling in a loop until completion.

**Parameters:**
- `query` (str): Initial user query
- `tools` (list): Available functions for tool calling
- `max_iterations` (int): Maximum loop iterations
- `**kwargs`: Additional parameters for invoke()

**Returns:**
- Final response with iteration count

#### Helper Methods

##### `has_tool_calls(response) -> bool`
Check if a response contains tool calls.

##### `is_tool_call_response(response) -> bool`
Check if response finished with tool calls.

##### `add_tool_results_to_conversation(messages)`
Add tool result messages to conversation history.

## Model Configuration

The client defaults to `"openai/gpt-oss-120b:free"` but supports any OpenRouter model:

```python
# Override default model
response = client.invoke("Hello", model_name="anthropic/claude-3-haiku")

# List available models
models = client.list_models()
print(f"Available models: {len(models)}")
```

## Error Handling

```python
try:
    response = client.invoke("Your query")
except ValueError as e:
    print(f"Configuration error: {e}")
except requests.RequestException as e:
    print(f"API request failed: {e}")
```

## Advanced Features

### Agent Context Management

The client automatically manages agent context with XML structure:

```xml
<Agent: agent_name>
### Task to be done by the agent:
Your role description

### User Input to the agent: user_query
### Agent's Response: agent_response
Tool(s) Invoked and the tool outputs obtained: {...}
</Agent: agent_name>
```

### Conversation History

- Automatic conversation history tracking
- Tool call and result message formatting
- Context preservation across multiple turns

### Tool Schema Generation

Functions are automatically converted to OpenRouter tool schemas:

```python
def example_function(param1: str, param2: int = 5) -> str:
    """Function description.
    
    Args:
        param1: Description of param1
        param2: Description of param2 with default value
    """
    return f"Result: {param1} - {param2}"

# Automatically generates proper OpenRouter tool schema
```

## Compatibility

This implementation maintains full compatibility with the OllamaClient interface while adding OpenRouter-specific enhancements:

- ✅ All OllamaClient methods supported
- ✅ Enhanced tool calling capabilities
- ✅ OpenRouter-specific features (structured outputs, advanced parameters)
- ✅ Backward compatibility with existing code

## Testing

Run the included tests to verify functionality:

```bash
python3 test_openrouter_structure.py  # Structure and method tests
python3 test_openrouter_tool_calling.py  # Full functionality tests (requires API key)
```

## Dependencies

- `requests`: HTTP client for API calls
- `typing`: Type hints support
- `json`: JSON parsing
- `inspect`: Function introspection for tool schema generation
- `os`: Environment variable access

## License

This implementation follows the same license as the original project.
