import inspect
import json
import typing as t
import requests
import os
import datetime
CURRENT_DATE=datetime.datetime.now().strftime("%d %b %Y")
GENERIC_AGENT_INSTRUCTIONS = """You are a helpful Agent among a group of agents trying to solve a problem. Each agent is tasked with a part or the entire problem.
You will be given your task. You will have access to all the relevant information and tools. You can also see the work already done by other agents. Use that information if required.


## Instructions
1. View the context, the task executed, the results, the tool call results of the other agents.
2. Reason and execute your task.
3. You have access to multiple tools as listed. You can only call tools which are relevant to your task. Other agents might have executed other tools which you dont have access to.
4. Make the necessary tool calls if your tool descriptions match the task
5. your task will be enclosed in <YOUR TASK></YOUR TASK> tags. This is your task. Only execute this task.
6. The work done by other agents will be enclosed in <Agent: Agent Name></Agent: Agent Name> tags. There may be multiple of these.
7. Any information to be used as reference (from documents or internet) about the problem is enclosed in <CONTEXT></CONTEXT> tags
8. Some general information is enclosed in <general information> tags

<general information>
1. The current date is: {CURRENT_DATE}
</general information>

Following is the relevant information from other agents (if any):
{other_agents_history}



Here is the required context (if any) that you should refer to:
<CONTEXT>
{context}
</CONTEXT>

<YOUR TASK>
{task}
</YOUR TASK>


"""

class Client:
    """OpenRouter API client that replicates OllamaClient functionality."""
    
    TYPE_MAPPING = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }

    def __init__(
        self, 
        role: str,
        history_from_other_agents: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "openai/gpt-oss-120b:free",
        this_agent_context: str = "",
        system_instructions: str = "",
        agent_name: str = "",
        api_key: str = None,
        site_url: str = "",
        app_name: str = ""
    ):
        """Initialize the OpenRouter agent client.
        
        Args:
            role: The role description for this agent
            history_from_other_agents: History from other agents
            base_url: URL of the OpenRouter API
            model_name: Default model to use for all interactions
            this_agent_context: Context specific to this agent
            system_instructions: System instructions to guide the agent's behavior
            agent_name: Name/identifier for this agent
            api_key: OpenRouter API key (if not provided, will use OPENROUTER_API_KEY env var)
            site_url: Site URL for OpenRouter rankings (optional)
            app_name: App name for OpenRouter rankings (optional)
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = model_name
        self.agent_name = agent_name
        self.conversation_history: list[dict] = []  # Store conversation history
        self.tool_call_results: dict = {}  # Store tool call results
        
        # OpenRouter specific configuration
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.site_url = site_url or os.getenv("OR_SITE_URL", "")
        self.app_name = app_name or os.getenv("OR_APP_NAME", "")
        
        # New agent context variables
        self.role = role
        self.history_from_other_agents = history_from_other_agents
        self.all_tool_names = ""
        self.only_this_agent_context = this_agent_context

        # System instructions is the same as the generic agent instructions - fix that to remove redundancy
        self.generic_agent_instructions = GENERIC_AGENT_INSTRUCTIONS.format(
            task=self.role, 
            other_agents_history=self.history_from_other_agents,
            context=self.only_this_agent_context,
            CURRENT_DATE=CURRENT_DATE
        )
        self.system_instructions = self.generic_agent_instructions
        
        # Test API connection
        self._test_connection()

    def _get_headers(self) -> dict:
        """Get headers for OpenRouter API requests."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Add optional headers for OpenRouter rankings
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
            
        return headers

    def _test_connection(self) -> None:
        """Test the connection to OpenRouter API."""
        try:
            response = requests.get(
                f"{self.base_url}/models", 
                headers=self._get_headers(),
                timeout=10.0
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to OpenRouter API: {e}")

    def set_system_instructions(self, instructions: str) -> None:
        """Update the system instructions for the agent.
        
        Args:
            instructions: New system instructions to guide the agent's behavior
        """
        self.system_instructions = instructions

    def get_system_instructions(self) -> str:
        """Get the current system instructions.
        
        Returns:
            Current system instructions
        """
        return self.system_instructions

    def get_default_model(self) -> str:
        """Get the current default model.
        
        Returns:
            Current default model name
        """
        return self.default_model

    def set_default_model(self, model_name: str) -> None:
        """Update the default model for the agent.
        
        Args:
            model_name: New default model name
        """
        self.default_model = model_name

    def get_conversation_history(self) -> list[dict]:
        """Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def list_available_models(self) -> list[dict]:
        """List all available models from OpenRouter.
        
        Returns:
            List of available models with their information
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10.0
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch models list: {e}")
            return []

    def _build_agent_context(self, agent_response: str = "", tool_results: dict = None) -> None:
        """Build the only_this_agent_context with the specified XML structure.
        
        Args:
            agent_response: The agent's response to include in context
            tool_results: Dictionary of tool names and their results
        """
        if not self.agent_name:
            return
            
        tool_section = ""
        if tool_results:
            tool_section = f"Tool(s) Invoked and the tool outputs obtained: {tool_results}"
        
        self.only_this_agent_context = f"""<Agent: {self.agent_name}>
            ### Task to be done by the agent:
            {self.role}

            ### User Input to the agent: {getattr(self, 'query', '')}
            ### Agent's Response: {agent_response}
            {tool_section}
            </Agent: {self.agent_name}>"""

    def _execute_tool_calls(self, tool_calls: list, available_tools: t.Optional[t.Iterable[t.Callable]]) -> dict:
        """Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool calls from the model response
            available_tools: List of available callable functions
            
        Returns:
            Dictionary mapping tool names to their execution results
        """
        if not tool_calls or not available_tools:
            return {}
        
        # Create a mapping of tool names to functions
        tool_map = {func.__name__: func for func in available_tools}
        execution_results = {}
        
        for tool_call in tool_calls:
            try:
                # Extract tool call information (OpenRouter format)
                function_info = tool_call.get("function", {})
                tool_name = function_info.get("name", "")
                tool_args = function_info.get("arguments", {})
                
                # Parse arguments if they're a JSON string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        execution_results[tool_name] = f"Error: Invalid JSON arguments: {tool_args}"
                        continue
                
                # Execute the tool if it's available
                if tool_name in tool_map:
                    func = tool_map[tool_name]
                    try:
                        result = func(**tool_args)
                        execution_results[tool_name] = result
                    except Exception as e:
                        execution_results[tool_name] = f"Error executing {tool_name}: {str(e)}"
                else:
                    execution_results[tool_name] = f"Error: Tool '{tool_name}' not found in available tools"
                    
            except Exception as e:
                execution_results[f"unknown_tool_{len(execution_results)}"] = f"Error processing tool call: {str(e)}"
        
        return execution_results

    def _execute_tool_calls_with_messages(
        self, 
        tool_calls: list, 
        available_tools: t.Optional[t.Iterable[t.Callable]]
    ) -> tuple[dict, list[dict]]:
        """Execute tool calls and return both results and proper message format.
        
        Args:
            tool_calls: List of tool calls from the model response
            available_tools: List of available callable functions
            
        Returns:
            Tuple of (execution_results_dict, tool_result_messages)
        """
        if not tool_calls or not available_tools:
            return {}, []
        
        # Create a mapping of tool names to functions
        tool_map = {func.__name__: func for func in available_tools}
        execution_results = {}
        tool_result_messages = []
        
        for tool_call in tool_calls:
            try:
                # Extract tool call information (OpenRouter format)
                tool_call_id = tool_call.get("id", f"call_{len(tool_result_messages)}")
                function_info = tool_call.get("function", {})
                tool_name = function_info.get("name", "")
                tool_args = function_info.get("arguments", {})
                
                # Parse arguments if they're a JSON string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        error_msg = f"Error: Invalid JSON arguments: {tool_args}"
                        execution_results[tool_name] = error_msg
                        tool_result_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": error_msg
                        })
                        continue
                
                # Execute the tool if it's available
                if tool_name in tool_map:
                    func = tool_map[tool_name]
                    try:
                        result = func(**tool_args)
                        execution_results[tool_name] = result
                        
                        # Convert result to string if it's not already
                        content = json.dumps(result) if not isinstance(result, str) else result
                        
                        tool_result_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content
                        })
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        execution_results[tool_name] = error_msg
                        tool_result_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": error_msg
                        })
                else:
                    error_msg = f"Error: Tool '{tool_name}' not found in available tools"
                    execution_results[tool_name] = error_msg
                    tool_result_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Error processing tool call: {str(e)}"
                execution_results[f"unknown_tool_{len(execution_results)}"] = error_msg
                tool_result_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_error_{len(tool_result_messages)}"),
                    "content": error_msg
                })
        
        return execution_results, tool_result_messages

    def add_tool_results_to_conversation(self, tool_result_messages: list[dict]) -> None:
        """Add tool result messages to conversation history.
        
        Args:
            tool_result_messages: List of tool result messages to add
        """
        self.conversation_history.extend(tool_result_messages)

    def has_tool_calls(self, response: dict) -> bool:
        """Check if a response contains tool calls.
        
        Args:
            response: Response from invoke() method
            
        Returns:
            True if response contains tool calls
        """
        return bool(response.get("tool_calls"))

    def is_tool_call_response(self, response: dict) -> bool:
        """Check if response finished with tool calls.
        
        Args:
            response: Response from invoke() method
            
        Returns:
            True if finish reason is tool_calls
        """
        return response.get("finish_reason") == "tool_calls"

    def invoke_with_tools_loop(
        self,
        query: str,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        max_iterations: int = 10,
        **kwargs
    ) -> dict:
        """Invoke with automatic tool calling loop (agentic behavior).
        
        This method automatically handles tool calls in a loop until the model
        provides a final response or max_iterations is reached.
        
        Args:
            query: The user's query
            tools: Optional list of functions the agent can call
            max_iterations: Maximum number of iterations to prevent infinite loops
            **kwargs: Additional parameters passed to invoke()
            
        Returns:
            Final response from the model after all tool calls are completed
        """
        if not tools:
            return self.invoke(query, **kwargs)
        
        # Start with the initial query
        current_response = self.invoke(query, tools=tools, **kwargs)
        iteration_count = 1
        
        # Continue while we have tool calls and haven't exceeded max iterations
        while (self.has_tool_calls(current_response) and 
               iteration_count < max_iterations):
            
            iteration_count += 1
            
            # Make another call with the updated conversation history
            # The tools results are already in the conversation history
            current_response = self.invoke(
                "",  # Empty query since context is in conversation history
                tools=tools,
                **kwargs
            )
        
        if iteration_count >= max_iterations:
            current_response["warning"] = "Maximum iterations reached in tool calling loop"
            
        current_response["total_iterations"] = iteration_count
        return current_response

    def update_agent_context(self, agent_response: str = "", tool_results: dict = None) -> None:
        """Update the agent context with new response and tool results.
        
        Args:
            agent_response: The agent's response to include in context
            tool_results: Dictionary of tool names and their results
        """
        self._build_agent_context(agent_response, tool_results)

    def get_agent_context(self) -> str:
        """Get the current agent context.
        
        Returns:
            The XML-formatted agent context string
        """
        return self.only_this_agent_context

    def set_role(self, role: str) -> None:
        """Set the agent's role.
        
        Args:
            role: The role description for this agent
        """
        self.role = role

    def get_role(self) -> str:
        """Get the agent's role.
        
        Returns:
            The agent's role
        """
        return self.role

    def set_generic_agent_instructions(self, instructions: str) -> None:
        """Set generic agent instructions.
        
        Args:
            instructions: Generic instructions for the agent
        """
        self.generic_agent_instructions = instructions

    def get_generic_agent_instructions(self) -> str:
        """Get generic agent instructions.
        
        Returns:
            The generic agent instructions
        """
        return self.generic_agent_instructions

    def set_all_tool_names(self, tool_names: str) -> None:
        """Set all available tool names.
        
        Args:
            tool_names: String containing all tool names
        """
        self.all_tool_names = tool_names

    def get_all_tool_names(self) -> str:
        """Get all available tool names.
        
        Returns:
            String containing all tool names
        """
        return self.all_tool_names

    def set_history_from_other_agents(self, history: str) -> None:
        """Set history from other agents.
        
        Args:
            history: History information from other agents
        """
        self.history_from_other_agents = history

    def get_history_from_other_agents(self) -> str:
        """Get history from other agents.
        
        Returns:
            History from other agents
        """
        return self.history_from_other_agents

    def get_tool_call_results(self) -> dict:
        """Get tool call results.
        
        Returns:
            Dictionary of tool call results
        """
        return self.tool_call_results.copy()

    def clear_tool_call_results(self) -> None:
        """Clear tool call results."""
        self.tool_call_results.clear()

    def add_context_from_other_agents(self, context: str) -> None:
        """Add context from other agents to the conversation history.
        
        Args:
            context: Context information from other agents
        """
        if context.strip():
            self.conversation_history.append({
                "role": "user",
                "content": f"Context from other agents: {context}"
            })

    def _get_json_type(self, python_type: t.Any) -> str:
        """Convert Python type to JSON schema type."""
        # Handle Optional types (Union with None)
        origin = t.get_origin(python_type)
        if origin is t.Union:
            args = t.get_args(python_type)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                python_type = non_none_types[0]
        
        # Handle generic types
        if origin in (list, tuple):
            return "array"
        if origin is dict:
            return "object"
            
        return self.TYPE_MAPPING.get(python_type, "string")

    def _extract_function_info(self, func: t.Callable) -> dict:
        """Extract function information for tool schema."""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
                
            param_type = self._get_json_type(param.annotation)
            param_info = {"type": param_type}
            
            # Add description if available in docstring
            description = self._extract_param_description(docstring, param_name)
            if description:
                param_info["description"] = description
            
            # Handle default values
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = param.default
            else:
                required.append(param_name)
                
            properties[param_name] = param_info
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": docstring,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _extract_param_description(self, docstring: str, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring or "Args:" not in docstring:
            return ""
            
        lines = docstring.split('\n')
        args_index = -1
        for i, line in enumerate(lines):
            if line.strip().lower() == "args:":
                args_index = i
                break
                
        if args_index == -1:
            return ""
            
        for line in lines[args_index + 1:]:
            line = line.strip()
            if not line:
                continue
            if line.startswith(param_name + " (") or line.startswith(param_name + ":"):
                # Extract description after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                break
                
        return ""

    def _build_tools(self, tools: t.Optional[t.Iterable[t.Callable]]) -> list[dict]:
        """Build tool schemas from callable functions."""
        if not tools:
            return []
        
        tool_schemas = []
        for func in tools:
            try:
                tool_schemas.append(self._extract_function_info(func))
            except Exception as e:
                print(f"Warning: Failed to build tool schema for {func.__name__}: {e}")
                continue
                
        return tool_schemas

    def _extract_json_schema(self, schema_input: t.Any) -> dict | None:
        """Extract JSON schema from various input types."""
        if schema_input is None:
            return None
            
        if isinstance(schema_input, dict):
            return schema_input
        
        # Try Pydantic v2 first, then v1
        for method_name in ("model_json_schema", "schema"):
            method = getattr(schema_input, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    continue
                    
        return None

    def _build_chat_payload(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        stream: bool = False,
        max_tokens: t.Optional[int] = None,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        top_k: t.Optional[int] = None,
        frequency_penalty: t.Optional[float] = None,
        presence_penalty: t.Optional[float] = None,
        repetition_penalty: t.Optional[float] = None,
        min_p: t.Optional[float] = None,
        top_a: t.Optional[float] = None,
        seed: t.Optional[int] = None,
        parallel_tool_calls: bool = True,
        tool_choice: t.Optional[str | dict] = None,
        stop: t.Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: t.Optional[int] = None
    ) -> dict:
        """Build the payload for OpenRouter chat API."""
        # Use provided model or fall back to default
        model = model_name or self.default_model
        
        # Build messages with system instructions and conversation history
        messages = []
        
        # Add system instructions if provided
        if self.system_instructions:
            messages.append({"role": "system", "content": self.system_instructions})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user query
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if min_p is not None:
            payload["min_p"] = min_p
        if top_a is not None:
            payload["top_a"] = top_a
        if seed is not None:
            payload["seed"] = seed
        if stop is not None:
            payload["stop"] = stop
        if logprobs:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
            
        # Add JSON schema if provided (OpenRouter supports structured outputs)
        schema = self._extract_json_schema(json_schema)
        if schema:
            # Use OpenRouter's structured outputs format
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",  # Required name for the schema
                    "strict": True,      # Enable strict mode for exact schema adherence
                    "schema": schema
                }
            }
        
        # Add tools if provided  
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["parallel_tool_calls"] = parallel_tool_calls
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
            
        return payload

    def invoke(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        response_format: t.Optional[dict] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        max_tokens: t.Optional[int] = None,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        top_k: t.Optional[int] = None,
        frequency_penalty: t.Optional[float] = None,
        presence_penalty: t.Optional[float] = None,
        repetition_penalty: t.Optional[float] = None,
        min_p: t.Optional[float] = None,
        top_a: t.Optional[float] = None,
        seed: t.Optional[int] = None,
        parallel_tool_calls: bool = True,
        tool_choice: t.Optional[str | dict] = None,
        stop: t.Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: t.Optional[int] = None
    ) -> dict:
        """Send a query to OpenRouter and return the response.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses (legacy parameter)
            response_format: Optional response format specification (OpenRouter format)
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter (0 or above)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            repetition_penalty: Repetition penalty (0.0 to 2.0)
            min_p: Minimum probability threshold (0.0 to 1.0)
            top_a: Top-a sampling parameter (0.0 to 1.0)
            seed: Deterministic sampling seed
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Controls which tool is called ('none', 'auto', 'required', or specific tool)
            stop: Stop sequences for generation
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top log probabilities to return (0-20)
        """
        # Handle both json_schema (legacy) and response_format parameters
        format_spec = response_format or json_schema
        
        url = f"{self.base_url}/chat/completions"
        payload = self._build_chat_payload(
            query, format_spec, tools, model_name, stream=False,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty,
            min_p=min_p, top_a=top_a, seed=seed,
            parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice,
            stop=stop, logprobs=logprobs, top_logprobs=top_logprobs
        )
        self.query = query
        
        response = requests.post(url, json=payload, headers=self._get_headers(), timeout=300)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract response from OpenRouter format
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No response choices received from OpenRouter")
            
        choice = choices[0]
        message = choice.get("message", {})
        assistant_response = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Execute tool calls if they exist
        tool_execution_results = {}
        tool_result_messages = []
        
        if tool_calls and tools:
            # Use the improved tool execution method
            tool_execution_results, tool_result_messages = self._execute_tool_calls_with_messages(tool_calls, tools)
            # Store tool call results in instance variable
            self.tool_call_results.update(tool_execution_results)
        
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Add assistant response with tool calls (if any) to conversation history
        assistant_message = {
            "role": "assistant",
            "content": assistant_response
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.conversation_history.append(assistant_message)
        
        # Add tool result messages to conversation history
        if tool_result_messages:
            self.add_tool_results_to_conversation(tool_result_messages)
        
        # Build agent context with actual tool execution results
        self._build_agent_context(assistant_response, tool_execution_results)
        
        return {
            "text": assistant_response,
            "raw": data,
            "conversation_history": self.get_conversation_history(),
            "tool_calls": tool_calls,
            "tool_results": tool_execution_results,
            "tool_result_messages": tool_result_messages,  # New: properly formatted tool messages
            "usage": data.get("usage", {}),
            "model": data.get("model", model_name),
            "provider": data.get("provider", ""),
            "finish_reason": choice.get("finish_reason", "")  # New: finish reason for agentic loops
        }

    def invoke_streaming(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        response_format: t.Optional[dict] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        max_tokens: t.Optional[int] = None,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        top_k: t.Optional[int] = None,
        frequency_penalty: t.Optional[float] = None,
        presence_penalty: t.Optional[float] = None,
        repetition_penalty: t.Optional[float] = None,
        min_p: t.Optional[float] = None,
        top_a: t.Optional[float] = None,
        seed: t.Optional[int] = None,
        parallel_tool_calls: bool = True,
        tool_choice: t.Optional[str | dict] = None,
        stop: t.Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: t.Optional[int] = None
    ) -> t.Iterator[str]:
        """Send a query to OpenRouter and return a streaming response.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses (legacy parameter)
            response_format: Optional response format specification (OpenRouter format)
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter (0 or above)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            repetition_penalty: Repetition penalty (0.0 to 2.0)
            min_p: Minimum probability threshold (0.0 to 1.0)
            top_a: Top-a sampling parameter (0.0 to 1.0)
            seed: Deterministic sampling seed
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Controls which tool is called ('none', 'auto', 'required', or specific tool)
            stop: Stop sequences for generation
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top log probabilities to return (0-20)
        """
        # Handle both json_schema (legacy) and response_format parameters
        format_spec = response_format or json_schema
        
        url = f"{self.base_url}/chat/completions"
        payload = self._build_chat_payload(
            query, format_spec, tools, model_name, stream=True,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty,
            min_p=min_p, top_a=top_a, seed=seed,
            parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice,
            stop=stop, logprobs=logprobs, top_logprobs=top_logprobs
        )
        
        # Add user query to conversation history at the start
        self.conversation_history.append({"role": "user", "content": query})
        
        full_response = ""
        tool_calls = []
        last_data = None
        
        with requests.post(url, json=payload, headers=self._get_headers(), stream=True, timeout=300) as response:
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                # Skip comment lines in SSE
                if line.startswith(":"):
                    continue
                    
                # Remove "data: " prefix from SSE
                if line.startswith("data: "):
                    line = line[6:]
                    
                # Check for end of stream
                if line.strip() == "[DONE]":
                    break
                    
                try:
                    data = json.loads(line)
                    choices = data.get("choices", [])
                    
                    if not choices:
                        continue
                        
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    
                    if content:
                        full_response += content
                        yield content
                        
                    # Check for tool calls in delta
                    if "tool_calls" in delta:
                        tool_calls.extend(delta["tool_calls"])
                        
                    # Save the last complete data for final processing
                    if choice.get("finish_reason"):
                        last_data = data
                        
                except json.JSONDecodeError:
                    continue
        
        # Execute tool calls if they exist
        tool_execution_results = {}
        tool_result_messages = []
        
        if tool_calls and tools:
            # Use the improved tool execution method
            tool_execution_results, tool_result_messages = self._execute_tool_calls_with_messages(tool_calls, tools)
            # Store tool call results in instance variable
            self.tool_call_results.update(tool_execution_results)
        
        # Add the complete assistant response with tool calls (if any) to conversation history
        assistant_message = {
            "role": "assistant",
            "content": full_response
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.conversation_history.append(assistant_message)
        
        # Add tool result messages to conversation history
        if tool_result_messages:
            self.add_tool_results_to_conversation(tool_result_messages)
        
        # Build agent context with actual tool execution results
        self._build_agent_context(full_response, tool_execution_results)

    def invoke_with_context(
        self,
        query: str,
        context_from_other_agents: str = "",
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
        max_tokens: t.Optional[int] = None,
        temperature: t.Optional[float] = None,
        top_p: t.Optional[float] = None,
        top_k: t.Optional[int] = None,
        frequency_penalty: t.Optional[float] = None,
        presence_penalty: t.Optional[float] = None,
        repetition_penalty: t.Optional[float] = None,
        min_p: t.Optional[float] = None,
        top_a: t.Optional[float] = None,
        seed: t.Optional[int] = None,
        parallel_tool_calls: bool = True,
        tool_choice: t.Optional[str | dict] = None,
        stop: t.Optional[list[str]] = None,
        logprobs: bool = False,
        top_logprobs: t.Optional[int] = None
    ) -> dict:
        """Send a query with context from other agents and return the response.
        
        Args:
            query: The user's query
            context_from_other_agents: Context information from other agents
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            top_k: Top-k sampling parameter (0 or above)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            repetition_penalty: Repetition penalty (0.0 to 2.0)
            min_p: Minimum probability threshold (0.0 to 1.0)
            top_a: Top-a sampling parameter (0.0 to 1.0)
            seed: Deterministic sampling seed
            parallel_tool_calls: Whether to allow parallel tool calls
            tool_choice: Controls which tool is called ('none', 'auto', 'required', or specific tool)
            stop: Stop sequences for generation
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top log probabilities to return (0-20)
        """
        # Add context from other agents if provided
        if context_from_other_agents.strip():
            self.add_context_from_other_agents(context_from_other_agents)
        
        # Use regular invoke method which now includes conversation history
        return self.invoke(
            query, json_schema, tools, model_name,
            max_tokens=max_tokens, temperature=temperature, 
            top_p=top_p, top_k=top_k, frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty,
            min_p=min_p, top_a=top_a, seed=seed,
            parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice,
            stop=stop, logprobs=logprobs, top_logprobs=top_logprobs
        )


# Backward compatibility alias for existing code
OpenRouterClient = Client


if __name__ == "__main__":
    """Example usage demonstrating conversation history and agent context features."""
    from pydantic import BaseModel
    import os

    # Set environment variable for testing (you should set this in your environment)
    # os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

    class Answer(BaseModel):
        """Schema for structured responses using OpenRouter's structured outputs."""
        title: str
        points: list[str]

    class WeatherInfo(BaseModel):
        """Weather information schema matching OpenRouter documentation example."""
        location: str
        temperature: float
        conditions: str

    def get_weather(city: str, unit: str = "C") -> str:
        """Get weather information for a city.

        Args:
            city (str): The name of the city.
            unit (str): Temperature unit ('C' for Celsius, 'F' for Fahrenheit).
        """
        return f"{city}: 24°{unit}, sunny"

    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable to run examples")
        exit(1)

    # Create an agent with system instructions, default model, and agent name
    agent = Client(
        role="Python programming consultant",
        model_name="openai/gpt-oss-120b:free",
        system_instructions="You are a helpful programming assistant. Always be concise and accurate.",
        agent_name="PythonExpert",
        site_url="https://example.com",  # Optional for OpenRouter rankings
        app_name="Python Assistant"      # Optional for OpenRouter rankings
    )
    
    # Set up agent properties
    agent.set_generic_agent_instructions("Provide accurate Python programming advice")
    agent.set_all_tool_names("get_weather")

    # Example 1: List available models
    print("=== Available Models ===")
    models = agent.list_available_models()
    print(f"Found {len(models)} models")
    if models:
        for model in models[:3]:  # Show first 3 models
            print(f"- {model.get('id', 'Unknown')}: {model.get('name', 'No name')}")

    # Example 2: First conversation
    print("\n=== Conversation Example ===")
    result1 = agent.invoke("What is Python?")
    print(f"Q1: What is Python?")
    print(f"A1: {result1['text'][:100]}...")
    print(f"Model used: {result1.get('model', 'Unknown')}")
    print(f"Provider: {result1.get('provider', 'Unknown')}")
    
    # Update agent context with response
    agent.update_agent_context(result1['text'][:100], {"get_weather": "not_used"})
    
    # Example 3: Follow-up question (uses conversation history)
    result2 = agent.invoke("What are its main advantages?")
    print(f"\nQ2: What are its main advantages?")
    print(f"A2: {result2['text'][:100]}...")
    
    # Example 4: Using context from other agents
    print(f"\n=== Context from Other Agents Example ===")
    context = "Agent A analyzed that the user is working on a web development project using Django."
    result3 = agent.invoke_with_context(
        "Should I use Python for this project?",
        context_from_other_agents=context
    )
    print(f"Q3: Should I use Python for this project?")
    print(f"Context: {context}")
    print(f"A3: {result3['text'][:100]}...")
    
    # Example 5: Show agent context
    print(f"\n=== Agent Context (XML Format) ===")
    agent.update_agent_context(result3['text'][:100], {"get_weather": "24°C, sunny"})
    print(agent.get_agent_context())
    
    # Example 6: Show conversation history
    print(f"\n=== Conversation History ===")
    history = agent.get_conversation_history()
    print(f"Total messages in history: {len(history)}")
    for i, msg in enumerate(history[-4:]):  # Show last 4 messages
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"{i+1}. {role}: {content}")
    
    # Example 7: Tool call functionality
    print(f"\n=== Tool Call Functionality Example ===")
    print("Testing with a model that supports tool calls...")
    
    # Use a model known to support tool calls
    tool_result = agent.invoke(
        "What's the weather like in Paris?", 
        tools=[get_weather],
        model_name="openai/gpt-oss-120b:free"  # Use the specified model for tool calling
    )
    print(f"Q: What's the weather like in Paris?")
    print(f"A: {tool_result['text'][:100]}...")
    print(f"Tool calls made: {len(tool_result.get('tool_calls', []))}")
    print(f"Tool results: {tool_result.get('tool_results', {})}")
    print(f"Stored tool call results: {agent.get_tool_call_results()}")
    
    # Example 8: Agent properties
    print(f"\n=== Agent Properties ===")
    print(f"Agent Name: {agent.agent_name}")
    print(f"Role: {agent.get_role()}")
    print(f"Generic Instructions: {agent.get_generic_agent_instructions()}")
    print(f"Available Tools: {agent.get_all_tool_names()}")
    print(f"Default Model: {agent.get_default_model()}")
    
    # Example 9: Structured Outputs Example
    print(f"\n=== Structured Outputs Example ===")
    try:
        structured_result = agent.invoke(
            "What's the weather like in London?",
            json_schema=WeatherInfo,  # Use Pydantic model for schema
            temperature=0.3
        )
        print(f"Q: What's the weather like in London? (with structured output)")
        print(f"A: {structured_result['text']}")
        print("✅ Structured output format enforced by OpenRouter")
    except Exception as e:
        print(f"Note: Structured outputs require compatible models. Error: {e}")
    
    # Example 10: Clear history and start fresh
    print(f"\n=== Starting Fresh ===")
    agent.clear_conversation_history()
    result4 = agent.invoke("Hello, I'm new here!")
    print(f"After clearing history - Q: Hello, I'm new here!")
    print(f"A: {result4['text'][:100]}...")
    print(f"New history length: {len(agent.get_conversation_history())}")
    
    # Example 11: Using different parameters
    print(f"\n=== Advanced Parameters Example ===")
    result5 = agent.invoke(
        "Explain machine learning in one sentence",
        temperature=0.1,  # More deterministic
        max_tokens=50     # Limit response length
    )
    print(f"Q: Explain machine learning in one sentence (temperature=0.1, max_tokens=50)")
    print(f"A: {result5['text']}")
    print(f"Usage: {result5.get('usage', {})}")
