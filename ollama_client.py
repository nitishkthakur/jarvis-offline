import inspect
import json
import typing as t
import requests
import subprocess
import time
import shutil
GENERIC_AGENT_INSTRUCTIONS = """You are a helpful Agent among a group of agents trying to solve a problem. Each agent is tasked with a part or the entire problem.
You will be given your task. You will have access to all the relevant information and tools. You can also see the work already done by other agents. Use that information if required.


## Instructions
1. View the context, the task executed, the results, the tool call results of the other agents.
2. Reason and execute your task.
3. You have access to multiple tools as listed. You can only call tools which are relevant to your task. Other agents might have executed other tools which you dont have access to.
4. Always output strictly JSON. Always.
5. your task will be enclosed in <YOUR TASK></YOUR TASK> tags. This is your task. Only execute this task.
6. The work done by other agents will be enclosed in <Agent: Agent Name></Agent: Agent Name> tags. There may be multiple of these.


Following is the relevant information from other agents (if any):
{other_agents_history}


<YOUR TASK>
{task}
</YOUR TASK>


"""

class OllamaClient:
    """A simplified agent client for interacting with Ollama API."""
    
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
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        system_instructions: str = "",
        agent_name: str = ""
    ):
        """Initialize the Ollama agent client.
        
        Args:
            base_url: URL of the Ollama server
            model_name: Default model to use for all interactions
            system_instructions: System instructions to guide the agent's behavior
            agent_name: Name/identifier for this agent
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = model_name
        #self.system_instructions = system_instructions
        self.agent_name = agent_name
        self.conversation_history: list[dict] = []  # Store conversation history
        
        # New agent context variables
        self.role = role
        self.history_from_other_agents = history_from_other_agents
        self.all_tool_names = ""
        self.only_this_agent_context = ""

        # System instructions is the same as the generic agent instruuctions  -  fix that to remove redundancy
        self.generic_agent_instructions = GENERIC_AGENT_INSTRUCTIONS.format(task=self.role, other_agents_history=self.history_from_other_agents)
        self.system_instructions = self.generic_agent_instructions
        
        


        
        self._ensure_server_ready()
        self._set_model_keepalive()

    def _is_server_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _start_server(self) -> None:
        """Start Ollama server if not running."""
        if not shutil.which("ollama"):
            raise RuntimeError("Ollama binary not found. Please install Ollama.")
        
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    def _wait_for_server(self, timeout: float = 30.0) -> bool:
        """Wait for server to start up."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_server_running():
                return True
            time.sleep(0.5)
        return False

    def _ensure_server_ready(self) -> None:
        """Ensure Ollama server is running."""
        if not self._is_server_running():
            self._start_server()
            if not self._wait_for_server():
                raise RuntimeError("Failed to start Ollama server.")

    def _set_model_keepalive(self) -> None:
        """Set the model keepalive to 15 minutes."""
        try:
            # Load the model with keepalive setting
            payload = {
                "model": self.default_model,
                "keep_alive": "15m"
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=10)
            # Don't raise for status here as this is just a keepalive setting
        except Exception:
            # If setting keepalive fails, continue anyway
            pass

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
        self._set_model_keepalive()  # Set keepalive for new model

    def get_conversation_history(self) -> list[dict]:
        """Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

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
            Task: This is the {self.role}
            Agent Response: {agent_response}
            {tool_section}
            </Agent: {self.agent_name}>"""

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
    ) -> dict:
        """Build the payload for Ollama chat API."""
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
            "stream": stream,
            "keep_alive": "15m"  # Set keepalive for this request
        }
        
        # Add JSON schema if provided
        schema = self._extract_json_schema(json_schema)
        if schema:
            payload["format"] = schema
        
        # Add tools if provided  
        tool_schemas = self._build_tools(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            
        return payload

    def invoke(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
    ) -> dict:
        """Send a query to Ollama and return the response.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
        """
        url = f"{self.base_url}/api/chat"
        payload = self._build_chat_payload(query, json_schema, tools, model_name, stream=False)
        
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        data = response.json()
        message = data.get("message", {})
        assistant_response = message.get("content", "")
        
        # Add user query and assistant response to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        return {
            "text": assistant_response,
            "raw": data,
            "conversation_history": self.get_conversation_history()
        }

    def invoke_streaming(
        self,
        query: str,
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
    ) -> t.Iterator[str]:
        """Send a query to Ollama and return a streaming response.
        
        Args:
            query: The user's query
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
        """
        url = f"{self.base_url}/api/chat"
        payload = self._build_chat_payload(query, json_schema, tools, model_name, stream=True)
        
        # Add user query to conversation history at the start
        self.conversation_history.append({"role": "user", "content": query})
        
        full_response = ""
        
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    content = message.get("content")
                    
                    if content:
                        full_response += content
                        yield content
                        
                    if data.get("done"):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        # Add the complete assistant response to conversation history
        self.conversation_history.append({"role": "assistant", "content": full_response})

    def invoke_with_context(
        self,
        query: str,
        context_from_other_agents: str = "",
        json_schema: t.Optional[dict | t.Any] = None,
        tools: t.Optional[t.Iterable[t.Callable]] = None,
        model_name: t.Optional[str] = None,
    ) -> dict:
        """Send a query with context from other agents and return the response.
        
        Args:
            query: The user's query
            context_from_other_agents: Context information from other agents
            json_schema: Optional JSON schema for structured responses
            tools: Optional list of functions the agent can call
            model_name: Optional model override (uses default if not provided)
        """
        # Add context from other agents if provided
        if context_from_other_agents.strip():
            self.add_context_from_other_agents(context_from_other_agents)
        
        # Use regular invoke method which now includes conversation history
        return self.invoke(query, json_schema, tools, model_name)


if __name__ == "__main__":
    """Example usage demonstrating conversation history and agent context features."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        """Schema for structured responses."""
        title: str
        points: list[str]

    def get_weather(city: str, unit: str = "C") -> str:
        """Get weather information for a city.

        Args:
            city (str): The name of the city.
            unit (str): Temperature unit ('C' for Celsius, 'F' for Fahrenheit).
        """
        return f"{city}: 24°{unit}, sunny"

    # Create an agent with system instructions, default model, and agent name
    agent = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a helpful programming assistant. Always be concise and accurate.",
        agent_name="PythonExpert"
    )
    
    # Set up agent properties
    agent.set_role("Python programming consultant")
    agent.set_generic_agent_instructions("Provide accurate Python programming advice")
    agent.set_all_tool_names("get_weather")

    # Example 1: First conversation
    print("=== Conversation Example ===")
    result1 = agent.invoke("What is Python?")
    print(f"Q1: What is Python?")
    print(f"A1: {result1['text'][:100]}...")
    
    # Update agent context with response
    agent.update_agent_context(result1['text'][:100], {"get_weather": "not_used"})
    
    # Example 2: Follow-up question (uses conversation history)
    result2 = agent.invoke("What are its main advantages?")
    print(f"\nQ2: What are its main advantages?")
    print(f"A2: {result2['text'][:100]}...")
    
    # Example 3: Using context from other agents
    print(f"\n=== Context from Other Agents Example ===")
    context = "Agent A analyzed that the user is working on a web development project using Django."
    result3 = agent.invoke_with_context(
        "Should I use Python for this project?",
        context_from_other_agents=context
    )
    print(f"Q3: Should I use Python for this project?")
    print(f"Context: {context}")
    print(f"A3: {result3['text'][:100]}...")
    
    # Example 4: Show agent context
    print(f"\n=== Agent Context (XML Format) ===")
    agent.update_agent_context(result3['text'][:100], {"get_weather": "24°C, sunny"})
    print(agent.get_agent_context())
    
    # Example 5: Show conversation history
    print(f"\n=== Conversation History ===")
    history = agent.get_conversation_history()
    print(f"Total messages in history: {len(history)}")
    for i, msg in enumerate(history[-4:]):  # Show last 4 messages
        role = msg['role'].upper()
        content = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
        print(f"{i+1}. {role}: {content}")
    
    # Example 6: Agent properties
    print(f"\n=== Agent Properties ===")
    print(f"Agent Name: {agent.agent_name}")
    print(f"Role: {agent.get_role()}")
    print(f"Generic Instructions: {agent.get_generic_agent_instructions()}")
    print(f"Available Tools: {agent.get_all_tool_names()}")
    
    # Example 7: Clear history and start fresh
    print(f"\n=== Starting Fresh ===")
    agent.clear_conversation_history()
    result4 = agent.invoke("Hello, I'm new here!")
    print(f"After clearing history - Q: Hello, I'm new here!")
    print(f"A: {result4['text'][:100]}...")
    print(f"New history length: {len(agent.get_conversation_history())}")
