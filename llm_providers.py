import json
from abc import ABC, abstractmethod
import asyncio
import ollama
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list, tools=None) -> dict:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, host='http://localhost:11434', model='llama3.2'):
        self.client = ollama.Client(host=host)
        self.model = model

    def _format_tools_for_prompt(self, tools) -> str:
        if not tools:
            return ""
            
        tool_descriptions = []
        for tool in tools:
            example_params = self._generate_example_params(tool['input_schema'])
            desc = f"""Tool: {tool['name']}
Description: {tool['description']}
Parameters: {json.dumps(tool['input_schema'], indent=2)}
Example: {{{{{tool['name']}({json.dumps(example_params)})}}}}"""
            tool_descriptions.append(desc)
            
        return "\n\n".join(tool_descriptions)

    def _generate_example_params(self, schema):
        """Generate example parameters based on schema"""
        if 'properties' in schema:
            example = {}
            for prop, details in schema['properties'].items():
                if details.get('type') == 'number':
                    example[prop] = 0.0
                elif details.get('type') == 'integer':
                    example[prop] = 0
                else:
                    example[prop] = "example"
            return example
        return {}

    async def chat(self, messages: list, tools=None) -> dict:
        system_message = """You are a helpful AI assistant. To use tools:

1. Use exactly this format: {{tool_name({"param1": "value1"})}}
2. Make sure to use double quotes in JSON
3. No spaces between parentheses and braces
4. One tool call per message

Example: {{get-weather({"city": "London"})}}
"""
        if tools:
            system_message += "\n\nAvailable Tools:\n" + self._format_tools_for_prompt(tools)

        formatted_messages = [{"role": "system", "content": system_message}] + messages
        
        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=formatted_messages
        )
        return response

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model='gpt-4o-mini'):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    def _convert_tools_to_functions(self, tools):
        """Convert MCP tools to OpenAI function definitions"""
        functions = []
        if tools:
            for tool in tools:
                functions.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                })
        return functions

    async def chat(self, messages: list, tools=None) -> dict:
        system_message = "You are a helpful AI assistant. When you need to use a tool, format your response as '{{tool_name(args)}}'."
        formatted_messages = [{"role": "system", "content": system_message}] + messages
        
        functions = self._convert_tools_to_functions(tools)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            functions=functions if functions else None,
            function_call="auto"
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        # If function was called, format it as {{function_name(args)}}
        if message.function_call:
            content += f"\n{{{{{message.function_call.name}({message.function_call.arguments})}}}}"
            
        return {"message": {"content": content}}

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model='claude-3-sonnet-20241022'):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(self, messages: list, tools=None) -> dict:
        system_message = "You are a helpful AI assistant. Use available tools when needed."
        formatted_messages = [{"role": "system", "content": system_message}] + messages
        
        # Add tools to API call if available
        kwargs = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": formatted_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)
        
        # Process response and handle tool calls
        content = []
        for message in response.content:
            if message.type == 'text':
                content.append(message.text)
            elif message.type == 'tool_use':
                # Format tool call in the expected {{tool_name(args)}} syntax
                tool_call = f"{{{{{message.name}({json.dumps(message.input)})}}}}"
                content.append(tool_call)
                if hasattr(message, 'text') and message.text:
                    content.append(message.text)

        return {"message": {"content": "\n".join(content)}}