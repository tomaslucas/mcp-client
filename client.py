import asyncio
import os
from typing import Optional
from contextlib import AsyncExitStack
import json
from llm_providers import OllamaProvider, OpenAIProvider, AnthropicProvider

import argparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self, provider_name='ollama'):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize LLM provider
        if provider_name == 'ollama':
            self.provider = OllamaProvider()
        elif provider_name == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.provider = OpenAIProvider(api_key, model='gpt-4o-mini')  # Using GPT-4 for better tool use
        elif provider_name == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            self.provider = AnthropicProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def chat_with_llm(self, messages, tools=None):
        """Send a chat request to the configured LLM provider"""
        return await self.provider.chat(messages, tools)

    async def process_query(self, query: str) -> str:
        """Process a query using the configured LLM provider and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial LLM API call
        response = await self.chat_with_llm(messages, available_tools)
        assistant_message = response["message"]["content"]

        # Simple parsing of tool calls (you may need to adjust based on your model's output)
        tool_results = []
        final_text = [assistant_message]

        # Look for tool invocations in the format: {{tool_name(args)}}
        while '{{' in assistant_message and '}}' in assistant_message:
            start = assistant_message.find('{{')
            end = assistant_message.find('}}')
            if start == -1 or end == -1:
                break

            tool_call = assistant_message[start+2:end].strip()
            try:
                tool_name, args_str = tool_call.split('(', 1)
                args_str = args_str.rstrip(')')
                tool_args = json.loads(args_str)

                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_message[:start]
                })
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool_name} returned: {result.content}"
                })

                response = await self.chat_with_llm(messages, available_tools)
                assistant_message = response["message"]["content"]
                final_text.append(assistant_message)
                
            except Exception as e:
                final_text.append(f"Error processing tool call: {str(e)}")
                break

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description='MCP Client')
    parser.add_argument('server_script', help='Path to the server script (.py or .js)')
    parser.add_argument('--provider', 
                       choices=['ollama', 'openai', 'anthropic'],
                       default='ollama',
                       help='LLM provider to use (default: ollama)')
    args = parser.parse_args()
    
    client = MCPClient(provider_name=args.provider)
    try:
        await client.connect_to_server(args.server_script)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())