import asyncio
import datetime

from src.agent import Agent
from src.run import Runner
from src.tool import Tool


def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")


time_tool = Tool(
    tool_name="get_time",
    tool_desc="Get current time in HH:MM:SS format",
    tool_fn=get_time,
    json_schema={
        "type": "function",
        "name": "get_time",
        "description": "Get the current server time in HH:MM:SS format.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    desc="General purpose agent",
    instructions="You are a helpful, concise assistant that follows instructions carefully and uses tools when appropriate.",
    tools=[time_tool],
)


async def main():
    response = await Runner.run(
        agent=agent, user_input="What might be the time right now?"
    )
    print(response)


asyncio.run(main())
