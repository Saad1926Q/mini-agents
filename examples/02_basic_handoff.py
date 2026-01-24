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

time_agent = Agent(
    name="TimeKeeper",
    model="gpt-4o-mini",
    desc="Agent that can check the current time",
    instructions="You are the TimeKeeper. You have access to the get_time tool. Use it to answer time-related questions.",
    tools=[time_tool],
)

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    desc="General purpose agent",
    instructions="You are a helpful assistant. If the user asks about the current time, hand off to the TimeKeeper agent.",
    handoffs=[time_agent],
)


async def main():
    response = await Runner.run(
        agent=agent,
        user_input="What time is it right now?",
        display_logs=True,
    )
    print(response)


asyncio.run(main())
