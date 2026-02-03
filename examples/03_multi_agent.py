"""
Multi-agent example: Student Academic Report System

The idea here is simple — we have three agents that work together in a chain
to produce an academic report for a student.

Here's how it flows:
    User asks for a report
        -> Triage Agent (figures out what to do, hands off)
            -> Results Agent (looks up grades for each subject)
                -> Retest Agent (checks retest dates, timestamps the report, gives final output)

Each handoff passes the full conversation history forward, so the next agent
can see everything the previous one did.

We also have a few tools that simulate a database:
    - get_student_result: fetches a student's grade for a given subject
    - get_retest_dates: checks if a student has any upcoming retests
    - get_current_time: just returns the current timestamp for the report
"""

import asyncio
import datetime

from src.agent import Agent
from src.run import Runner
from src.tool import Tool

# This is our fake database - just dictionaries pretending to be a real DB.
RESULTS_DB = {
    "rahul": {"math": "A", "physics": "B", "chemistry": "C"},
    "priya": {"math": "B", "physics": "A", "chemistry": "A"},
    "amit": {"math": "F", "physics": "D", "chemistry": "F"},
}

RETEST_DB = {
    "amit": ["2025-07-10 (math)", "2025-07-15 (chemistry)"],
    "rahul": ["2025-07-12 (chemistry)"],
}


# Tools - these are the functions our agents can call.


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_student_result(name: str, class_name: str):
    name = name.lower()
    class_name = class_name.lower()
    student = RESULTS_DB.get(name)
    if not student:
        return {"error": f"No records found for student '{name}'"}
    grade = student.get(class_name)
    if not grade:
        return {"error": f"No result found for '{name}' in '{class_name}'"}
    return {"name": name, "class": class_name, "grade": grade}


def get_retest_dates(name: str):
    name = name.lower()
    dates = RETEST_DB.get(name)
    if not dates:
        return {"name": name, "retests": []}
    return {"name": name, "retests": dates}


time_tool = Tool(
    tool_name="get_current_time",
    tool_desc="Get the current date and time",
    tool_fn=get_current_time,
    json_schema={
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current date and time in YYYY-MM-DD HH:MM:SS format.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)

result_tool = Tool(
    tool_name="get_student_result",
    tool_desc="Look up a student's grade for a specific class/subject",
    tool_fn=get_student_result,
    json_schema={
        "type": "function",
        "name": "get_student_result",
        "description": "Fetch a student's grade for a given subject. Returns the grade (A/B/C/D/F).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The student's name",
                },
                "class_name": {
                    "type": "string",
                    "description": "The subject/class name (e.g. math, physics, chemistry)",
                },
            },
            "required": ["name", "class_name"],
        },
    },
)

retest_tool = Tool(
    tool_name="get_retest_dates",
    tool_desc="Get the retest dates for a student",
    tool_fn=get_retest_dates,
    json_schema={
        "type": "function",
        "name": "get_retest_dates",
        "description": "Fetch the list of upcoming retest dates for a student.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The student's name",
                },
            },
            "required": ["name"],
        },
    },
)


retest_agent = Agent(
    name="RetestAgent",
    model="gpt-4o-mini",
    desc="Agent that checks retest schedules and generates the final report",
    instructions="""You are the Retest Agent. You receive a conversation that already contains
a student's grades (fetched by the Results Agent). Your job is to:
1. Use the get_retest_dates tool to check if the student has any upcoming retests.
2. Use the get_current_time tool to timestamp the report.
3. Compile everything into a clean final academic report that includes:
   grades per subject, retest dates (if any), and the report generation timestamp.""",
    tools=[retest_tool, time_tool],
)

results_agent = Agent(
    name="ResultsAgent",
    model="gpt-4o-mini",
    desc="Agent that fetches student grades across all subjects",
    instructions="""You are the Results Agent. Your job is to look up a student's grades
across all subjects using the get_student_result tool.
Call the tool once for each subject: math, physics, and chemistry.
Once you have all the grades, hand off to the RetestAgent so it can
check for retests and compile the final report.""",
    tools=[result_tool],
    handoffs=[retest_agent],
)

triage_agent = Agent(
    name="TriageAgent",
    model="gpt-4o-mini",
    desc="Entry-point agent that routes requests to the right agent",
    instructions="""You are the Triage Agent. You receive user queries about students.
Your only job is to hand off to the ResultsAgent so it can start
fetching the student's academic data. Do not answer directly.""",
    handoffs=[results_agent],
)


async def main():
    response = await Runner.run(
        agent=triage_agent,
        user_input="Give me a complete academic report for Rahul",
        max_turns=15,
        display_logs=True,
    )

    print(response)


asyncio.run(main())
