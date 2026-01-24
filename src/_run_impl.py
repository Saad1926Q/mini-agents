import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import RateLimitError
from openai.types.responses import Response

from src.agent import Agent
from src.clients.openai import client
from src.tool import execute_function

MAX_RETRIES = 5


@dataclass
class NextStepFinalOutput:
    output: Any


@dataclass
class NextStepRunAgain:
    output: Any


@dataclass
class NextStepHandoff:
    output: Any
    handoff_agent: Agent


def is_handoff(handoffs: List[Agent], fn_name: str) -> bool:
    handoff_agent_name = fn_name.split("transfer_to_", 1)[1]

    return handoff_agent_name is not None and handoff_agent_name in [
        handoff.name for handoff in handoffs
    ]


def _call_openai_with_retry(
    agent: Agent,
    tool_schemas: List[Dict[str, Any]],
    model_input: List[Dict[str, Any]],
) -> Response:
    attempts = 0
    delay = 1

    while True:
        attempts += 1
        try:
            response = client.responses.create(
                model=agent.model,
                tools=tool_schemas,
                input=model_input,
            )
            return response
        except RateLimitError as e:
            if attempts > MAX_RETRIES:
                raise Exception(f"Maximum number of retries ({MAX_RETRIES}) exceeded.")

            time.sleep(delay)
            delay *= 2
        except Exception as e:
            raise


async def execute_single_step(
    agent: Agent,
    tool_schemas: List[Dict[str, Any]],
    model_input: List[Dict[str, Any]],
    handoffs: List[Agent],
) -> NextStepFinalOutput | NextStepRunAgain | NextStepHandoff:
    response = _call_openai_with_retry(agent, tool_schemas, model_input)

    outputs = list()

    outputs.extend(response.output)

    called_tools = False

    for output in response.output:
        if output.type != "function_call":
            continue

        fn_name = output.name
        try:
            args = json.loads(output.arguments)
        except Exception as e:
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": json.dumps(
                        {
                            "error": f"Error during deserialization of tool call arguments: {e}"
                        }
                    ),
                }
            )
            called_tools = True
            continue  # skip to next tool call

        if fn_name.startswith("transfer_to_") and is_handoff(handoffs, fn_name):
            handoff_agent = None

            for h in handoffs:
                if h.name == fn_name.split("transfer_to_", 1)[1]:
                    handoff_agent = h
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": output.call_id,
                            "output": f"Control transferred to {h.name}",
                        }
                    )
                    break

            return NextStepHandoff(outputs, handoff_agent)

        for fn in agent.tools:
            if fn.json_schema.get("name") == fn_name:
                try:
                    result = await execute_function(fn.tool_fn, args)
                    result_json = json.dumps(result)

                except Exception as e:
                    result_json = json.dumps(
                        {"error": f"Tool '{fn_name}' failed: {str(e)}"}
                    )

                called_tools = True
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": result_json,
                    }
                )
                break

    if not called_tools:
        return NextStepFinalOutput(outputs)

    return NextStepRunAgain(outputs)
