import json

from src.agent import Agent
from src.clients.openai import client
from src.tool import execute_function

DEFAULT_MAX_TURNS = 10


class Runner:
    async def run(self, agent: Agent, user_input: str, *, max_turns: int | None = None):
        max_turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS

        tool_schemas = [tool.json_schema for tool in agent.tools]

        model_input = [
            {"role": "system", "content": agent.instructions},
            {"role": "user", "content": user_input},
        ]

        curr_turn = 0

        while curr_turn < max_turns:
            curr_turn += 1

            response = client.responses.create(
                model=agent.model,
                tools=tool_schemas,
                input=model_input,
            )

            model_input.extend(response.output)

            called_tools = False

            for output in response.output:
                if output.type != "function_call":
                    continue

                print(output)

                fn_name = output.name
                args = json.loads(output.arguments)

                for fn in agent.tools:
                    if fn.json_schema.get("name") == fn_name:
                        result = await execute_function(fn.tool_fn, args)
                        result_json = json.dumps(result)
                        model_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": output.call_id,
                                "output": result_json,
                            }
                        )
                        called_tools = True
                        break

            if called_tools is False:
                return response.output_text
