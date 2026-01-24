from src._run_impl import (
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    execute_single_step,
)
from src.agent import Agent

DEFAULT_MAX_TURNS = 10


class Runner:
    @classmethod
    async def run(
        cls,
        agent: Agent,
        user_input: str,
        *,
        max_turns: int | None = None,
        display_logs: bool = False,
    ):
        current_agent = agent
        max_turns = max_turns if max_turns is not None else DEFAULT_MAX_TURNS

        model_input = [
            {"role": "system", "content": agent.instructions},
            {"role": "user", "content": user_input},
        ]

        for i in range(max_turns):
            if display_logs:
                print(f"[Turn {i + 1}/{max_turns}] Agent: {current_agent.name}")

            tool_schemas = [tool.json_schema for tool in current_agent.tools]

            result = await execute_single_step(
                current_agent, tool_schemas, model_input, current_agent.handoffs
            )

            if isinstance(result, NextStepFinalOutput):
                if display_logs:
                    print(f"[Turn {i + 1}] Final output received")
                return result.output
            elif isinstance(result, NextStepRunAgain):
                if display_logs:
                    print(f"[Turn {i + 1}] Continuing loop")
                model_input.extend(result.output)
            elif isinstance(result, NextStepHandoff):
                if display_logs:
                    print(
                        f"[Turn {i + 1}] Handoff: {current_agent.name} -> {result.handoff_agent.name}"
                    )
                current_agent = result.handoff_agent
                model_input[0]["content"] = (
                    current_agent.instructions
                )  # Replace the system prompt with the handoff agent's system prompt
                model_input.extend(result.output)

        raise Exception(f"Max turns ({max_turns}) exceeded")
