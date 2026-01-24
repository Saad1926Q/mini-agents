from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.tool import Tool


@dataclass
class Agent:
    name: str
    model: str
    desc: str
    instructions: str
    tools: List[Tool] = field(default_factory=list)
    handoffs: List[Agent] = field(default_factory=list)

    def __post_init__(self):
        for handoff in self.handoffs:
            self.tools.append(
                Tool(
                    tool_name=f"transfer_to_{handoff.name}",
                    tool_desc=f"Route this request to the {handoff.name} agent. Agent description: {handoff.desc}",
                )
            )
