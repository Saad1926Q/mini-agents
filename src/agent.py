from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.tool import Tool


@dataclass
class Agent:
    name: str
    model: str
    desc: str
    instructions: str
    tools: List[Tool] | None = None
    handoffs: List[Agent] | None = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = list()
