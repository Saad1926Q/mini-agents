import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    tool_name: str
    tool_desc: str
    tool_fn: Callable[..., Any] | None = None
    json_schema: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.json_schema == {}:
            self.json_schema["name"] = self.tool_name
            self.json_schema["parameters"] = {}
            self.json_schema["type"] = "function"
            self.json_schema["description"] = self.tool_desc


async def execute_function(
    func: Callable[..., Any], func_params: Dict[str, Any]
) -> Any:
    if inspect.iscoroutinefunction(func):
        return await func(**func_params)

    return func(**func_params)
