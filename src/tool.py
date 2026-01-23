import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    tool_name: str
    tool_desc: str
    tool_fn: Callable[..., Any]
    json_schema: Dict[str, Any]


async def execute_function(
    func: Callable[..., Any], func_params: Dict[str, Any]
) -> Any:
    if inspect.iscoroutine(func):
        return await func(**func_params)

    return func(**func_params)


def to_json_schema(callable: Callable) -> Dict[str, Any]:
    positional_arguments: List[Any] = []
    keyword_arguments: Dict[str, Any] = {}

    sig: inspect.Signature = inspect.signature(callable)

    for name, param in sig.parameters.items():
        print(param.annotation)

    # TODO: Implement a functionality to take a fn and automatically create a json schema

    return {}
