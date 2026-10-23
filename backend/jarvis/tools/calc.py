"""Calculator and unit conversion tools."""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def evaluate(args: dict) -> dict:
    expr = args.get("expression", "")
    try:
        allowed = {"__builtins__": {}, "abs": abs, "round": round, "min": min, "max": max, "pow": pow, "sqrt": math.sqrt, "pi": math.pi, "e": math.e, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor, "ceil": math.ceil}
        result = eval(expr, allowed)
        return {"success": True, "expression": expr, "result": result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def convert_units(args: dict) -> dict:
    value = float(args.get("value", 0))
    from_unit = args.get("from", "")
    to_unit = args.get("to", "")
    conversions = {
        ("km", "mi"): 0.621371,
        ("mi", "km"): 1.60934,
        ("kg", "lb"): 2.20462,
        ("lb", "kg"): 0.453592,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("mb", "gb"): 0.001,
        ("gb", "mb"): 1024,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return {"success": False, "error": f"Unknown conversion: {from_unit} to {to_unit}"}
    factor = conversions[key]
    result = factor(value) if callable(factor) else value * factor
    return {"success": True, "value": value, "from": from_unit, "to": to_unit, "result": result}


def register_calc_tools(registry) -> None:
    from jarvis.tools.registry import Tool
    registry.register(Tool(
        name="calculator",
        description="Evaluate a mathematical expression.",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        handler=evaluate,
        level=PermissionLevel.READ,
    ))
    registry.register(Tool(
        name="unit_convert",
        description="Convert between common units.",
        parameters={"type": "object", "properties": {"value": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}},
        handler=convert_units,
        level=PermissionLevel.READ,
    ))