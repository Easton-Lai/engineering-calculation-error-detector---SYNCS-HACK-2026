from __future__ import annotations

import ast
import keyword
import operator as op
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn, TypeAlias, cast

import pandas as pd
from pint.registry import Quantity, Unit, UnitRegistry
import streamlit as st
from streamlit import runtime as st_runtime


ureg: UnitRegistry[float] = UnitRegistry("")

Scalar: TypeAlias = int | float
CalculationQuantity: TypeAlias = Quantity[Any]
CalculationValue: TypeAlias = Scalar | CalculationQuantity
BinaryOperation: TypeAlias = Callable[[CalculationValue, CalculationValue], object]
UnaryOperation: TypeAlias = Callable[[CalculationValue], object]
QuantityFactory: TypeAlias = Callable[
    [Scalar, str | Unit | None], CalculationQuantity
]


def make_quantity(
    magnitude: Scalar,
    unit: str | Unit | None = None,
) -> CalculationQuantity:
    """Create a quantity bound to this module's unit registry."""
    quantity_factory = cast(QuantityFactory, cast(object, ureg.Quantity))
    return quantity_factory(magnitude, unit)


def parse_unit(unit_text: str) -> Unit:
    """Parse a unit with this module's unit registry."""
    return ureg.parse_units(unit_text)


# -----------------------------
# Helpers
# -----------------------------
def normalize_text(text: str | None) -> str:
    """Normalize user input for expressions and units."""
    if text is None:
        return ""

    normalized = text.strip()
    replacements = {
        "²": "**2",
        "³": "**3",
        "^": "**",
        "×": "*",
        "·": "*",
        "⋅": "*",
        "÷": "/",
        "−": "-",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def format_exponent(value: Any) -> str:
    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)

    if integer_value == value:
        return str(integer_value)
    return str(value)


def format_dimensions(quantity: CalculationQuantity) -> str:
    dims = quantity.dimensionality
    if not dims:
        return "dimensionless"

    symbol_map = {
        "[mass]": "M",
        "[length]": "L",
        "[time]": "T",
        "[current]": "I",
        "[temperature]": "Θ",
        "[substance]": "N",
        "[luminosity]": "J",
    }
    order = [
        "[mass]",
        "[length]",
        "[time]",
        "[current]",
        "[temperature]",
        "[substance]",
        "[luminosity]",
    ]

    parts = []
    for key in order:
        if key in dims:
            exponent = dims[key]
            symbol = symbol_map.get(key, key)
            if exponent == 1:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}^{format_exponent(exponent)}")
    return " ".join(parts)


def compatible_quantity_name(quantity: CalculationQuantity) -> str:
    dims = quantity.dimensionality
    known = {
        frozenset({"[mass]": 1, "[length]": -1, "[time]": -2}.items()): "Pressure / Stress",
        frozenset({"[mass]": 1, "[length]": 2, "[time]": -2}.items()): "Energy / Torque",
        frozenset({"[length]": 1, "[time]": -1}.items()): "Velocity",
        frozenset({"[length]": 1, "[time]": -2}.items()): "Acceleration",
        frozenset({"[mass]": 1, "[length]": 1, "[time]": -1}.items()): "Momentum",
        frozenset({"[mass]": 1, "[length]": 2, "[time]": -3}.items()): "Power",
        frozenset({"[mass]": 1, "[length]": 1, "[time]": -2}.items()): "Force",
        frozenset({"[length]": 3, "[time]": -1}.items()): "Volumetric Flow Rate",
    }
    return known.get(frozenset(dims.items()), "Unknown / Not mapped")


# -----------------------------
# Safe expression evaluator
# -----------------------------
def _as_binary_operation(operation: object) -> BinaryOperation:
    return cast(BinaryOperation, operation)


def _as_unary_operation(operation: object) -> UnaryOperation:
    return cast(UnaryOperation, operation)


ALLOWED_BIN_OPS: dict[type[ast.operator], BinaryOperation] = {
    ast.Add: _as_binary_operation(op.add),
    ast.Sub: _as_binary_operation(op.sub),
    ast.Mult: _as_binary_operation(op.mul),
    ast.Div: _as_binary_operation(op.truediv),
    ast.Pow: _as_binary_operation(op.pow),
}

ALLOWED_UNARY_OPS: dict[type[ast.unaryop], UnaryOperation] = {
    ast.UAdd: _as_unary_operation(op.pos),
    ast.USub: _as_unary_operation(op.neg),
}


def _require_calculation_value(value: object) -> CalculationValue:
    if isinstance(value, bool):
        raise ValueError("Expressions may only contain real numbers and quantities.")

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, Quantity):
        magnitude = value.magnitude
        if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
            raise ValueError("Expressions may only contain real numbers and quantities.")
        return value

    raise ValueError("Expressions may only contain real numbers and quantities.")


def eval_expr(
    expr: str,
    variable_map: Mapping[str, CalculationValue],
) -> CalculationValue:
    normalized_expr = normalize_text(expr)
    tree = ast.parse(normalized_expr, mode="eval")
    return _eval_ast(tree.body, variable_map)


def _eval_ast(
    node: ast.expr,
    variable_map: Mapping[str, CalculationValue],
) -> CalculationValue:
    if isinstance(node, ast.Constant):
        return _require_calculation_value(node.value)

    if isinstance(node, ast.Name):
        if node.id not in variable_map:
            raise ValueError(f"Unknown variable: {node.id}")
        return variable_map[node.id]

    if isinstance(node, ast.BinOp):
        operation = ALLOWED_BIN_OPS.get(type(node.op))
        if operation is None:
            raise ValueError("Operator not allowed.")
        left = _eval_ast(node.left, variable_map)
        right = _eval_ast(node.right, variable_map)
        return _require_calculation_value(operation(left, right))

    if isinstance(node, ast.UnaryOp):
        operation = ALLOWED_UNARY_OPS.get(type(node.op))
        if operation is None:
            raise ValueError("Unary operator not allowed.")
        operand = _eval_ast(node.operand, variable_map)
        return _require_calculation_value(operation(operand))

    raise ValueError("Invalid expression.")


def _is_missing(value: Any) -> bool:
    """Return whether a scalar table cell contains a missing value."""
    return bool(pd.isna(value))


def build_variables(df: pd.DataFrame) -> dict[str, CalculationValue]:
    variable_map: dict[str, CalculationValue] = {}

    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        raw_value = row["Value"]
        if _is_missing(raw_value) or raw_value == "":
            continue

        raw_name = row["Variable"]
        if _is_missing(raw_name) or not str(raw_name).strip():
            raise ValueError(f"Variable name is required in row {row_number}.")

        name = str(raw_name).strip()
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Invalid variable name: {name}")
        if name in variable_map:
            raise ValueError(f"Duplicate variable name: {name}")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for variable: {name}") from None

        raw_unit = row["Unit"]
        unit = "" if _is_missing(raw_unit) else normalize_text(str(raw_unit))
        variable_map[name] = make_quantity(value, unit) if unit else value

    return variable_map


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="UnitGuard", layout="wide")
    st.title("UnitGuard")

    st.subheader("Expression")
    expression = (
        st.text_input(
            "Expression input",
            value="F / A",
            label_visibility="collapsed",
        )
        or ""
    )

    st.write("Then a table:")
    default_df = pd.DataFrame(
        [
            {"Variable": "F", "Value": 10.0, "Unit": "kN"},
            {"Variable": "A", "Value": 200.0, "Unit": "mm^2"},
        ]
    )

    edited_df = cast(
        pd.DataFrame,
        st.data_editor(
            default_df,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Variable": st.column_config.TextColumn("Variable"),
                "Value": st.column_config.NumberColumn("Value"),
                "Unit": st.column_config.TextColumn("Unit"),
            },
        ),
    )

    st.write("Then enter:")
    expected_unit = st.text_input("Expected output unit:", value="MPa") or ""

    if st.button("CHECK CALCULATION"):
        try:
            variable_map = build_variables(edited_df)

            if not expression.strip():
                raise ValueError("Expression cannot be empty.")

            evaluated_result = eval_expr(expression, variable_map)
            result = (
                evaluated_result
                if isinstance(evaluated_result, Quantity)
                else make_quantity(evaluated_result)
            )

            actual_dims = format_dimensions(result)
            actual_kind = compatible_quantity_name(result)
            expected_unit_text = normalize_text(expected_unit)

            if expected_unit_text:
                expected_unit_obj = parse_unit(expected_unit_text)
                expected_q = make_quantity(1, expected_unit_obj)
                is_consistent = result.dimensionality == expected_q.dimensionality

                if is_consistent:
                    converted = result.to(expected_unit_obj)
                    si_result = cast(CalculationQuantity, result.to_base_units())

                    output = (
                        "✓ DIMENSIONALLY CONSISTENT\n\n"
                        f"Result\n{format(converted, '~P')}\n\n"
                        f"SI form\n{format(si_result, '~P')}\n\n"
                        f"Dimensions\n{actual_dims}\n\n"
                        f"Compatible quantity dimension:\n{actual_kind}"
                    )
                else:
                    expected_dims = format_dimensions(expected_q)
                    output = (
                        "✗ DIMENSIONAL MISMATCH\n\n"
                        f"Expression produces\n{actual_dims}\n\n"
                        f"Expected\n{expected_dims}\n\n"
                        f"Compatible quantity dimension:\n{actual_kind}"
                    )
            else:
                si_result = cast(CalculationQuantity, result.to_base_units())
                output = (
                    "✓ CALCULATION COMPLETE\n\n"
                    f"Result\n{format(result, '~P')}\n\n"
                    f"SI form\n{format(si_result, '~P')}\n\n"
                    f"Dimensions\n{actual_dims}\n\n"
                    f"Compatible quantity dimension:\n{actual_kind}"
                )

            st.write("Result:")
            st.code(output)

        except Exception as error:
            st.write("Result:")
            st.code(f"ERROR\n\n{error}")


def _launch_streamlit() -> NoReturn:
    """Relaunch this file through Streamlit when run as a plain Python script."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    raise SystemExit(completed.returncode)


def _run_entrypoint() -> None:
    """Render inside Streamlit, or bootstrap Streamlit for a direct run."""
    if st_runtime.exists():
        main()
    else:
        _launch_streamlit()


if __name__ == "__main__":
    _run_entrypoint()
