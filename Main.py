from __future__ import annotations

import ast
import keyword
import operator as op
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pint import UnitRegistry
from streamlit import runtime as st_runtime


# =========================================================
# Unit registry
# =========================================================
ureg = UnitRegistry()


# =========================================================
# Text normalisation
# =========================================================
def normalize_text(text: str | None) -> str:
    if text is None:
        return ""

    normalized = str(text).strip()

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


# =========================================================
# Dimension formatting
# =========================================================
DIMENSION_SYMBOLS = {
    "[mass]": "M",
    "[length]": "L",
    "[time]": "T",
    "[current]": "I",
    "[temperature]": "Θ",
    "[substance]": "N",
    "[luminosity]": "J",
}

DIMENSION_ORDER = [
    "[mass]",
    "[length]",
    "[time]",
    "[current]",
    "[temperature]",
    "[substance]",
    "[luminosity]",
]


def format_exponent(value) -> str:
    """Format a dimensional exponent cleanly."""
    try:
        integer_value = int(value)

        if integer_value == value:
            return str(integer_value)

    except (TypeError, ValueError, OverflowError):
        pass

    return str(value)


def format_dimensionality(dimensionality) -> str:
    if not dimensionality:
        return "dimensionless"

    dimensionality_dict = dict(dimensionality)

    ordered_keys = DIMENSION_ORDER.copy()

    for key in dimensionality_dict:
        if key not in ordered_keys:
            ordered_keys.append(key)

    parts: list[str] = []

    for key in ordered_keys:
        exponent = dimensionality_dict.get(key, 0)

        if exponent == 0:
            continue

        symbol = DIMENSION_SYMBOLS.get(key, str(key))

        if exponent == 1:
            parts.append(symbol)
        else:
            parts.append(f"{symbol}^{format_exponent(exponent)}")

    return " ".join(parts) if parts else "dimensionless"


def format_dimensions(quantity) -> str:
    """Return the dimension notation of a Pint quantity."""
    return format_dimensionality(quantity.dimensionality)


def compatible_quantity_name(quantity) -> str:
    dimensions = frozenset(quantity.dimensionality.items())

    known_dimensions = {
        frozenset(
            {
                "[mass]": 1,
                "[length]": -1,
                "[time]": -2,
            }.items()
        ): "Pressure / Stress",

        frozenset(
            {
                "[mass]": 1,
                "[length]": 2,
                "[time]": -2,
            }.items()
        ): "Energy / Torque",

        frozenset(
            {
                "[length]": 1,
                "[time]": -1,
            }.items()
        ): "Velocity",

        frozenset(
            {
                "[length]": 1,
                "[time]": -2,
            }.items()
        ): "Acceleration",

        frozenset(
            {
                "[mass]": 1,
                "[length]": 1,
                "[time]": -1,
            }.items()
        ): "Momentum",

        frozenset(
            {
                "[mass]": 1,
                "[length]": 2,
                "[time]": -3,
            }.items()
        ): "Power",

        frozenset(
            {
                "[mass]": 1,
                "[length]": 1,
                "[time]": -2,
            }.items()
        ): "Force",

        frozenset(
            {
                "[length]": 2,
            }.items()
        ): "Area",

        frozenset(
            {
                "[length]": 3,
            }.items()
        ): "Volume",

        frozenset(
            {
                "[length]": 3,
                "[time]": -1,
            }.items()
        ): "Volumetric Flow Rate",

        frozenset(
            {
                "[mass]": 1,
                "[length]": -3,
            }.items()
        ): "Density",

        frozenset(): "Dimensionless",
    }

    return known_dimensions.get(dimensions, "Unknown / Not mapped")


def dimensional_ratio(left_quantity, right_quantity) -> str:

    left_dims = dict(left_quantity.dimensionality)
    right_dims = dict(right_quantity.dimensionality)

    all_keys = set(left_dims) | set(right_dims)

    difference = {}

    for key in all_keys:
        exponent = right_dims.get(key, 0) - left_dims.get(key, 0)

        if exponent != 0:
            difference[key] = exponent

    return format_dimensionality(difference)


# =========================================================
# Safe expression evaluator
# =========================================================
ALLOWED_BINARY_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def is_valid_calculation_value(value) -> bool:

    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return True

    if hasattr(value, "dimensionality"):
        return True

    return False


def eval_expr(expression: str, variable_map: dict):
    normalized_expression = normalize_text(expression)

    try:
        syntax_tree = ast.parse(normalized_expression, mode="eval")
    except SyntaxError:
        raise ValueError("Invalid expression syntax.") from None

    return evaluate_ast_node(syntax_tree.body, variable_map)


def evaluate_ast_node(node, variable_map: dict):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                "Expressions may only contain numbers and variables."
            )

        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not allowed.")

        return node.value

    if isinstance(node, ast.Name):
        if node.id not in variable_map:
            raise ValueError(f"Unknown variable: {node.id}")

        return variable_map[node.id]

    if isinstance(node, ast.BinOp):
        operator_function = ALLOWED_BINARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("This operator is not allowed.")

        left_value = evaluate_ast_node(node.left, variable_map)
        right_value = evaluate_ast_node(node.right, variable_map)

        result = operator_function(left_value, right_value)

        if not is_valid_calculation_value(result):
            raise ValueError("The expression produced an invalid result.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_function = ALLOWED_UNARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("This unary operator is not allowed.")

        operand = evaluate_ast_node(node.operand, variable_map)
        result = operator_function(operand)

        if not is_valid_calculation_value(result):
            raise ValueError("The expression produced an invalid result.")

        return result

    raise ValueError("Invalid or unsupported expression.")



def is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_variable_name(name: str, row_number: int) -> None:

    if not name:
        raise ValueError(
            f"Variable name is required in row {row_number}."
        )

    if not name.isidentifier():
        raise ValueError(
            f"Invalid variable name in row {row_number}: {name}"
        )

    if keyword.iskeyword(name):
        raise ValueError(
            f"Python keyword cannot be used as a variable: {name}"
        )


# =========================================================
# Calculation Checker variable builder
# =========================================================
def build_calculation_variables(
    dataframe: pd.DataFrame,
) -> dict:

    variable_map = {}

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        raw_name = row.get("Variable", "")
        raw_value = row.get("Value", "")
        raw_unit = row.get("Unit", "")

        empty_name = is_missing(raw_name) or not str(raw_name).strip()
        empty_value = is_missing(raw_value) or raw_value == ""
        empty_unit = is_missing(raw_unit) or not str(raw_unit).strip()

        # Ignore completely empty rows
        if empty_name and empty_value and empty_unit:
            continue

        if empty_name:
            raise ValueError(
                f"Variable name is required in row {row_number}."
            )

        name = str(raw_name).strip()
        validate_variable_name(name, row_number)

        if name in variable_map:
            raise ValueError(f"Duplicate variable name: {name}")

        if empty_value:
            raise ValueError(
                f"Value is required for variable: {name}"
            )

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid numeric value for variable: {name}"
            ) from None

        unit_text = (
            ""
            if empty_unit
            else normalize_text(str(raw_unit))
        )

        if unit_text:
            try:
                variable_map[name] = ureg.Quantity(
                    numeric_value,
                    unit_text,
                )
            except Exception:
                raise ValueError(
                    f"Invalid unit for variable {name}: {unit_text}"
                ) from None
        else:
            variable_map[name] = numeric_value

    if not variable_map:
        raise ValueError("At least one variable is required.")

    return variable_map


# =========================================================
# Dimensional Checker variable builder
# =========================================================
def build_dimension_variables(
    dataframe: pd.DataFrame,
) -> dict:

    variable_map = {}

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        raw_name = row.get("Variable", "")
        raw_unit = row.get("Unit", "")

        empty_name = is_missing(raw_name) or not str(raw_name).strip()
        empty_unit = is_missing(raw_unit) or not str(raw_unit).strip()

        # Ignore completely empty rows
        if empty_name and empty_unit:
            continue

        if empty_name:
            raise ValueError(
                f"Variable name is required in row {row_number}."
            )

        name = str(raw_name).strip()
        validate_variable_name(name, row_number)

        if name in variable_map:
            raise ValueError(f"Duplicate variable name: {name}")

        if empty_unit:
            raise ValueError(
                f"Unit is required for variable: {name}"
            )

        unit_text = normalize_text(str(raw_unit))

        try:
            variable_map[name] = ureg.Quantity(1, unit_text)
        except Exception:
            raise ValueError(
                f"Invalid unit for variable {name}: {unit_text}"
            ) from None

    if not variable_map:
        raise ValueError("At least one variable is required.")

    return variable_map


def ensure_quantity(value):
    """
    Convert a plain numerical result into a dimensionless Pint quantity.
    """
    if hasattr(value, "dimensionality"):
        return value

    return ureg.Quantity(value)


# =========================================================
# Calculation Checker UI
# =========================================================
def render_calculation_checker() -> None:
    st.subheader("Calculation Checker")

    st.write(
        "Enter an engineering expression, numerical values and units."
    )

    expression = (
        st.text_input(
            "Expression",
            value="F / A",
            key="calculation_expression",
        )
        or ""
    )

    default_calculation_table = pd.DataFrame(
        [
            {
                "Variable": "F",
                "Value": 10.0,
                "Unit": "kN",
            },
            {
                "Variable": "A",
                "Value": 200.0,
                "Unit": "mm^2",
            },
        ]
    )

    calculation_table = st.data_editor(
        default_calculation_table,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="calculation_variable_table",
        column_config={
            "Variable": st.column_config.TextColumn(
                "Variable"
            ),
            "Value": st.column_config.NumberColumn(
                "Value"
            ),
            "Unit": st.column_config.TextColumn(
                "Unit"
            ),
        },
    )

    expected_unit = (
        st.text_input(
            "Expected output unit",
            value="MPa",
            key="calculation_expected_unit",
        )
        or ""
    )

    if st.button(
        "CHECK CALCULATION",
        key="check_calculation_button",
    ):
        try:
            if not expression.strip():
                raise ValueError("Expression cannot be empty.")

            variable_map = build_calculation_variables(
                calculation_table
            )

            evaluated_result = eval_expr(
                expression,
                variable_map,
            )

            result = ensure_quantity(evaluated_result)

            actual_dimensions = format_dimensions(result)
            actual_quantity_type = compatible_quantity_name(result)

            expected_unit_text = normalize_text(expected_unit)

            if expected_unit_text:
                try:
                    expected_unit_object = ureg.parse_units(
                        expected_unit_text
                    )
                except Exception:
                    raise ValueError(
                        f"Invalid expected output unit: "
                        f"{expected_unit_text}"
                    ) from None

                expected_quantity = ureg.Quantity(
                    1,
                    expected_unit_object,
                )

                is_consistent = (
                    result.dimensionality
                    == expected_quantity.dimensionality
                )

                if is_consistent:
                    converted_result = result.to(
                        expected_unit_object
                    )

                    si_result = result.to_base_units()

                    output = (
                        "✓ DIMENSIONALLY CONSISTENT\n\n"
                        f"Result\n"
                        f"{format(converted_result, '~P')}\n\n"
                        f"SI form\n"
                        f"{format(si_result, '~P')}\n\n"
                        f"Dimensions\n"
                        f"{actual_dimensions}\n\n"
                        f"Compatible quantity dimension:\n"
                        f"{actual_quantity_type}"
                    )

                else:
                    expected_dimensions = format_dimensions(
                        expected_quantity
                    )

                    output = (
                        "✗ DIMENSIONAL MISMATCH\n\n"
                        f"Expression produces\n"
                        f"{actual_dimensions}\n\n"
                        f"Expected\n"
                        f"{expected_dimensions}\n\n"
                        f"Expression quantity type\n"
                        f"{actual_quantity_type}"
                    )

            else:
                si_result = result.to_base_units()

                output = (
                    "✓ CALCULATION COMPLETE\n\n"
                    f"Result\n"
                    f"{format(result, '~P')}\n\n"
                    f"SI form\n"
                    f"{format(si_result, '~P')}\n\n"
                    f"Dimensions\n"
                    f"{actual_dimensions}\n\n"
                    f"Compatible quantity dimension:\n"
                    f"{actual_quantity_type}"
                )

            st.write("Result:")
            st.code(output)

        except Exception as error:
            st.write("Result:")
            st.code(f"ERROR\n\n{error}")


# =========================================================
# Dimensional Checker UI
# =========================================================
def render_dimensional_checker() -> None:
    st.subheader("Dimensional Checker")

    st.write(
        "Enter a complete equation and assign a unit to each variable. "
        "Numerical values are not required."
    )

    equation = (
        st.text_input(
            "Equation",
            value="F = m * a",
            key="dimension_equation",
        )
        or ""
    )

    default_dimension_table = pd.DataFrame(
        [
            {
                "Variable": "F",
                "Unit": "N",
            },
            {
                "Variable": "m",
                "Unit": "kg",
            },
            {
                "Variable": "a",
                "Unit": "m/s^2",
            },
        ]
    )

    dimension_table = st.data_editor(
        default_dimension_table,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="dimension_variable_table",
        column_config={
            "Variable": st.column_config.TextColumn(
                "Variable"
            ),
            "Unit": st.column_config.TextColumn(
                "Unit"
            ),
        },
    )

    if st.button(
        "CHECK DIMENSIONS",
        key="check_dimensions_button",
    ):
        try:
            if not equation.strip():
                raise ValueError("Equation cannot be empty.")

            if equation.count("=") != 1:
                raise ValueError(
                    "The equation must contain exactly one '=' sign."
                )

            left_expression, right_expression = equation.split(
                "=",
                maxsplit=1,
            )

            left_expression = left_expression.strip()
            right_expression = right_expression.strip()

            if not left_expression:
                raise ValueError(
                    "The left-hand side cannot be empty."
                )

            if not right_expression:
                raise ValueError(
                    "The right-hand side cannot be empty."
                )

            variable_map = build_dimension_variables(
                dimension_table
            )

            left_result = ensure_quantity(
                eval_expr(
                    left_expression,
                    variable_map,
                )
            )

            right_result = ensure_quantity(
                eval_expr(
                    right_expression,
                    variable_map,
                )
            )

            left_dimensions = format_dimensions(left_result)
            right_dimensions = format_dimensions(right_result)

            left_quantity_type = compatible_quantity_name(
                left_result
            )

            right_quantity_type = compatible_quantity_name(
                right_result
            )

            is_consistent = (
                left_result.dimensionality
                == right_result.dimensionality
            )

            if is_consistent:
                output = (
                    "✓ DIMENSIONALLY CONSISTENT\n\n"
                    f"Left-hand side\n"
                    f"{left_expression}\n"
                    f"Dimensions: {left_dimensions}\n"
                    f"Quantity type: {left_quantity_type}\n\n"
                    f"Right-hand side\n"
                    f"{right_expression}\n"
                    f"Dimensions: {right_dimensions}\n"
                    f"Quantity type: {right_quantity_type}\n\n"
                    f"Both sides have the same physical dimensions."
                )

            else:
                ratio = dimensional_ratio(
                    left_result,
                    right_result,
                )

                output = (
                    "✗ DIMENSIONAL MISMATCH\n\n"
                    f"Left-hand side\n"
                    f"{left_expression}\n"
                    f"Dimensions: {left_dimensions}\n"
                    f"Quantity type: {left_quantity_type}\n\n"
                    f"Right-hand side\n"
                    f"{right_expression}\n"
                    f"Dimensions: {right_dimensions}\n"
                    f"Quantity type: {right_quantity_type}\n\n"
                    f"Right / left dimensional ratio\n"
                    f"{ratio}\n\n"
                    f"A valid physical equation requires this ratio "
                    f"to be dimensionless."
                )

            st.write("Result:")
            st.code(output)

        except Exception as error:
            st.write("Result:")
            st.code(f"ERROR\n\n{error}")

    st.caption(
        "Dimensional consistency is necessary, but it does not prove "
        "that an equation is physically correct."
    )


# =========================================================
# Main Streamlit application
# =========================================================
def main() -> None:
    st.set_page_config(
        page_title="UnitGuard",
        layout="wide",
    )

    st.title("UnitGuard")

    calculation_tab, dimensional_tab = st.tabs(
        [
            "Calculation Checker",
            "Dimensional Checker",
        ]
    )

    with calculation_tab:
        render_calculation_checker()

    with dimensional_tab:
        render_dimensional_checker()





def launch_streamlit() -> None:

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]

    try:
        completed_process = subprocess.run(
            command,
            check=False,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None

    raise SystemExit(completed_process.returncode)


def run_entrypoint() -> None:
    if st_runtime.exists():
        main()
    else:
        launch_streamlit()


if __name__ == "__main__":
    run_entrypoint()