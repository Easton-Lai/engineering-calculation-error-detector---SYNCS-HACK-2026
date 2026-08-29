import ast
import operator as op

import pandas as pd
import streamlit as st
from pint import UnitRegistry


# -----------------------------
# Basic setup
# -----------------------------
st.set_page_config(page_title="UnitGuard", layout="wide")
st.title("UnitGuard")

ureg = UnitRegistry()


# -----------------------------
# Helpers
# -----------------------------
def normalize_text(text: str) -> str:
    """Normalize user input for expressions and units."""
    if text is None:
        return ""
    text = str(text).strip()
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
        text = text.replace(old, new)
    return text


def format_exponent(value):
    if int(value) == value:
        value = int(value)
    return str(value)


def format_dimensions(quantity) -> str:
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

    order = ["[mass]", "[length]", "[time]", "[current]", "[temperature]", "[substance]", "[luminosity]"]

    parts = []
    for key in order:
        if key in dims:
            exp = dims[key]
            sym = symbol_map.get(key, key)
            if exp == 1:
                parts.append(sym)
            else:
                parts.append(f"{sym}^{format_exponent(exp)}")
    return " ".join(parts)


def compatible_quantity_name(quantity) -> str:
    dims = quantity.dimensionality
    known = {
        frozenset({"[mass]": 1, "[length]": -1, "[time]": -2}.items()): "Pressure / Stress",
        frozenset({"[mass]": 1, "[length]": 1, "[time]": -2}.items()): "Energy / Torque",
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
ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}

ALLOWED_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def eval_expr(expr: str, variables: dict):
    expr = normalize_text(expr)
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body, variables)


def _eval_ast(node, variables):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Num):  # compatibility
        return node.n

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown variable: {node.id}")
        return variables[node.id]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise ValueError("Operator not allowed.")
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        return ALLOWED_BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("Unary operator not allowed.")
        operand = _eval_ast(node.operand, variables)
        return ALLOWED_UNARY_OPS[op_type](operand)

    raise ValueError("Invalid expression.")


def build_variables(df: pd.DataFrame) -> dict:
    variables = {}
    for _, row in df.iterrows():
        name = str(row["Variable"]).strip()
        if not name:
            continue

        value = row["Value"]
        if value == "" or pd.isna(value):
            continue

        value = float(value)
        unit = normalize_text(str(row["Unit"]).strip())

        if unit:
            variables[name] = value * ureg(unit)
        else:
            variables[name] = value
    return variables


# -----------------------------
# UI
# -----------------------------
st.subheader("Expression")
expression = st.text_input("", value="F / A", label_visibility="collapsed")

st.write("Then a table:")
default_df = pd.DataFrame(
    [
        {"Variable": "F", "Value": 10.0, "Unit": "kN"},
        {"Variable": "A", "Value": 200.0, "Unit": "mm^2"},
    ]
)

edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "Variable": st.column_config.TextColumn("Variable"),
        "Value": st.column_config.NumberColumn("Value"),
        "Unit": st.column_config.TextColumn("Unit"),
    },
)

st.write("Then enter:")
expected_unit = st.text_input("Expected output unit:", value="MPa")

if st.button("CHECK CALCULATION"):
    try:
        variables = build_variables(edited_df)

        if not expression.strip():
            raise ValueError("Expression cannot be empty.")

        result = eval_expr(expression, variables)

        if not hasattr(result, "dimensionality"):
            result = result * ureg.dimensionless

        actual_dims = format_dimensions(result)
        actual_kind = compatible_quantity_name(result)

        if expected_unit.strip():
            expected_q = 1 * ureg(normalize_text(expected_unit))
            is_consistent = (result.dimensionality == expected_q.dimensionality)

            if is_consistent:
                converted = result.to(normalize_text(expected_unit))
                si_result = result.to_base_units()

                output = (
                    "✓ DIMENSIONALLY CONSISTENT\n\n"
                    f"Result\n{converted:~P}\n\n"
                    f"SI form\n{si_result:~P}\n\n"
                    f"Dimensions\n{actual_dims}\n\n"
                    f"Compatible quantity dimension:\n{actual_kind}"
                )
                st.write("Result:")
                st.code(output)
            else:
                expected_dims = format_dimensions(expected_q)
                output = (
                    "✗ DIMENSIONAL MISMATCH\n\n"
                    f"Expression produces\n{actual_dims}\n\n"
                    f"Expected\n{expected_dims}\n\n"
                    f"Compatible quantity dimension:\n{actual_kind}"
                )
                st.write("Result:")
                st.code(output)
        else:
            si_result = result.to_base_units()
            output = (
                "✓ CALCULATION COMPLETE\n\n"
                f"Result\n{result:~P}\n\n"
                f"SI form\n{si_result:~P}\n\n"
                f"Dimensions\n{actual_dims}\n\n"
                f"Compatible quantity dimension:\n{actual_kind}"
            )
            st.write("Result:")
            st.code(output)

    except Exception as e:
        st.write("Result:")
        st.code(f"ERROR\n\n{str(e)}")