from __future__ import annotations

import ast
import keyword
import logging
import math
import operator as op
import re
import subprocess
import sys
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, NoReturn, TypeAlias, cast

import pandas as pd
from pint.errors import PintError
from pint.registry import Quantity, Unit, UnitRegistry
import streamlit as st
from streamlit import runtime as st_runtime


Scalar: TypeAlias = int | float
CalculationQuantity: TypeAlias = Quantity[Any]
CalculationValue: TypeAlias = Scalar | CalculationQuantity
DimensionMap: TypeAlias = dict[str, Fraction]
DimensionKey: TypeAlias = frozenset[tuple[str, Fraction]]
BinaryOperation: TypeAlias = Callable[[CalculationValue, CalculationValue], object]
UnaryOperation: TypeAlias = Callable[[CalculationValue], object]
QuantityFactory: TypeAlias = Callable[
    [Scalar, str | Unit | None], CalculationQuantity
]

MAX_EXPRESSION_LENGTH: Final = 1_000
MAX_AST_NODES: Final = 200
MAX_ABS_EXPONENT: Final = 1_000
MAX_INTEGER_BITS: Final = 4_096
MAX_EXPONENT_DENOMINATOR: Final = 1_000_000
EXPONENT_REL_TOLERANCE: Final = 1e-12
EXPONENT_ABS_TOLERANCE: Final = 1e-15

LOGGER: Final = logging.getLogger(__name__)

ureg: UnitRegistry[Any] = UnitRegistry("")

# Pint creates the registry-bound Quantity class dynamically.
_QUANTITY_FACTORY = cast(
    QuantityFactory,
    cast(object, ureg.Quantity),
)

_SUPERSCRIPT_PATTERN: Final = re.compile(
    r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+"
)

_SUPERSCRIPT_TRANSLATION: Final = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)

_TEXT_REPLACEMENTS: Final = {
    "Δ°C": "delta_degC",
    "Δ°F": "delta_degF",
    "℃": "degC",
    "℉": "degF",
    "°C": "degC",
    "°F": "degF",
    "^": "**",
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "−": "-",
}

DIMENSION_SYMBOLS: Final = {
    "[mass]": "M",
    "[length]": "L",
    "[time]": "T",
    "[current]": "I",
    "[temperature]": "Θ",
    "[substance]": "N",
    "[luminosity]": "J",
}

DIMENSION_ORDER: Final = (
    "[mass]",
    "[length]",
    "[time]",
    "[current]",
    "[temperature]",
    "[substance]",
    "[luminosity]",
)


class UserInputError(ValueError):
    """An invalid expression, unit, table row, or expected result."""


@dataclass(frozen=True, slots=True)
class DimensionValue:
    dimensions: Mapping[str, Fraction]
    scalar: Scalar | None = None


def _dimension_key(**dimensions: int) -> DimensionKey:
    return frozenset(
        (f"[{name}]", Fraction(exponent))
        for name, exponent in dimensions.items()
        if exponent
    )


KNOWN_DIMENSIONS: Final[dict[DimensionKey, str]] = {
    _dimension_key(length=1): "Length",
    _dimension_key(length=2): "Area",
    _dimension_key(length=3): "Volume",
    _dimension_key(length=4): "Second Moment of Area",

    _dimension_key(mass=1): "Mass",
    _dimension_key(time=1): "Time",
    _dimension_key(temperature=1): "Temperature",
    _dimension_key(current=1): "Electric Current",

    _dimension_key(time=-1): "Frequency / Rotational Speed",

    _dimension_key(
        length=1,
        time=-1,
    ): "Velocity",

    _dimension_key(
        length=1,
        time=-2,
    ): "Acceleration",

    _dimension_key(
        mass=1,
        length=1,
        time=-2,
    ): "Force",

    _dimension_key(
        mass=1,
        length=-1,
        time=-2,
    ): "Pressure / Stress",

    _dimension_key(
        mass=1,
        length=2,
        time=-2,
    ): "Energy / Torque",

    _dimension_key(
        mass=1,
        length=2,
        time=-3,
    ): "Power",

    _dimension_key(
        mass=1,
        length=1,
        time=-1,
    ): "Momentum",

    _dimension_key(
        mass=1,
        length=-3,
    ): "Density",

    _dimension_key(
        length=3,
        time=-1,
    ): "Volumetric Flow Rate",

    _dimension_key(
        mass=1,
        time=-1,
    ): "Mass Flow Rate",

    _dimension_key(
        mass=1,
        length=-1,
        time=-1,
    ): "Dynamic Viscosity",

    _dimension_key(
        length=2,
        time=-1,
    ): "Kinematic Viscosity",

    _dimension_key(
        mass=1,
        length=2,
    ): "Mass Moment of Inertia",

    _dimension_key(
        mass=1,
        time=-2,
    ): "Surface Tension",

    _dimension_key(
        length=2,
        time=-2,
    ): "Specific Energy",

    _dimension_key(
        length=2,
        time=-2,
        temperature=-1,
    ): "Specific Heat Capacity",

    _dimension_key(
        mass=1,
        length=1,
        time=-3,
        temperature=-1,
    ): "Thermal Conductivity",

    _dimension_key(
        mass=1,
        time=-3,
        temperature=-1,
    ): "Heat Transfer Coefficient",

    _dimension_key(
        mass=1,
        length=2,
        time=-3,
        current=-1,
    ): "Voltage",

    _dimension_key(
        mass=1,
        length=2,
        time=-3,
        current=-2,
    ): "Electrical Resistance",

    _dimension_key(
        current=1,
        time=1,
    ): "Electric Charge",

    frozenset(): "Dimensionless",
}


UNIT_CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "Length": (
        "mm",
        "cm",
        "m",
        "km",
        "inch",
        "foot",
        "yard",
        "mile",
        "nautical_mile",
    ),

    "Area": (
        "mm**2",
        "cm**2",
        "m**2",
        "km**2",
        "inch**2",
        "foot**2",
        "hectare",
        "acre",
    ),

    "Volume": (
        "mm**3",
        "cm**3",
        "m**3",
        "milliliter",
        "liter",
        "inch**3",
        "foot**3",
        "US_liquid_gallon",
        "imperial_gallon",
    ),

    "Mass": (
        "mg",
        "g",
        "kg",
        "tonne",
        "ounce",
        "pound",
        "stone",
        "short_ton",
        "long_ton",
    ),

    "Time": (
        "ms",
        "s",
        "minute",
        "hour",
        "day",
    ),

    "Absolute Temperature": (
        "degC",
        "degF",
        "kelvin",
        "degR",
    ),

    "Temperature Difference": (
        "delta_degC",
        "delta_degF",
        "kelvin",
    ),

    "Angle": (
        "degree",
        "radian",
        "revolution",
        "arcminute",
        "arcsecond",
    ),

    "Speed": (
        "m/s",
        "km/hour",
        "mph",
        "foot/s",
        "knot",
    ),

    "Acceleration": (
        "m/s**2",
        "cm/s**2",
        "foot/s**2",
        "Gal",
    ),

    "Frequency": (
        "Hz",
        "kHz",
        "MHz",
        "1/minute",
    ),

    "Rotational Speed": (
        "rpm",
        "rps",
        "radian/s",
        "degree/s",
    ),

    "Force": (
        "N",
        "kN",
        "MN",
        "dyn",
        "kgf",
        "lbf",
        "kip",
    ),

    "Pressure / Stress": (
        "Pa",
        "kPa",
        "MPa",
        "GPa",
        "bar",
        "mbar",
        "atm",
        "torr",
        "psi",
        "ksi",
        "mH2O",
        "inH2O",
        "inHg",
    ),

    "Torque": (
        "N*m",
        "kN*m",
        "N*mm",
        "lbf*foot",
        "lbf*inch",
    ),

    "Energy / Work": (
        "J",
        "kJ",
        "MJ",
        "Wh",
        "kWh",
        "cal",
        "kcal",
        "Btu",
    ),

    "Power": (
        "W",
        "kW",
        "MW",
        "hp",
        "metric_horsepower",
        "Btu/hour",
    ),

    "Density": (
        "kg/m**3",
        "g/cm**3",
        "kg/liter",
        "lb/foot**3",
        "lb/inch**3",
    ),

    "Dynamic Viscosity": (
        "Pa*s",
        "mPa*s",
        "cP",
        "P",
        "reyn",
    ),

    "Kinematic Viscosity": (
        "m**2/s",
        "mm**2/s",
        "cSt",
        "St",
        "foot**2/s",
    ),

    "Volumetric Flow Rate": (
        "m**3/s",
        "m**3/hour",
        "liter/s",
        "liter/minute",
        "cm**3/s",
        "foot**3/s",
        "foot**3/minute",
        "US_liquid_gallon/minute",
        "imperial_gallon/minute",
    ),

    "Mass Flow Rate": (
        "kg/s",
        "kg/hour",
        "g/s",
        "tonne/hour",
        "lb/s",
        "lb/hour",
    ),

    "Mass Moment of Inertia": (
        "kg*m**2",
        "kg*mm**2",
        "g*mm**2",
        "lb*foot**2",
        "lb*inch**2",
    ),

    "Second Moment of Area": (
        "m**4",
        "cm**4",
        "mm**4",
        "inch**4",
        "foot**4",
    ),

    "Surface Tension": (
        "N/m",
        "mN/m",
        "dyn/cm",
        "lbf/foot",
    ),

    "Specific Energy": (
        "J/kg",
        "kJ/kg",
        "Btu/lb",
    ),

    "Specific Heat Capacity": (
        "J/(kg*kelvin)",
        "kJ/(kg*kelvin)",
        "Btu/(lb*delta_degF)",
    ),

    "Thermal Conductivity": (
        "W/(m*kelvin)",
        "mW/(m*kelvin)",
        "Btu/(hour*foot*delta_degF)",
    ),

    "Heat Transfer Coefficient": (
        "W/(m**2*kelvin)",
        "Btu/(hour*foot**2*delta_degF)",
    ),

    "Electric Current": (
        "uA",
        "mA",
        "A",
        "kA",
    ),

    "Voltage": (
        "mV",
        "V",
        "kV",
    ),

    "Resistance": (
        "milliohm",
        "ohm",
        "kiloohm",
        "megaohm",
    ),

    "Electric Charge": (
        "C",
        "mAh",
        "Ah",
    ),
}


_UNIT_LABEL_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("US_liquid_gallon", "US gal"),
    ("imperial_gallon", "Imp gal"),
    ("metric_horsepower", "metric hp"),
    ("nautical_mile", "nmi"),
    ("delta_degC", "Δ°C"),
    ("delta_degF", "Δ°F"),
    ("milliliter", "mL"),
    ("liter", "L"),
    ("revolution", "rev"),
    ("radian", "rad"),
    ("degree", "deg"),
    ("arcminute", "arcmin"),
    ("arcsecond", "arcsec"),
    ("minute", "min"),
    ("hour", "h"),
    ("foot", "ft"),
    ("inch", "in"),
    ("pound", "lb"),
    ("milliohm", "mΩ"),
    ("kiloohm", "kΩ"),
    ("megaohm", "MΩ"),
    ("ohm", "Ω"),
    ("**4", "⁴"),
    ("**3", "³"),
    ("**2", "²"),
    ("*", "·"),
    ("_", " "),
)


def _as_binary_operation(operation: object) -> BinaryOperation:
    return cast(BinaryOperation, operation)


def _as_unary_operation(operation: object) -> UnaryOperation:
    return cast(UnaryOperation, operation)


ALLOWED_BINARY_OPERATORS: Final[dict[type[ast.operator], BinaryOperation]] = {
    ast.Add: _as_binary_operation(op.add),
    ast.Sub: _as_binary_operation(op.sub),
    ast.Mult: _as_binary_operation(op.mul),
    ast.Div: _as_binary_operation(op.truediv),
    ast.Pow: _as_binary_operation(op.pow),
}

ALLOWED_UNARY_OPERATORS: Final[dict[type[ast.unaryop], UnaryOperation]] = {
    ast.UAdd: _as_unary_operation(op.pos),
    ast.USub: _as_unary_operation(op.neg),
}


def make_quantity(
    magnitude: Scalar,
    unit: str | Unit | None = None,
) -> CalculationQuantity:
    return _QUANTITY_FACTORY(_require_real_scalar(magnitude), unit)


def parse_unit(unit_text: str) -> Unit:
    return ureg.parse_units(unit_text)


def _replace_superscript(match: re.Match[str]) -> str:
    exponent = match.group().translate(
        _SUPERSCRIPT_TRANSLATION
    )

    signless_exponent = (
        exponent[1:]
        if exponent[:1] in {"+", "-"}
        else exponent
    )

    if not signless_exponent.isdigit():
        raise UserInputError(
            f"Invalid superscript exponent: {match.group()}"
        )

    return f"**{exponent}"


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""

    normalized = _SUPERSCRIPT_PATTERN.sub(
        _replace_superscript,
        text.strip(),
    )

    for old, new in _TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)

    return normalized


def _canonical_fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        raise UserInputError(
            "An exponent must be a finite real number."
        )

    if isinstance(value, Fraction):
        exponent = value

    elif isinstance(value, int):
        exponent = Fraction(value)

    elif isinstance(value, float):
        if not math.isfinite(value):
            raise UserInputError(
                "An exponent must be a finite real number."
            )

        candidate = Fraction(value).limit_denominator(
            MAX_EXPONENT_DENOMINATOR
        )

        exponent = (
            candidate
            if math.isclose(
                value,
                float(candidate),
                rel_tol=EXPONENT_REL_TOLERANCE,
                abs_tol=EXPONENT_ABS_TOLERANCE,
            )
            else Fraction(str(value))
        )

    else:
        raise UserInputError(
            "An exponent must be a finite real number."
        )

    if abs(exponent) > MAX_ABS_EXPONENT:
        raise UserInputError(
            f"Exponent magnitude cannot exceed "
            f"{MAX_ABS_EXPONENT}."
        )

    return exponent


def _fraction_as_scalar(value: Fraction) -> Scalar:
    return value.numerator if value.denominator == 1 else float(value)


def _normalized_power(value: CalculationValue) -> Scalar:
    if isinstance(value, Quantity):
        if value.dimensionality:
            raise UserInputError(
                "An exponent must be dimensionless."
            )

        scalar = _require_real_scalar(
            value.magnitude
        )

    else:
        scalar = _require_real_scalar(value)

    exponent = _canonical_fraction(scalar)

    return _fraction_as_scalar(exponent)


def _require_real_scalar(value: object) -> Scalar:
    if isinstance(value, bool):
        raise UserInputError(
            "Boolean values are not allowed."
        )

    if isinstance(value, int):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise UserInputError(
                "The expression produced an integer "
                "that is too large."
            )

        return value

    if (
        isinstance(value, float)
        and math.isfinite(value)
    ):
        return value

    raise UserInputError(
        "Values must be finite real numbers."
    )


def _require_calculation_value(value: object) -> CalculationValue:
    if isinstance(value, Quantity):
        _require_real_scalar(value.magnitude)
        return value

    return _require_real_scalar(value)


def _parse_expression(expression: str) -> ast.expr:
    normalized = normalize_text(expression)

    if not normalized:
        raise UserInputError(
            "Expression cannot be empty."
        )

    if len(normalized) > MAX_EXPRESSION_LENGTH:
        raise UserInputError(
            f"Expression cannot exceed "
            f"{MAX_EXPRESSION_LENGTH} characters."
        )

    try:
        tree = ast.parse(
            normalized,
            mode="eval",
        )

    except SyntaxError:
        raise UserInputError(
            "Invalid expression syntax."
        ) from None

    node_count = sum(
        1
        for _ in ast.walk(tree)
    )

    if node_count > MAX_AST_NODES:
        raise UserInputError(
            f"Expression cannot exceed "
            f"{MAX_AST_NODES} syntax nodes."
        )

    return tree.body


def _apply_binary_operation(
    operation: BinaryOperation,
    left: CalculationValue,
    right: CalculationValue,
) -> CalculationValue:
    try:
        result = operation(left, right)

    except (
        ArithmeticError,
        PintError,
        TypeError,
        ValueError,
    ) as error:
        raise UserInputError(
            str(error)
            or "The calculation could not be completed."
        ) from None

    return _require_calculation_value(result)


def _apply_unary_operation(
    operation: UnaryOperation,
    operand: CalculationValue,
) -> CalculationValue:
    try:
        result = operation(operand)

    except (
        ArithmeticError,
        PintError,
        TypeError,
        ValueError,
    ) as error:
        raise UserInputError(
            str(error)
            or "The calculation could not be completed."
        ) from None

    return _require_calculation_value(result)


def eval_expr(
    expression: str,
    variable_map: Mapping[
        str,
        CalculationValue,
    ],
) -> CalculationValue:
    return _eval_ast(
        _parse_expression(expression),
        variable_map,
    )


def _eval_ast(
    node: ast.expr,
    variable_map: Mapping[
        str,
        CalculationValue,
    ],
) -> CalculationValue:
    if isinstance(node, ast.Constant):
        return _require_calculation_value(
            node.value
        )

    if isinstance(node, ast.Name):
        if node.id not in variable_map:
            raise UserInputError(
                f"Unknown variable: {node.id}"
            )

        return _require_calculation_value(
            variable_map[node.id]
        )

    if isinstance(node, ast.BinOp):
        left = _eval_ast(
            node.left,
            variable_map,
        )

        right = _eval_ast(
            node.right,
            variable_map,
        )

        if isinstance(node.op, ast.Pow):
            exponent = _normalized_power(right)

            if (
                isinstance(left, int)
                and isinstance(exponent, int)
                and exponent > 0
                and max(
                    1,
                    left.bit_length(),
                ) * exponent
                > MAX_INTEGER_BITS
            ):
                raise UserInputError(
                    "The expression would produce "
                    "an integer that is too large."
                )

            return _apply_binary_operation(
                ALLOWED_BINARY_OPERATORS[ast.Pow],
                left,
                exponent,
            )

        operation = ALLOWED_BINARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise UserInputError(
                "This operator is not allowed."
            )

        return _apply_binary_operation(
            operation,
            left,
            right,
        )

    if isinstance(node, ast.UnaryOp):
        operation = ALLOWED_UNARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise UserInputError(
                "This unary operator is not allowed."
            )

        return _apply_unary_operation(
            operation,
            _eval_ast(
                node.operand,
                variable_map,
            ),
        )

    raise UserInputError(
        "Invalid or unsupported expression."
    )


def _dimension_map(value: CalculationQuantity | Unit) -> DimensionMap:
    return {
        str(key): _canonical_fraction(exponent)
        for key, exponent
        in value.dimensionality.items()
        if exponent
    }


def _combine_dimensions(
    left: Mapping[str, Fraction],
    right: Mapping[str, Fraction],
    right_factor: int,
) -> DimensionMap:
    combined: DimensionMap = {}

    for key in set(left) | set(right):
        exponent = (
            left.get(
                key,
                Fraction(),
            )
            + right_factor
            * right.get(
                key,
                Fraction(),
            )
        )

        if exponent:
            combined[key] = exponent

    return combined


def _scale_dimensions(
    dimensions: Mapping[str, Fraction],
    exponent: Fraction,
) -> DimensionMap:
    return {
        key: scaled_value
        for key, value in dimensions.items()
        if (scaled_value := value * exponent)
    }


def _format_exponent(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)

    return f"{float(exponent):.12g}"


def format_dimensionality(
    dimensions: Mapping[str, Fraction],
) -> str:
    if not dimensions:
        return "dimensionless"

    ordered_keys = [
        *DIMENSION_ORDER,
        *(
            key
            for key in dimensions
            if key not in DIMENSION_ORDER
        ),
    ]

    parts: list[str] = []

    for key in ordered_keys:
        exponent = dimensions.get(
            key,
            Fraction(),
        )

        if not exponent:
            continue

        symbol = DIMENSION_SYMBOLS.get(
            key,
            key,
        )

        parts.append(
            symbol
            if exponent == 1
            else (
                f"{symbol}^"
                f"{_format_exponent(exponent)}"
            )
        )

    return (
        " ".join(parts)
        if parts
        else "dimensionless"
    )


def format_dimensions(quantity: CalculationQuantity) -> str:
    return format_dimensionality(
        _dimension_map(quantity)
    )


def _compatible_dimension_name(
    dimensions: Mapping[str, Fraction],
) -> str:
    return KNOWN_DIMENSIONS.get(
        frozenset(dimensions.items()),
        "Unknown / Not mapped",
    )


def compatible_quantity_name(quantity: CalculationQuantity) -> str:
    return _compatible_dimension_name(
        _dimension_map(quantity)
    )


def dimensional_ratio(
    left: Mapping[str, Fraction],
    right: Mapping[str, Fraction],
) -> str:
    return format_dimensionality(
        _combine_dimensions(
            right,
            left,
            -1,
        )
    )


def _known_binary_scalar_result(
    operation: BinaryOperation,
    left: Scalar | None,
    right: Scalar | None,
) -> Scalar | None:
    if left is None or right is None:
        return None

    return _require_real_scalar(
        _apply_binary_operation(
            operation,
            _require_real_scalar(left),
            _require_real_scalar(right),
        )
    )


def _known_unary_scalar_result(
    operation: UnaryOperation,
    operand: Scalar | None,
) -> Scalar | None:
    if operand is None:
        return None

    return _require_real_scalar(
        _apply_unary_operation(
            operation,
            _require_real_scalar(operand),
        )
    )


def eval_dimension_expr(
    expression: str,
    variable_map: Mapping[
        str,
        DimensionValue,
    ],
) -> DimensionValue:
    return _eval_dimension_ast(
        _parse_expression(expression),
        variable_map,
    )


def _eval_dimension_ast(
    node: ast.expr,
    variable_map: Mapping[
        str,
        DimensionValue,
    ],
) -> DimensionValue:
    if isinstance(node, ast.Constant):
        return DimensionValue(
            {},
            _require_real_scalar(node.value),
        )

    if isinstance(node, ast.Name):
        if node.id not in variable_map:
            raise UserInputError(
                f"Unknown variable: {node.id}"
            )

        return variable_map[node.id]

    if isinstance(node, ast.UnaryOp):
        operation = ALLOWED_UNARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise UserInputError(
                "This unary operator is not allowed."
            )

        operand = _eval_dimension_ast(
            node.operand,
            variable_map,
        )

        return DimensionValue(
            operand.dimensions,
            _known_unary_scalar_result(
                operation,
                operand.scalar,
            ),
        )

    if isinstance(node, ast.BinOp):
        left = _eval_dimension_ast(
            node.left,
            variable_map,
        )

        right = _eval_dimension_ast(
            node.right,
            variable_map,
        )

        if isinstance(node.op, ast.Pow):
            if right.dimensions or right.scalar is None:
                raise UserInputError(
                    "A dimensional exponent must be "
                    "a known dimensionless number."
                )

            exponent = _canonical_fraction(
                right.scalar
            )

            return DimensionValue(
                _scale_dimensions(
                    left.dimensions,
                    exponent,
                ),
                _known_binary_scalar_result(
                    ALLOWED_BINARY_OPERATORS[ast.Pow],
                    left.scalar,
                    _fraction_as_scalar(exponent),
                ),
            )

        operation = ALLOWED_BINARY_OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise UserInputError(
                "This operator is not allowed."
            )

        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left.dimensions != right.dimensions:
                raise UserInputError(
                    "Addition and subtraction "
                    "require matching dimensions."
                )

            return DimensionValue(
                left.dimensions,
                _known_binary_scalar_result(
                    operation,
                    left.scalar,
                    right.scalar,
                ),
            )

        if isinstance(node.op, (ast.Mult, ast.Div)):
            right_factor = (
                1
                if isinstance(node.op, ast.Mult)
                else -1
            )

            return DimensionValue(
                _combine_dimensions(
                    left.dimensions,
                    right.dimensions,
                    right_factor,
                ),
                _known_binary_scalar_result(
                    operation,
                    left.scalar,
                    right.scalar,
                ),
            )

        raise UserInputError(
            "This operator is not allowed."
        )

    raise UserInputError(
        "Invalid or unsupported expression."
    )


def _is_blank(value: Any) -> bool:
    try:
        if bool(pd.isna(value)):
            return True

    except (
        TypeError,
        ValueError,
    ):
        pass

    return not str(value).strip()


def _validated_variable_name(
    raw_name: Any,
    row_number: int,
    existing_names: Collection[str],
) -> str:
    if _is_blank(raw_name):
        raise UserInputError(
            f"Variable name is required "
            f"in row {row_number}."
        )

    name = str(raw_name).strip()

    if (
        not name.isidentifier()
        or keyword.iskeyword(name)
    ):
        raise UserInputError(
            f"Invalid variable name "
            f"in row {row_number}: {name}"
        )

    if name in existing_names:
        raise UserInputError(
            f"Duplicate variable name: {name}"
        )

    return name


def _numeric_table_value(
    raw_value: Any,
    name: str,
    row_number: int,
) -> float:
    if isinstance(raw_value, bool):
        raise UserInputError(
            f"Value for variable {name} "
            f"in row {row_number} "
            f"must be a finite number."
        )

    try:
        value = float(raw_value)

    except (
        TypeError,
        ValueError,
    ):
        raise UserInputError(
            f"Invalid numeric value for "
            f"variable {name} "
            f"in row {row_number}."
        ) from None

    if not math.isfinite(value):
        raise UserInputError(
            f"Value for variable {name} "
            f"in row {row_number} "
            f"must be finite."
        )

    return value


def _quantity_from_text(
    magnitude: Scalar,
    unit_text: str,
    *,
    context: str,
) -> CalculationQuantity:
    try:
        return make_quantity(
            magnitude,
            unit_text,
        )

    except (
        PintError,
        TypeError,
        ValueError,
    ) as error:
        raise UserInputError(
            f"Invalid unit for "
            f"{context}: {unit_text}"
        ) from error


def _unit_from_text(
    unit_text: str,
    *,
    context: str,
) -> Unit:
    try:
        return parse_unit(unit_text)

    except (
        PintError,
        TypeError,
        ValueError,
    ) as error:
        raise UserInputError(
            f"Invalid unit for "
            f"{context}: {unit_text}"
        ) from error


def build_calculation_variables(
    dataframe: pd.DataFrame,
) -> dict[str, CalculationValue]:
    variable_map: dict[
        str,
        CalculationValue,
    ] = {}

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        raw_name: Any = row.get(
            "Variable",
            "",
        )

        raw_value: Any = row.get(
            "Value",
            "",
        )

        raw_unit: Any = row.get(
            "Unit",
            "",
        )

        name_is_blank = _is_blank(raw_name)
        value_is_blank = _is_blank(raw_value)
        unit_is_blank = _is_blank(raw_unit)

        if (
            name_is_blank
            and value_is_blank
            and unit_is_blank
        ):
            continue

        name = _validated_variable_name(
            raw_name,
            row_number,
            variable_map,
        )

        if value_is_blank:
            raise UserInputError(
                f"Value is required for "
                f"variable: {name}"
            )

        value = _numeric_table_value(
            raw_value,
            name,
            row_number,
        )

        if unit_is_blank:
            variable_map[name] = value

        else:
            unit_text = normalize_text(
                str(raw_unit)
            )

            variable_map[name] = (
                _quantity_from_text(
                    value,
                    unit_text,
                    context=f"variable {name}",
                )
            )

    return variable_map


def build_dimension_variables(dataframe: pd.DataFrame) -> dict[str, DimensionValue]:
    variable_map: dict[
        str,
        DimensionValue,
    ] = {}

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        raw_name: Any = row.get(
            "Variable",
            "",
        )

        raw_unit: Any = row.get(
            "Unit",
            "",
        )

        name_is_blank = _is_blank(raw_name)
        unit_is_blank = _is_blank(raw_unit)

        if (
            name_is_blank
            and unit_is_blank
        ):
            continue

        name = _validated_variable_name(
            raw_name,
            row_number,
            variable_map,
        )

        if unit_is_blank:
            raise UserInputError(
                f"Unit is required for variable "
                f"{name}; use 1 for dimensionless."
            )

        unit_text = normalize_text(
            str(raw_unit)
        )

        unit = _unit_from_text(
            unit_text,
            context=f"variable {name}",
        )

        variable_map[name] = DimensionValue(
            _dimension_map(unit)
        )

    return variable_map


def _as_quantity(value: CalculationValue) -> CalculationQuantity:
    return (
        value
        if isinstance(value, Quantity)
        else make_quantity(value)
    )


def _format_calculation_success(
    heading: str,
    displayed_result: CalculationQuantity,
    original_result: CalculationQuantity,
    dimensions: str,
    quantity_type: str,
) -> str:
    si_result = original_result.to_base_units()

    return (
        f"{heading}\n\n"
        "Result\n"
        f"{format(displayed_result, '~P')}\n\n"
        "SI form\n"
        f"{format(si_result, '~P')}\n\n"
        "Dimensions\n"
        f"{dimensions}\n\n"
        "Compatible quantity dimension:\n"
        f"{quantity_type}"
    )


def check_calculation(
    expression: str,
    variable_map: Mapping[
        str,
        CalculationValue,
    ],
    expected_unit: str,
) -> str:
    result = _as_quantity(
        eval_expr(
            expression,
            variable_map,
        )
    )

    actual_dimension_map = _dimension_map(
        result
    )

    actual_dimensions = format_dimensionality(
        actual_dimension_map
    )

    quantity_type = _compatible_dimension_name(
        actual_dimension_map
    )

    expected_unit_text = normalize_text(
        expected_unit
    )

    if not expected_unit_text:
        return _format_calculation_success(
            "✓ CALCULATION COMPLETE",
            result,
            result,
            actual_dimensions,
            quantity_type,
        )

    expected_unit_object = _unit_from_text(
        expected_unit_text,
        context="expected output",
    )

    expected_dimension_map = _dimension_map(
        expected_unit_object
    )

    if (
        actual_dimension_map
        != expected_dimension_map
    ):
        return (
            "✗ DIMENSIONAL MISMATCH\n\n"
            "Expression produces\n"
            f"{actual_dimensions}\n\n"
            "Expected\n"
            f"{format_dimensionality(expected_dimension_map)}\n\n"
            "Expression quantity type\n"
            f"{quantity_type}"
        )

    try:
        converted_result = result.to(
            expected_unit_object
        )

    except PintError as error:
        raise UserInputError(
            str(error)
        ) from None

    return _format_calculation_success(
        "✓ DIMENSIONALLY CONSISTENT",
        converted_result,
        result,
        actual_dimensions,
        quantity_type,
    )


def _split_equation(equation: str) -> tuple[str, str]:
    if not equation.strip():
        raise UserInputError(
            "Equation cannot be empty."
        )

    if equation.count("=") != 1:
        raise UserInputError(
            "The equation must contain "
            "exactly one '=' sign."
        )

    left_expression, right_expression = (
        expression.strip()
        for expression in equation.split(
            "=",
            maxsplit=1,
        )
    )

    if not left_expression:
        raise UserInputError(
            "The left-hand side cannot be empty."
        )

    if not right_expression:
        raise UserInputError(
            "The right-hand side cannot be empty."
        )

    return (
        left_expression,
        right_expression,
    )


def _format_dimension_side(
    label: str,
    expression: str,
    result: DimensionValue,
) -> str:
    return (
        f"{label}\n"
        f"{expression}\n"
        f"Dimensions: "
        f"{format_dimensionality(result.dimensions)}\n"
        f"Quantity type: "
        f"{_compatible_dimension_name(result.dimensions)}"
    )


def check_dimensions(
    equation: str,
    variable_map: Mapping[
        str,
        DimensionValue,
    ],
) -> str:
    left_expression, right_expression = (
        _split_equation(equation)
    )

    left_result = eval_dimension_expr(
        left_expression,
        variable_map,
    )

    right_result = eval_dimension_expr(
        right_expression,
        variable_map,
    )

    left_summary = _format_dimension_side(
        "Left-hand side",
        left_expression,
        left_result,
    )

    right_summary = _format_dimension_side(
        "Right-hand side",
        right_expression,
        right_result,
    )

    if (
        left_result.dimensions
        == right_result.dimensions
    ):
        return (
            "✓ DIMENSIONALLY CONSISTENT\n\n"
            f"{left_summary}\n\n"
            f"{right_summary}\n\n"
            "Both sides have the same "
            "physical dimensions."
        )

    ratio = dimensional_ratio(
        left_result.dimensions,
        right_result.dimensions,
    )

    return (
        "✗ DIMENSIONAL MISMATCH\n\n"
        f"{left_summary}\n\n"
        f"{right_summary}\n\n"
        "Right / left dimensional ratio\n"
        f"{ratio}\n\n"
        "A valid physical equation requires "
        "this ratio to be dimensionless."
    )


def _show_result(output: str) -> None:
    st.write("Result:")
    st.code(output)


def _show_unexpected_error(error: Exception) -> None:
    LOGGER.exception(
        "Unexpected UnitGuard error",
        exc_info=error,
    )

    _show_result(
        "ERROR\n\n"
        "An unexpected internal error occurred."
    )


def _run_ui_action(
    action: Callable[[], str],
) -> bool:
    try:
        output = action()

    except UserInputError as error:
        output = f"ERROR\n\n{error}"

    except Exception as error:
        _show_unexpected_error(error)
        return False

    _show_result(output)
    return True


def render_calculation_checker() -> None:
    st.subheader(
        "Calculation Checker"
    )

    st.write(
        "Enter an engineering expression, "
        "numerical values and units."
    )

    expression = (
        st.text_input(
            "Expression",
            value="F / A",
            key="calculation_expression",
        )
        or ""
    )

    default_table = pd.DataFrame(
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

    calculation_table = cast(
        pd.DataFrame,
        st.data_editor(
            default_table,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="calculation_variable_table",
            column_config={
                "Variable":
                    st.column_config.TextColumn(
                        "Variable"
                    ),
                "Value":
                    st.column_config.NumberColumn(
                        "Value"
                    ),
                "Unit":
                    st.column_config.TextColumn(
                        "Unit"
                    ),
            },
        ),
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
        if not _run_ui_action(
            lambda: check_calculation(
                expression,
                build_calculation_variables(
                    calculation_table
                ),
                expected_unit,
            )
        ):
            return


def render_dimensional_checker() -> None:
    st.subheader(
        "Dimensional Checker"
    )

    st.write(
        "Enter a complete equation and assign "
        "a unit to each variable. Numerical "
        "values are not required."
    )

    equation = (
        st.text_input(
            "Equation",
            value="F = m * a",
            key="dimension_equation",
        )
        or ""
    )

    default_table = pd.DataFrame(
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

    dimension_table = cast(
        pd.DataFrame,
        st.data_editor(
            default_table,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="dimension_variable_table",
            column_config={
                "Variable":
                    st.column_config.TextColumn(
                        "Variable"
                    ),
                "Unit":
                    st.column_config.TextColumn(
                        "Unit"
                    ),
            },
        ),
    )

    if st.button(
        "CHECK DIMENSIONS",
        key="check_dimensions_button",
    ):
        if not _run_ui_action(
            lambda: check_dimensions(
                equation,
                build_dimension_variables(
                    dimension_table
                ),
            )
        ):
            return

    st.caption(
        "Dimensional consistency is necessary, "
        "but it does not prove that an equation "
        "is physically correct."
    )


def display_unit_label(unit_text: str) -> str:
    display_text = unit_text

    for old, new in _UNIT_LABEL_REPLACEMENTS:
        display_text = display_text.replace(
            old,
            new,
        )

    return display_text


def convert_quantity(
    value: Scalar,
    from_unit: str,
    to_unit: str,
) -> tuple[
    CalculationQuantity,
    CalculationQuantity,
]:
    source_unit_text = normalize_text(
        from_unit
    )

    target_unit_text = normalize_text(
        to_unit
    )

    if not source_unit_text:
        raise UserInputError(
            "A source unit is required."
        )

    if not target_unit_text:
        raise UserInputError(
            "A target unit is required."
        )

    source_quantity = _quantity_from_text(
        _require_real_scalar(value),
        source_unit_text,
        context="source unit",
    )

    target_unit = _unit_from_text(
        target_unit_text,
        context="target unit",
    )

    source_dimensions = _dimension_map(
        source_quantity
    )

    target_dimensions = _dimension_map(
        target_unit
    )

    if (
        source_dimensions
        != target_dimensions
    ):
        raise UserInputError(
            "Incompatible units: "
            f"{from_unit} has dimensions "
            f"{format_dimensionality(source_dimensions)}, "
            f"while {to_unit} has dimensions "
            f"{format_dimensionality(target_dimensions)}."
        )

    try:
        converted_quantity = (
            source_quantity.to(target_unit)
        )

    except PintError as error:
        raise UserInputError(
            str(error)
        ) from None

    return (
        source_quantity,
        converted_quantity,
    )


def check_unit_conversion(
    value: Scalar,
    from_unit: str,
    to_unit: str,
) -> str:
    source_quantity, converted_quantity = (
        convert_quantity(
            value,
            from_unit,
            to_unit,
        )
    )

    dimensions = _dimension_map(
        source_quantity
    )

    base_quantity = (
        source_quantity.to_base_units()
    )

    return (
        "✓ CONVERSION COMPLETE\n\n"
        "Input\n"
        f"{format(source_quantity, '~P')}\n\n"
        "Converted result\n"
        f"{format(converted_quantity, '~P')}\n\n"
        "SI / base-unit form\n"
        f"{format(base_quantity, '~P')}\n\n"
        "Dimensions\n"
        f"{format_dimensionality(dimensions)}\n\n"
        "Compatible quantity dimension:\n"
        f"{_compatible_dimension_name(dimensions)}"
    )


def render_unit_converter() -> None:
    st.subheader(
        "Engineering Unit Converter"
    )

    st.write(
        "Convert common engineering units "
        "or enter a custom unit expression."
    )

    entry_mode = cast(
        str,
        st.radio(
            "Unit selection mode",
            (
                "Common engineering units",
                "Custom units",
            ),
            horizontal=True,
            key="converter_entry_mode",
        ),
    )

    value = cast(
        Scalar,
        st.number_input(
            "Value",
            value=1.0,
            key="converter_value",
        ),
    )

    if (
        entry_mode
        == "Common engineering units"
    ):
        category = cast(
            str,
            st.selectbox(
                "Quantity category",
                tuple(UNIT_CATEGORIES),
                key="converter_category",
            ),
        )

        unit_options = (
            UNIT_CATEGORIES[category]
        )

        left_column, right_column = (
            st.columns(2)
        )

        with left_column:
            source_unit = cast(
                str,
                st.selectbox(
                    "From unit",
                    unit_options,
                    index=0,
                    format_func=display_unit_label,
                    key="converter_source_common",
                ),
            )

        with right_column:
            target_unit = cast(
                str,
                st.selectbox(
                    "To unit",
                    unit_options,
                    index=(
                        1
                        if len(unit_options) > 1
                        else 0
                    ),
                    format_func=display_unit_label,
                    key="converter_target_common",
                ),
            )

    else:
        left_column, right_column = (
            st.columns(2)
        )

        with left_column:
            source_unit = (
                st.text_input(
                    "From unit",
                    value="kPa",
                    key="converter_source_custom",
                )
                or ""
            )

        with right_column:
            target_unit = (
                st.text_input(
                    "To unit",
                    value="psi",
                    key="converter_target_custom",
                )
                or ""
            )

        st.caption(
            "Examples: mm², N·m, kg·m⁻³, "
            "L/min, cP, rpm, degC."
        )

    if st.button(
        "CONVERT",
        key="convert_units_button",
    ):
        if not _run_ui_action(
            lambda: check_unit_conversion(
                value,
                source_unit,
                target_unit,
            )
        ):
            return

    st.caption(
        "Common-unit mode keeps both units "
        "in the same engineering category. "
        "Custom mode checks dimensional "
        "compatibility only; equal dimensions "
        "do not always imply the same "
        "physical meaning."
    )


def main() -> None:
    st.set_page_config(
        page_title="UnitGuard",
        layout="wide",
    )

    st.title("UnitGuard")

    (
        calculation_tab,
        dimensional_tab,
        converter_tab,
    ) = st.tabs(
        [
            "Calculation Checker",
            "Dimensional Checker",
            "Unit Converter",
        ]
    )

    with calculation_tab:
        render_calculation_checker()

    with dimensional_tab:
        render_dimensional_checker()

    with converter_tab:
        render_unit_converter()


def _launch_streamlit() -> NoReturn:
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

    raise SystemExit(
        completed_process.returncode
    )


def _run_entrypoint() -> None:
    if st_runtime.exists():
        main()
    else:
        _launch_streamlit()


if __name__ == "__main__":
    _run_entrypoint()
