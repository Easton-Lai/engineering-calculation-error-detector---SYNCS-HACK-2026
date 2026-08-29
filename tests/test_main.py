import math
import runpy
import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
from pint import Quantity
from streamlit.testing.v1 import AppTest

from Main import (
    MAX_ABS_EXPONENT,
    MAX_AST_NODES,
    MAX_EXPRESSION_LENGTH,
    MAX_INTEGER_BITS,
    UNIT_CATEGORIES,
    build_calculation_variables,
    build_dimension_variables,
    check_calculation,
    check_dimensions,
    check_unit_conversion,
    compatible_quantity_name,
    convert_quantity,
    display_unit_label,
    eval_expr,
    format_dimensions,
    make_quantity,
    normalize_text,
)


APP_PATH = Path(__file__).resolve().parents[1] / "Main.py"


class TextNormalizationTests(unittest.TestCase):
    def test_normalizes_engineering_symbols(self) -> None:
        self.assertEqual(
            normalize_text(" A² × B ÷ C − 1 "),
            "A**2 * B / C - 1",
        )

    def test_normalizes_complete_and_signed_superscript_exponents(self) -> None:
        self.assertEqual(normalize_text("x²³"), "x**23")
        self.assertEqual(normalize_text("x⁻²"), "x**-2")
        self.assertEqual(normalize_text("kg·m⁻³"), "kg*m**-3")

    def test_none_becomes_empty_text(self) -> None:
        self.assertEqual(normalize_text(None), "")


class ExpressionEvaluatorTests(unittest.TestCase):
    def test_default_pressure_expression(self) -> None:
        variable_map = {
            "F": make_quantity(10, "kN"),
            "A": make_quantity(200, "mm**2"),
        }

        result = eval_expr("F / A", variable_map)

        self.assertIsInstance(result, Quantity)
        self.assertAlmostEqual(result.to("MPa").magnitude, 50)

    def test_numeric_binary_power_and_unary_operations(self) -> None:
        self.assertEqual(eval_expr("2 + 3 * 4", {}), 14)
        self.assertEqual(eval_expr("2**3", {}), 8)
        self.assertEqual(eval_expr("-5", {}), -5)

    def test_evaluates_complete_and_negative_superscripts(self) -> None:
        self.assertEqual(eval_expr("x²³", {"x": 2}), 2**23)
        self.assertAlmostEqual(eval_expr("x⁻²", {"x": 2}), 0.25)

    def test_supports_pure_constant_expressions(self) -> None:
        self.assertEqual(eval_expr("2 + 3", {}), 5)

    def test_rejects_non_real_constants(self) -> None:
        expressions = ["True", "'x'", "b'x'", "None", "...", "1j"]

        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_expr(expression, {})

    def test_rejects_unsafe_ast_nodes(self) -> None:
        expressions = [
            "abs(1)",
            "value.real",
            "[1][0]",
            "5 // 2",
            "5 % 2",
            "1 << 2",
            "~1",
        ]

        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_expr(expression, {"value": 1})

    def test_rejects_unknown_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown variable"):
            eval_expr("missing", {})

    def test_rejects_invalid_variable_values(self) -> None:
        invalid_values = [
            True,
            1 + 2j,
            object(),
            float("nan"),
            float("inf"),
            float("-inf"),
        ]

        for index, value in enumerate(invalid_values):
            with self.subTest(index=index, value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    eval_expr(
                        "value",
                        cast(Any, {"value": value}),
                    )

    def test_make_quantity_rejects_non_finite_magnitudes(self) -> None:
        for magnitude in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(ValueError):
                    make_quantity(magnitude, "m")

    def test_rejects_non_finite_or_complex_operation_results(self) -> None:
        cases = [
            ("1e309", {}),
            ("value * value", {"value": 1e308}),
            ("value**0.5", {"value": make_quantity(-1, "m**2")}),
        ]

        for expression, variable_map in cases:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_expr(expression, variable_map)

    def test_rejects_expression_over_length_limit(self) -> None:
        expression = "1" * (MAX_EXPRESSION_LENGTH + 1)

        with self.assertRaises(ValueError):
            eval_expr(expression, {})

    def test_rejects_expression_over_ast_node_limit(self) -> None:
        expression = "+".join("1" for _ in range(MAX_AST_NODES + 1))
        self.assertLessEqual(len(expression), MAX_EXPRESSION_LENGTH)

        with self.assertRaises(ValueError):
            eval_expr(expression, {})

    def test_rejects_exponents_outside_limit(self) -> None:
        expressions = [
            f"1**{MAX_ABS_EXPONENT + 1}",
            f"1**-{MAX_ABS_EXPONENT + 1}",
        ]

        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_expr(expression, {})

    def test_rejects_integer_results_over_bit_limit(self) -> None:
        factor_count = MAX_INTEGER_BITS // MAX_ABS_EXPONENT + 1
        expression = " * ".join(
            f"(2**{MAX_ABS_EXPONENT})" for _ in range(factor_count)
        )

        with self.assertRaises(ValueError):
            eval_expr(expression, {})

    def test_normalizes_equivalent_floating_point_exponents(self) -> None:
        variable_map = {"x": make_quantity(1, "m")}

        direct = eval_expr("x**0.3", variable_map)
        computed = eval_expr("x**(0.1 + 0.2)", variable_map)

        self.assertIsInstance(direct, Quantity)
        self.assertIsInstance(computed, Quantity)
        self.assertEqual(direct.dimensionality, computed.dimensionality)


class QuantityTests(unittest.TestCase):
    def test_formats_pressure_dimensions(self) -> None:
        pressure = make_quantity(1, "MPa")

        self.assertEqual(format_dimensions(pressure), "M L^-1 T^-2")
        self.assertEqual(compatible_quantity_name(pressure), "Pressure / Stress")

    def test_distinguishes_force_from_energy_and_torque(self) -> None:
        self.assertEqual(compatible_quantity_name(make_quantity(1, "N")), "Force")
        self.assertEqual(
            compatible_quantity_name(make_quantity(1, "J")),
            "Energy / Torque",
        )
        self.assertEqual(
            compatible_quantity_name(make_quantity(1, "N*m")),
            "Energy / Torque",
        )

    def test_formats_dimensionless_quantity(self) -> None:
        self.assertEqual(format_dimensions(make_quantity(2)), "dimensionless")
        self.assertEqual(
            compatible_quantity_name(make_quantity(2)),
            "Dimensionless",
        )


class CalculationVariableBuilderTests(unittest.TestCase):
    def test_skips_empty_rows_and_treats_missing_unit_as_dimensionless(self) -> None:
        table = pd.DataFrame(
            [
                {"Variable": "x", "Value": 2.0, "Unit": None},
                {"Variable": None, "Value": None, "Unit": None},
            ]
        )

        self.assertEqual(build_calculation_variables(table), {"x": 2.0})

    def test_empty_table_supports_pure_constant_expression(self) -> None:
        table = pd.DataFrame(columns=["Variable", "Value", "Unit"])

        self.assertEqual(build_calculation_variables(table), {})

    def test_supports_offset_temperature_units(self) -> None:
        table = pd.DataFrame(
            [{"Variable": "T", "Value": 25.0, "Unit": "degC"}]
        )

        temperature = build_calculation_variables(table)["T"]

        self.assertIsInstance(temperature, Quantity)
        self.assertAlmostEqual(temperature.to("degF").magnitude, 77)

    def test_rejects_value_without_variable_name(self) -> None:
        table = pd.DataFrame(
            [{"Variable": None, "Value": 1.0, "Unit": "m"}]
        )

        with self.assertRaisesRegex(ValueError, "Variable name"):
            build_calculation_variables(table)

    def test_rejects_invalid_and_keyword_variable_names(self) -> None:
        for name in ("not-valid", "for"):
            with self.subTest(name=name):
                table = pd.DataFrame(
                    [{"Variable": name, "Value": 1.0, "Unit": "m"}]
                )
                with self.assertRaises(ValueError):
                    build_calculation_variables(table)

    def test_rejects_duplicate_variable_names(self) -> None:
        table = pd.DataFrame(
            [
                {"Variable": "x", "Value": 1.0, "Unit": "m"},
                {"Variable": "x", "Value": 2.0, "Unit": "s"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_calculation_variables(table)

    def test_rejects_missing_or_invalid_values_and_units(self) -> None:
        rows = [
            {"Variable": "x", "Value": None, "Unit": "m"},
            {"Variable": "x", "Value": "not-a-number", "Unit": "m"},
            {"Variable": "x", "Value": 1.0, "Unit": "not_a_real_unit"},
        ]

        for row in rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    build_calculation_variables(pd.DataFrame([row]))

    def test_rejects_boolean_and_non_finite_values(self) -> None:
        values = [True, "nan", "inf", "-inf", "1e309"]

        for value in values:
            with self.subTest(value=value):
                table = pd.DataFrame(
                    [{"Variable": "x", "Value": value, "Unit": "m"}]
                )
                with self.assertRaises(ValueError):
                    build_calculation_variables(table)


class DimensionVariableBuilderTests(unittest.TestCase):
    def test_builds_default_dimension_variables(self) -> None:
        table = pd.DataFrame(
            [
                {"Variable": "F", "Unit": "N"},
                {"Variable": "m", "Unit": "kg"},
                {"Variable": "a", "Unit": "m/s**2"},
            ]
        )

        variable_map = build_dimension_variables(table)

        self.assertEqual(set(variable_map), {"F", "m", "a"})

    def test_skips_empty_rows_and_allows_empty_table(self) -> None:
        table = pd.DataFrame(
            [{"Variable": None, "Unit": None}],
            columns=["Variable", "Unit"],
        )

        self.assertEqual(build_dimension_variables(table), {})

    def test_accepts_explicit_dimensionless_unit(self) -> None:
        table = pd.DataFrame([{"Variable": "n", "Unit": "1"}])
        variable_map = build_dimension_variables(table)

        output = check_dimensions("n = 1", variable_map)

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_rejects_missing_name_or_unit(self) -> None:
        rows = [
            {"Variable": None, "Unit": "m"},
            {"Variable": "x", "Unit": None},
        ]

        for row in rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    build_dimension_variables(pd.DataFrame([row]))

    def test_rejects_invalid_keyword_duplicate_and_unknown_units(self) -> None:
        tables = [
            pd.DataFrame([{"Variable": "not-valid", "Unit": "m"}]),
            pd.DataFrame([{"Variable": "for", "Unit": "m"}]),
            pd.DataFrame([{"Variable": "x", "Unit": "not_a_real_unit"}]),
            pd.DataFrame(
                [
                    {"Variable": "x", "Unit": "m"},
                    {"Variable": "x", "Unit": "s"},
                ]
            ),
        ]

        for table in tables:
            with self.subTest(table=table.to_dict("records")):
                with self.assertRaises(ValueError):
                    build_dimension_variables(table)


class CalculationCheckerTests(unittest.TestCase):
    def test_default_calculation_is_consistent(self) -> None:
        variable_map = {
            "F": make_quantity(10, "kN"),
            "A": make_quantity(200, "mm**2"),
        }

        output = check_calculation("F / A", variable_map, "MPa")

        self.assertIn("DIMENSIONALLY CONSISTENT", output)
        self.assertRegex(output, r"50(?:\.0+)? MPa")

    def test_dimension_mismatch_is_reported(self) -> None:
        output = check_calculation(
            "F / A",
            {
                "F": make_quantity(10, "kN"),
                "A": make_quantity(200, "mm**2"),
            },
            "N",
        )

        self.assertIn("DIMENSIONAL MISMATCH", output)

    def test_blank_expected_unit_completes_calculation(self) -> None:
        output = check_calculation("2 + 3", {}, "")

        self.assertIn("CALCULATION COMPLETE", output)
        self.assertRegex(output, r"\b5\b")

    def test_equivalent_floating_exponent_converts_to_expected_unit(self) -> None:
        output = check_calculation(
            "x**(0.1 + 0.2)",
            {"x": make_quantity(1, "m")},
            "m**0.3",
        )

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_rejects_empty_unknown_and_invalid_expected_unit(self) -> None:
        cases = [
            ("", {}, ""),
            ("missing", {}, ""),
            ("2 + 3", {}, "not_a_real_unit"),
        ]

        for expression, variable_map, expected_unit in cases:
            with self.subTest(
                expression=expression,
                expected_unit=expected_unit,
            ):
                with self.assertRaises(ValueError):
                    check_calculation(expression, variable_map, expected_unit)


class DimensionCheckerTests(unittest.TestCase):
    @staticmethod
    def default_variables() -> dict:
        return build_dimension_variables(
            pd.DataFrame(
                [
                    {"Variable": "F", "Unit": "N"},
                    {"Variable": "m", "Unit": "kg"},
                    {"Variable": "a", "Unit": "m/s**2"},
                ]
            )
        )

    def test_default_equation_is_consistent(self) -> None:
        output = check_dimensions("F = m * a", self.default_variables())

        self.assertIn("DIMENSIONALLY CONSISTENT", output)
        self.assertIn("Force", output)

    def test_mismatch_reports_right_to_left_ratio(self) -> None:
        output = check_dimensions("F = m", self.default_variables())

        self.assertIn("DIMENSIONAL MISMATCH", output)
        self.assertRegex(output, r"L\^-1 T\^2")

    def test_dimension_analysis_does_not_depend_on_placeholder_magnitudes(self) -> None:
        variable_map = build_dimension_variables(
            pd.DataFrame(
                [
                    {"Variable": "k", "Unit": "1/m"},
                    {"Variable": "x", "Unit": "m"},
                    {"Variable": "y", "Unit": "m"},
                ]
            )
        )

        output = check_dimensions("k = 1 / (x - y)", variable_map)

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_offset_temperature_units_are_dimension_only(self) -> None:
        variable_map = build_dimension_variables(
            pd.DataFrame(
                [
                    {"Variable": "E", "Unit": "J"},
                    {"Variable": "mass", "Unit": "kg"},
                    {"Variable": "c", "Unit": "J/(kg*K)"},
                    {"Variable": "T", "Unit": "degC"},
                ]
            )
        )

        output = check_dimensions("E = mass * c * T", variable_map)

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_equivalent_floating_exponents_are_consistent(self) -> None:
        variable_map = build_dimension_variables(
            pd.DataFrame([{"Variable": "x", "Unit": "m"}])
        )

        output = check_dimensions(
            "x**0.3 = x**(0.1 + 0.2)",
            variable_map,
        )

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_supports_pure_constant_equations(self) -> None:
        output = check_dimensions("2 = 1 + 1", {})

        self.assertIn("DIMENSIONALLY CONSISTENT", output)

    def test_rejects_invalid_equation_shapes(self) -> None:
        equations = ["", "F", "F = m = a", "= m * a", "F ="]

        for equation in equations:
            with self.subTest(equation=equation):
                with self.assertRaises(ValueError):
                    check_dimensions(equation, self.default_variables())

    def test_rejects_unknown_variables(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown variable"):
            check_dimensions("F = missing", self.default_variables())


class UnitConverterTests(unittest.TestCase):
    def test_default_length_conversion(self) -> None:
        source, converted = convert_quantity(1.0, "mm", "cm")

        self.assertEqual(source.magnitude, 1.0)
        self.assertAlmostEqual(converted.magnitude, 0.1)

        _, density = convert_quantity(1.0, "kg·m⁻³", "g/L")
        self.assertAlmostEqual(density.magnitude, 1.0)

    def test_converts_offset_temperatures(self) -> None:
        _, converted = convert_quantity(0.0, "degC", "degF")

        self.assertAlmostEqual(converted.magnitude, 32.0)

    def test_formats_engineering_unit_labels(self) -> None:
        cases = {
            "delta_degC": "Δ°C",
            "delta_degF": "Δ°F",
            "mm**2": "mm²",
            "kg*m**4": "kg·m⁴",
            "N*m": "N·m",
            "milliohm": "mΩ",
            "kiloohm": "kΩ",
        }

        for unit_text, expected in cases.items():
            with self.subTest(unit_text=unit_text):
                self.assertEqual(display_unit_label(unit_text), expected)

    def test_all_catalog_units_parse_and_are_compatible_within_category(
        self,
    ) -> None:
        self.assertEqual(len(UNIT_CATEGORIES), 33)
        self.assertEqual(
            sum(len(units) for units in UNIT_CATEGORIES.values()),
            177,
        )

        for category, units in UNIT_CATEGORIES.items():
            reference_unit = units[0]

            for unit in units:
                with self.subTest(category=category, unit=unit):
                    _, converted = convert_quantity(
                        1.0,
                        unit,
                        reference_unit,
                    )
                    self.assertTrue(
                        math.isfinite(float(converted.magnitude))
                    )

    def test_rejects_blank_and_unknown_units(self) -> None:
        cases = [
            ("", "m", "A source unit is required."),
            ("m", "", "A target unit is required."),
            (
                "not_a_real_unit",
                "m",
                "Invalid unit for source unit",
            ),
            (
                "m",
                "not_a_real_unit",
                "Invalid unit for target unit",
            ),
        ]

        for source_unit, target_unit, message in cases:
            with self.subTest(
                source_unit=source_unit,
                target_unit=target_unit,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    convert_quantity(1.0, source_unit, target_unit)

    def test_rejects_dimensionally_incompatible_units(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Incompatible units: m .* while s ",
        ):
            convert_quantity(1.0, "m", "s")

    def test_rejects_non_real_or_non_finite_values(self) -> None:
        invalid_values = [
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
            complex(1, 2),
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    convert_quantity(
                        cast(Any, value),
                        "m",
                        "cm",
                    )

    def test_conversion_output_is_stable(self) -> None:
        self.assertEqual(
            check_unit_conversion(1.0, "mm", "cm"),
            (
                "✓ CONVERSION COMPLETE\n\n"
                "Input\n"
                "1.0 mm\n\n"
                "Converted result\n"
                "0.1 cm\n\n"
                "SI / base-unit form\n"
                "0.001 m\n\n"
                "Dimensions\n"
                "L\n\n"
                "Compatible quantity dimension:\n"
                "Length"
            ),
        )


class ApplicationEntrypointTests(unittest.TestCase):
    def test_plain_python_run_launches_streamlit_and_propagates_exit_code(
        self,
    ) -> None:
        python_executable = r"C:\Python\python.exe"
        forwarded_args = ["--server.headless", "true"]

        with (
            patch("streamlit.runtime.exists", return_value=False),
            patch("subprocess.run") as run_process,
            patch.object(sys, "argv", [str(APP_PATH), *forwarded_args]),
            patch.object(sys, "executable", python_executable),
            self.assertRaises(SystemExit) as raised,
        ):
            run_process.return_value.returncode = 7
            runpy.run_path(str(APP_PATH), run_name="__main__")

        self.assertEqual(raised.exception.code, 7)
        run_process.assert_called_once_with(
            [
                python_executable,
                "-m",
                "streamlit",
                "run",
                str(APP_PATH),
                *forwarded_args,
            ],
            check=False,
        )

    def test_plain_python_run_handles_keyboard_interrupt_without_traceback(
        self,
    ) -> None:
        with (
            patch("streamlit.runtime.exists", return_value=False),
            patch("subprocess.run", side_effect=KeyboardInterrupt),
            patch.object(sys, "argv", [str(APP_PATH)]),
            self.assertRaises(SystemExit) as raised,
        ):
            runpy.run_path(str(APP_PATH), run_name="__main__")

        self.assertEqual(raised.exception.code, 130)


class StreamlitAppTests(unittest.TestCase):
    @staticmethod
    def make_app() -> AppTest:
        return AppTest.from_file(APP_PATH, default_timeout=10).run()

    def test_renders_all_tool_tabs(self) -> None:
        app = self.make_app()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            [
                "Calculation Checker",
                "Dimensional Checker",
                "Unit Converter",
            ],
        )
        self.assertEqual(
            {button.key for button in app.button},
            {
                "check_calculation_button",
                "check_dimensions_button",
                "convert_units_button",
            },
        )

    def test_default_calculation_is_consistent(self) -> None:
        app = self.make_app()
        app.button("check_calculation_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONALLY CONSISTENT", app.code[0].value)
        self.assertRegex(app.code[0].value, r"50(?:\.0+)? MPa")

    def test_default_dimensional_equation_is_consistent(self) -> None:
        app = self.make_app()
        app.button("check_dimensions_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONALLY CONSISTENT", app.code[0].value)
        self.assertIn("Force", app.code[0].value)

    def test_streamlit_runtime_does_not_launch_a_child_process(self) -> None:
        with patch("subprocess.run") as run_process:
            app = self.make_app()

        self.assertEqual(len(app.exception), 0)
        run_process.assert_not_called()

    def test_calculation_dimension_mismatch_is_reported(self) -> None:
        app = self.make_app()
        app.text_input("calculation_expected_unit").set_value("N").run()
        app.button("check_calculation_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONAL MISMATCH", app.code[0].value)

    def test_unknown_calculation_variable_is_reported(self) -> None:
        app = self.make_app()
        app.text_input("calculation_expression").set_value("missing").run()
        app.button("check_calculation_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("ERROR", app.code[0].value)
        self.assertIn("Unknown variable", app.code[0].value)

    def test_blank_expected_unit_completes_calculation(self) -> None:
        app = self.make_app()
        app.text_input("calculation_expected_unit").set_value("").run()
        app.button("check_calculation_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CALCULATION COMPLETE", app.code[0].value)

    def test_empty_calculation_expression_is_reported(self) -> None:
        app = self.make_app()
        app.text_input("calculation_expression").set_value("").run()
        app.button("check_calculation_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("ERROR", app.code[0].value)
        self.assertIn("empty", app.code[0].value.lower())

    def test_dimensional_mismatch_is_reported(self) -> None:
        app = self.make_app()
        app.text_input("dimension_equation").set_value("F = m").run()
        app.button("check_dimensions_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONAL MISMATCH", app.code[0].value)
        self.assertRegex(app.code[0].value, r"L\^-1 T\^2")

    def test_malformed_dimensional_equation_is_reported(self) -> None:
        app = self.make_app()
        app.text_input("dimension_equation").set_value("F").run()
        app.button("check_dimensions_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("ERROR", app.code[0].value)
        self.assertIn("=", app.code[0].value)

    def test_default_common_unit_conversion(self) -> None:
        app = self.make_app()

        self.assertEqual(
            app.radio("converter_entry_mode").value,
            "Common engineering units",
        )
        self.assertEqual(app.selectbox("converter_category").value, "Length")
        self.assertEqual(app.selectbox("converter_source_common").value, "mm")
        self.assertEqual(app.selectbox("converter_target_common").value, "cm")

        app.button("convert_units_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CONVERSION COMPLETE", app.code[0].value)
        self.assertIn("1.0 mm", app.code[0].value)
        self.assertIn("0.1 cm", app.code[0].value)

    def test_custom_unit_conversion(self) -> None:
        app = self.make_app()
        app.radio("converter_entry_mode").set_value("Custom units").run()

        self.assertEqual(
            app.text_input("converter_source_custom").value,
            "kPa",
        )
        self.assertEqual(
            app.text_input("converter_target_custom").value,
            "psi",
        )

        app.button("convert_units_button").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CONVERSION COMPLETE", app.code[0].value)
        self.assertIn("1.0 kPa", app.code[0].value)
        self.assertIn("psi", app.code[0].value)


if __name__ == "__main__":
    unittest.main()
