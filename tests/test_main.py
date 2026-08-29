import runpy
import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
from pint.registry import Quantity
from streamlit.testing.v1 import AppTest

from Main import (
    build_variables,
    compatible_quantity_name,
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

    def test_none_becomes_empty_text(self) -> None:
        self.assertEqual(normalize_text(None), "")


class ExpressionEvaluatorTests(unittest.TestCase):
    def test_default_pressure_expression(self) -> None:
        variable_map = {
            "F": make_quantity(10, "kN"),
            "A": make_quantity(200, "mm**2"),
        }

        result = cast(Quantity[Any], eval_expr("F / A", variable_map))

        self.assertAlmostEqual(result.to("MPa").magnitude, 50)

    def test_numeric_binary_power_and_unary_operations(self) -> None:
        self.assertEqual(eval_expr("2 + 3 * 4", {}), 14)
        self.assertEqual(eval_expr("2**3", {}), 8)
        self.assertEqual(eval_expr("-5", {}), -5)

    def test_rejects_non_real_constants(self) -> None:
        expressions = ["True", "'x'", "b'x'", "None", "...", "1j"]

        for expression in expressions:
            with self.assertRaises(ValueError, msg=f"expression={expression!r}"):
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
            with self.assertRaises(ValueError, msg=f"expression={expression!r}"):
                eval_expr(expression, {"value": 1})

    def test_rejects_unknown_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown variable"):
            eval_expr("missing", {})


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


class VariableTableTests(unittest.TestCase):
    def test_skips_empty_rows_and_treats_missing_unit_as_dimensionless(self) -> None:
        table = pd.DataFrame(
            [
                {"Variable": "x", "Value": 2.0, "Unit": None},
                {"Variable": None, "Value": None, "Unit": None},
            ]
        )

        self.assertEqual(build_variables(table), {"x": 2.0})

    def test_supports_offset_temperature_units(self) -> None:
        table = pd.DataFrame(
            [{"Variable": "T", "Value": 25.0, "Unit": "degC"}]
        )

        temperature = cast(Quantity[Any], build_variables(table)["T"])

        self.assertAlmostEqual(temperature.to("degF").magnitude, 77)

    def test_rejects_value_without_variable_name(self) -> None:
        table = pd.DataFrame(
            [{"Variable": None, "Value": 1.0, "Unit": "m"}]
        )

        with self.assertRaisesRegex(ValueError, "Variable name is required"):
            build_variables(table)

    def test_rejects_invalid_and_keyword_variable_names(self) -> None:
        for name in ("not-valid", "for"):
            table = pd.DataFrame(
                [{"Variable": name, "Value": 1.0, "Unit": "m"}]
            )
            with self.assertRaisesRegex(
                ValueError,
                "Invalid variable name",
                msg=f"name={name!r}",
            ):
                build_variables(table)

    def test_rejects_duplicate_variable_names(self) -> None:
        table = pd.DataFrame(
            [
                {"Variable": "x", "Value": 1.0, "Unit": "m"},
                {"Variable": "x", "Value": 2.0, "Unit": "s"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Duplicate variable name"):
            build_variables(table)


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

    def test_default_calculation_is_consistent(self) -> None:
        app = self.make_app()
        self.assertEqual(len(app.exception), 0)

        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONALLY CONSISTENT", app.code[0].value)
        self.assertRegex(app.code[0].value, r"50(?:\.0+)? MPa")

    def test_streamlit_runtime_does_not_launch_a_child_process(self) -> None:
        with patch("subprocess.run") as run_process:
            app = self.make_app()

        self.assertEqual(len(app.exception), 0)
        run_process.assert_not_called()

    def test_dimension_mismatch_is_reported(self) -> None:
        app = self.make_app()
        app.text_input[1].set_value("N").run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("DIMENSIONAL MISMATCH", app.code[0].value)

    def test_unknown_variable_is_reported(self) -> None:
        app = self.make_app()
        app.text_input[0].set_value("missing").run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("ERROR", app.code[0].value)
        self.assertIn("Unknown variable", app.code[0].value)

    def test_blank_expected_unit_completes_calculation(self) -> None:
        app = self.make_app()
        app.text_input[1].set_value("").run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("CALCULATION COMPLETE", app.code[0].value)

    def test_empty_expression_is_reported(self) -> None:
        app = self.make_app()
        app.text_input[0].set_value("").run()
        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Expression cannot be empty", app.code[0].value)


if __name__ == "__main__":
    unittest.main()
