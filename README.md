# engineering-calculation-error-detector---SYNCS-HACK-2026

A web-based tool that detects unit, formula, and calculation errors in engineering solutions.

## Requirements

- Python 3.11 or newer (verified with Python 3.14)

## Setup

From PowerShell in the project directory:

```powershell
py -3.14 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Run

Run the app directly:

```powershell
& '.\.venv\Scripts\python.exe' Main.py
```

Any arguments after `Main.py` are forwarded to Streamlit. The equivalent standard
Streamlit command is:

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run Main.py
```

## Supported expressions and limits

- Expressions support `+`, `-`, `*`, `/`, `**`, parentheses, and unary `+` and `-`.
- `^` is normalized to exponentiation. Engineering symbols `×`, `·`, `⋅`, `÷`,
  and `−` are normalized to their Python equivalents.
- Superscript integer exponents may use the full `⁰`-`⁹` digit set with an
  optional leading `⁺` or `⁻`.
- Function calls, attribute access, and subscripting are not allowed.
- Expressions are limited to 1,000 characters and 200 AST nodes, and absolute
  exponent values must not exceed 1,000.
- In Calculation Checker, a blank unit is dimensionless. In Dimensional Checker,
  enter `1` for a dimensionless variable.

## Unit Converter

The Unit Converter provides two input modes:

- **Common engineering units** offers 177 units across 33 engineering categories.
  Both selections stay within the chosen category, including separate choices for
  absolute temperature and temperature differences.
- **Custom units** accepts Pint-compatible unit expressions such as `mm²`, `N·m`,
  `kg·m⁻³`, `L/min`, `cP`, `rpm`, and `degC`. Source and target units must have
  matching physical dimensions.

Affine temperature conversions such as `degC` to `degF` are supported. The result
also reports its SI/base-unit form, dimensions, and compatible quantity dimension.
Dimensional compatibility alone does not guarantee that two units represent the
same engineering concept, so common-unit mode should be preferred when a matching
category is available.

## Test

```powershell
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```
