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

## Test

```powershell
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```
