@echo off
REM Delete existing combined.py if present
if exist combined.py del combined.py

REM Loop over all .py files in the directory
for %%f in (*.py) do (
    REM Skip combined.py if it exists in the loop
    if /I not "%%f"=="combined.py" (
        echo # %%f >> combined.py
        type "%%f" >> combined.py
        echo. >> combined.py
    )
)

echo All .py files have been combined into combined.py.
