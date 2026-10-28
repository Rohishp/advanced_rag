"""Run every offline test: python tests/run_all.py (no API keys or network needed)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
failed = []
for test in sorted(HERE.glob("test_*.py")):
    p = subprocess.run([sys.executable, str(test)], cwd=HERE.parent, capture_output=True,
                       text=True)
    print(p.stdout, end="")
    if p.returncode != 0:
        failed.append(test.name)
        print(p.stderr[-3000:])
print("ALL TEST FILES PASSED" if not failed else f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
