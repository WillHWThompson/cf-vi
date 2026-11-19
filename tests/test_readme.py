import re
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_readme_example_runs():
    """The README's first python block is examples/regression.py verbatim, and it runs in under 60 s."""
    code = re.search(r"```python\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S).group(1)
    assert code == (ROOT / "examples" / "regression.py").read_text(encoding="utf-8")
    start = time.perf_counter()
    exec(code, {})
    assert time.perf_counter() - start < 60
