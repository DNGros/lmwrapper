"""Tests to verify that any examples given in user-facing documentation
will execute."""
import re
from pathlib import Path
import pytest

from lmwrapper.secret_manage import SecretEnvVar
SKIP_OPENAI = not SecretEnvVar("OPENAI_API_KEY").is_readable()

cur_file = Path(__file__).parent.absolute()


def extract_code_blocks(file):
    with open(file) as f:
        content = f.read()
    code_blocks = re.findall(r'```python\r?\n(.*?)```', content, re.DOTALL)
    return code_blocks


@pytest.mark.parametrize(
    'code',
    extract_code_blocks(cur_file / '../README.md')
)
@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_readme_code(code):
    print("### CODE BLOCK")
    print(code)
    print("### OUTPUT")
    exec(code)
