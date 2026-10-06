import shutil
import uuid
from pathlib import Path

import pytest

LOCAL_TMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@pytest.fixture
def local_tmp():
    """Temporary directory inside the project (nothing is written outside the repository)."""
    path = LOCAL_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
