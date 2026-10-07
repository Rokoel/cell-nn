"""Create distinct run folders even on platforms with coarse clock resolution."""
from datetime import datetime
from pathlib import Path
import tempfile


def new_run_directory(root, prefix):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'{prefix}-{datetime.now():%Y%m%d-%H%M%S-%f}-', dir=root))
