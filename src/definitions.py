import os
from pathlib import Path

definitions_path: Path = Path(os.path.dirname(__file__))
root_path: Path = definitions_path.parent
data_path: Path = root_path / "data"
datafile_path: Path = data_path / "source.xlsx"
log_path: Path = root_path / "log"
