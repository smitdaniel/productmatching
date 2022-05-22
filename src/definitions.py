import os
from pathlib import Path

definitions_path: Path = Path(os.path.dirname(__file__))
root_path: Path = definitions_path.parent
data_path: Path = root_path / "data"
datafile_path: Path = data_path / "source.xlsx"
distance_cache_path: Path = data_path / "distance.json"
log_path: Path = root_path / "log"
config_path: Path = root_path / "config/config.yaml"
out_path: Path = root_path / "out"
