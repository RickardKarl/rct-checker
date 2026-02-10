import os
import pathlib

_default_data_dir = pathlib.Path(__file__).parent.parent.parent / "data"
DATABASE_PATH = pathlib.Path(
    os.getenv("RCT_CHECKER_DB_PATH", str(_default_data_dir / "paper_database.sqlite"))
)
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
