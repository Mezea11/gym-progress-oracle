from pathlib import Path
import sqlite3

import pandas as pd
from app.config import settings

DB_PATH = Path(settings.database_path)
DATASET_TABLE = settings.database_table_name


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database() -> None:
    # Vi skapar ingen tabell här eftersom Pandas to_sql hanterar schema.
    # Funktionen säkerställer att filen finns och går att öppna.
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL;")


def save_dataset_to_db(dataframe: pd.DataFrame) -> None:
    with _connect() as connection:
        dataframe.to_sql(
            DATASET_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )


def dataset_exists_in_db() -> bool:
    with _connect() as connection:
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (DATASET_TABLE,),
        ).fetchone()[0]

        if table_exists == 0:
            return False

        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {DATASET_TABLE}"
        ).fetchone()[0]

        return row_count > 0


def load_dataset_from_db() -> pd.DataFrame:
    with _connect() as connection:
        return pd.read_sql_query(f"SELECT * FROM {DATASET_TABLE}", connection)


def clear_dataset_in_db() -> int:
    with _connect() as connection:
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (DATASET_TABLE,),
        ).fetchone()[0]

        if table_exists == 0:
            return 0

        row_count = connection.execute(
            f"SELECT COUNT(*) FROM {DATASET_TABLE}"
        ).fetchone()[0]

        connection.execute(f"DROP TABLE IF EXISTS {DATASET_TABLE}")

        return int(row_count)
