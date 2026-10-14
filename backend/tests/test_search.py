import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import sqlite3

from backend.search import _connect, DB_PATH

def test_db_path_and_connection(tmp_path):
    """Verify DB_PATH points to rag_dw/data/search.db, directory exists, and connection works."""
    # Ensure DB_PATH is absolute and contains the expected "data" folder
    assert DB_PATH.is_absolute(), "DB_PATH should be absolute"
    assert "data" in str(DB_PATH), "DB_PATH should be inside the data directory"

    # The parent directory should exist (the _connect function creates it if missing)
    assert DB_PATH.parent.exists(), f"Parent directory {DB_PATH.parent} does not exist"

    # Try to open a SQLite connection without error
    conn: sqlite3.Connection = _connect()
    cur = conn.cursor()
    cur.execute("SELECT 1")
    assert cur.fetchone()[0] == 1
    conn.close()
