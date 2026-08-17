# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compact, hierarchically mergeable SQLite storage for CBTS coverage."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterable, Mapping, Set
from pathlib import Path

SCHEMA_VERSION = "3"
STORAGE_SCHEMA = "compact-v1"

_CANON_RE = re.compile(r"(tensorrt_llm/.*)$")
_REQUIRED_COLUMNS = {
    "stage": {"id", "name"},
    "case_stage": {"id", "test", "stage_id"},
    "file": {"id", "path"},
    "symbol": {"id", "file_id", "qualname"},
    "process": {"id", "uid", "stage_id"},
    "process_case": {"process_id", "case_stage_id"},
    "touch": {"case_stage_id", "symbol_id"},
    "test_result": {
        "process_id",
        "case_stage_id",
        "outcome",
        "expected_workers",
    },
    "meta": {"key", "value"},
    "touch_rows": {"test", "file", "qualname", "stage"},
    "test_case_meta": {
        "test",
        "stage",
        "outcome",
        "expected_workers",
        "saved_procs",
    },
}


def canonicalize_path(path: str) -> str:
    """Collapse an install/source path to the product-relative database form."""
    match = _CANON_RE.search(path)
    return match.group(1) if match else path


def _bare_test_id(test: str, stage: str) -> str:
    prefix = f"{stage}/"
    return test[len(prefix) :] if stage and test.startswith(prefix) else test


def read_only_uri(path: str | os.PathLike[str]) -> str:
    """Return a SQLite URI that cannot create or mutate ``path``."""
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def _configure_writer(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE stage (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE case_stage (
            id INTEGER PRIMARY KEY,
            test TEXT NOT NULL,
            stage_id INTEGER NOT NULL REFERENCES stage(id),
            UNIQUE(test, stage_id)
        );
        CREATE TABLE file (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE
        );
        CREATE TABLE symbol (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES file(id),
            qualname TEXT NOT NULL,
            UNIQUE(file_id, qualname)
        );
        CREATE TABLE process (
            id INTEGER PRIMARY KEY,
            uid TEXT NOT NULL,
            stage_id INTEGER NOT NULL REFERENCES stage(id),
            UNIQUE(uid, stage_id)
        );
        CREATE TABLE process_case (
            process_id INTEGER NOT NULL REFERENCES process(id),
            case_stage_id INTEGER NOT NULL REFERENCES case_stage(id),
            PRIMARY KEY(process_id, case_stage_id)
        ) WITHOUT ROWID;
        CREATE TABLE touch (
            case_stage_id INTEGER NOT NULL REFERENCES case_stage(id),
            symbol_id INTEGER NOT NULL REFERENCES symbol(id),
            PRIMARY KEY(case_stage_id, symbol_id)
        ) WITHOUT ROWID;
        CREATE TABLE test_result (
            process_id INTEGER NOT NULL REFERENCES process(id),
            case_stage_id INTEGER NOT NULL REFERENCES case_stage(id),
            outcome TEXT NOT NULL,
            expected_workers INTEGER NOT NULL,
            PRIMARY KEY(process_id, case_stage_id, outcome, expected_workers)
        ) WITHOUT ROWID;
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE VIEW touch_rows AS
        SELECT CASE
                   WHEN case_stage.test = '' THEN ''
                   WHEN stage.name = '' THEN case_stage.test
                   ELSE stage.name || '/' || case_stage.test
               END AS test,
               file.path AS file,
               symbol.qualname AS qualname,
               stage.name AS stage
        FROM touch
        JOIN case_stage ON case_stage.id = touch.case_stage_id
        JOIN stage ON stage.id = case_stage.stage_id
        JOIN symbol ON symbol.id = touch.symbol_id
        JOIN file ON file.id = symbol.file_id;

        CREATE VIEW test_case_meta AS
        WITH per_process AS (
            SELECT process_id,
                   case_stage_id,
                   MAX(expected_workers) AS expected_workers,
                   SUM(CASE WHEN outcome != '' THEN 1 ELSE 0 END) AS outcome_count,
                   MAX(CASE WHEN outcome NOT IN ('', 'passed') THEN 1 ELSE 0 END)
                       AS has_non_passed
            FROM test_result
            GROUP BY process_id, case_stage_id
        ),
        result_summary AS (
            SELECT case_stage_id,
                   SUM(expected_workers) AS expected_workers,
                   SUM(outcome_count) AS outcome_count,
                   MAX(has_non_passed) AS has_non_passed
            FROM per_process
            GROUP BY case_stage_id
        ),
        save_summary AS (
            SELECT case_stage_id, COUNT(*) AS saved_procs
            FROM process_case
            GROUP BY case_stage_id
        )
        SELECT CASE
                   WHEN case_stage.test = '' THEN ''
                   WHEN stage.name = '' THEN case_stage.test
                   ELSE stage.name || '/' || case_stage.test
               END AS test,
               stage.name AS stage,
               CASE
                   WHEN COALESCE(result_summary.outcome_count, 0) = 0 THEN NULL
                   WHEN result_summary.has_non_passed != 0 THEN 'incomplete'
                   ELSE 'passed'
               END AS outcome,
               COALESCE(result_summary.expected_workers, 0) AS expected_workers,
               COALESCE(save_summary.saved_procs, 0) AS saved_procs
        FROM case_stage
        JOIN stage ON stage.id = case_stage.stage_id
        LEFT JOIN result_summary ON result_summary.case_stage_id = case_stage.id
        LEFT JOIN save_summary ON save_summary.case_stage_id = case_stage.id;
        """
    )
    connection.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?)",
        (("schema_version", SCHEMA_VERSION), ("storage_schema", STORAGE_SCHEMA)),
    )


def create_query_indexes(connection: sqlite3.Connection) -> None:
    """Create the reverse indexes used by reports and coverage selection."""
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS ix_case_test ON case_stage(test);
        CREATE INDEX IF NOT EXISTS ix_case_stage ON case_stage(stage_id, id);
        CREATE INDEX IF NOT EXISTS ix_touch_symbol ON touch(symbol_id, case_stage_id);
        CREATE INDEX IF NOT EXISTS ix_process_case_case
            ON process_case(case_stage_id, process_id);
        CREATE INDEX IF NOT EXISTS ix_test_result_case
            ON test_result(case_stage_id, process_id);
        ANALYZE main;
        """
    )


def _new_database(path: str | os.PathLike[str]) -> sqlite3.Connection:
    output_path = Path(path)
    if output_path.exists():
        output_path.unlink()
    connection = sqlite3.connect(output_path)
    _configure_writer(connection)
    _create_schema(connection)
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def validate_database(path: str | os.PathLike[str]) -> None:
    """Raise ``ValueError`` unless ``path`` is a compact CBTS database."""
    connection = sqlite3.connect(read_only_uri(path), uri=True)
    try:
        for table, required in _REQUIRED_COLUMNS.items():
            missing = required - _table_columns(connection, table)
            if missing:
                raise ValueError(f"{path}: {table} is missing columns: {sorted(missing)}")
        metadata = dict(connection.execute("SELECT key, value FROM meta"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: expected schema {SCHEMA_VERSION}, found {metadata.get('schema_version')}"
            )
        if metadata.get("storage_schema") != STORAGE_SCHEMA:
            raise ValueError(
                f"{path}: expected storage {STORAGE_SCHEMA}, found {metadata.get('storage_schema')}"
            )
    finally:
        connection.close()


def write_leaf_database(
    path: str | os.PathLike[str],
    *,
    stage: str,
    process_uid: str,
    touches: Mapping[str, Set[tuple[str, str]]],
    outcomes: Mapping[str, str],
    expected_workers: Mapping[str, int],
) -> None:
    """Write one process snapshot in the same compact schema used by aggregates."""
    connection = _new_database(path)
    try:
        stage_id = connection.execute("INSERT INTO stage(name) VALUES (?)", (stage,)).lastrowid
        process_id = connection.execute(
            "INSERT INTO process(uid, stage_id) VALUES (?, ?)", (process_uid, stage_id)
        ).lastrowid

        source_tests = set(touches) | set(outcomes) | set(expected_workers)
        bare_tests = {test: _bare_test_id(test, stage) for test in source_tests}
        tests = sorted(set(bare_tests.values()))
        connection.executemany(
            "INSERT INTO case_stage(test, stage_id) VALUES (?, ?)",
            ((test, stage_id) for test in tests),
        )
        case_ids = dict(connection.execute("SELECT test, id FROM case_stage"))

        canonical_touches = {
            test: {(canonicalize_path(file), qualname) for file, qualname in symbols}
            for test, symbols in touches.items()
        }
        symbol_keys = {symbol for symbols in canonical_touches.values() for symbol in symbols}
        files = sorted({file for file, _ in symbol_keys})
        connection.executemany("INSERT INTO file(path) VALUES (?)", ((file,) for file in files))
        file_ids = dict(connection.execute("SELECT path, id FROM file"))
        connection.executemany(
            "INSERT INTO symbol(file_id, qualname) VALUES (?, ?)",
            ((file_ids[file], qualname) for file, qualname in sorted(symbol_keys)),
        )
        symbol_ids = {
            (file, qualname): symbol_id
            for file, qualname, symbol_id in connection.execute(
                "SELECT file.path, symbol.qualname, symbol.id "
                "FROM symbol JOIN file ON file.id = symbol.file_id"
            )
        }

        connection.executemany(
            "INSERT OR IGNORE INTO touch(case_stage_id, symbol_id) VALUES (?, ?)",
            (
                (case_ids[_bare_test_id(test, stage)], symbol_ids[symbol])
                for test, symbols in canonical_touches.items()
                for symbol in symbols
            ),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO process_case(process_id, case_stage_id) VALUES (?, ?)",
            (
                (process_id, case_ids[_bare_test_id(test, stage)])
                for test, symbols in canonical_touches.items()
                if test and symbols
            ),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO test_result "
            "(process_id, case_stage_id, outcome, expected_workers) VALUES (?, ?, ?, ?)",
            (
                (
                    process_id,
                    case_ids[_bare_test_id(test, stage)],
                    outcomes.get(test) or "",
                    int(expected_workers.get(test, 0)),
                )
                for test in sorted(set(outcomes) | set(expected_workers))
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _merge_attached(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        INSERT OR IGNORE INTO main.stage(name)
        SELECT name FROM source.stage;

        INSERT OR IGNORE INTO main.case_stage(test, stage_id)
        SELECT source_case.test, target_stage.id
        FROM source.case_stage AS source_case
        JOIN source.stage AS source_stage ON source_stage.id = source_case.stage_id
        JOIN main.stage AS target_stage ON target_stage.name = source_stage.name;

        INSERT OR IGNORE INTO main.file(path)
        SELECT path FROM source.file;

        INSERT OR IGNORE INTO main.symbol(file_id, qualname)
        SELECT target_file.id, source_symbol.qualname
        FROM source.symbol AS source_symbol
        JOIN source.file AS source_file ON source_file.id = source_symbol.file_id
        JOIN main.file AS target_file ON target_file.path = source_file.path;

        INSERT OR IGNORE INTO main.process(uid, stage_id)
        SELECT source_process.uid, target_stage.id
        FROM source.process AS source_process
        JOIN source.stage AS source_stage ON source_stage.id = source_process.stage_id
        JOIN main.stage AS target_stage ON target_stage.name = source_stage.name;

        INSERT OR IGNORE INTO main.touch(case_stage_id, symbol_id)
        SELECT target_case.id, target_symbol.id
        FROM source.touch AS source_touch
        JOIN source.case_stage AS source_case
          ON source_case.id = source_touch.case_stage_id
        JOIN source.stage AS source_stage ON source_stage.id = source_case.stage_id
        JOIN main.stage AS target_stage ON target_stage.name = source_stage.name
        JOIN main.case_stage AS target_case
          ON target_case.test = source_case.test
         AND target_case.stage_id = target_stage.id
        JOIN source.symbol AS source_symbol ON source_symbol.id = source_touch.symbol_id
        JOIN source.file AS source_file ON source_file.id = source_symbol.file_id
        JOIN main.file AS target_file ON target_file.path = source_file.path
        JOIN main.symbol AS target_symbol
          ON target_symbol.file_id = target_file.id
         AND target_symbol.qualname = source_symbol.qualname;

        INSERT OR IGNORE INTO main.process_case(process_id, case_stage_id)
        SELECT target_process.id, target_case.id
        FROM source.process_case AS source_process_case
        JOIN source.process AS source_process
          ON source_process.id = source_process_case.process_id
        JOIN source.stage AS process_stage ON process_stage.id = source_process.stage_id
        JOIN main.stage AS target_process_stage ON target_process_stage.name = process_stage.name
        JOIN main.process AS target_process
          ON target_process.uid = source_process.uid
         AND target_process.stage_id = target_process_stage.id
        JOIN source.case_stage AS source_case
          ON source_case.id = source_process_case.case_stage_id
        JOIN source.stage AS case_stage ON case_stage.id = source_case.stage_id
        JOIN main.stage AS target_case_stage ON target_case_stage.name = case_stage.name
        JOIN main.case_stage AS target_case
          ON target_case.test = source_case.test
         AND target_case.stage_id = target_case_stage.id;

        INSERT OR IGNORE INTO main.test_result(
            process_id, case_stage_id, outcome, expected_workers
        )
        SELECT target_process.id,
               target_case.id,
               source_result.outcome,
               source_result.expected_workers
        FROM source.test_result AS source_result
        JOIN source.process AS source_process ON source_process.id = source_result.process_id
        JOIN source.stage AS process_stage ON process_stage.id = source_process.stage_id
        JOIN main.stage AS target_process_stage ON target_process_stage.name = process_stage.name
        JOIN main.process AS target_process
          ON target_process.uid = source_process.uid
         AND target_process.stage_id = target_process_stage.id
        JOIN source.case_stage AS source_case
          ON source_case.id = source_result.case_stage_id
        JOIN source.stage AS case_stage ON case_stage.id = source_case.stage_id
        JOIN main.stage AS target_case_stage ON target_case_stage.name = case_stage.name
        JOIN main.case_stage AS target_case
          ON target_case.test = source_case.test
         AND target_case.stage_id = target_case_stage.id;
        """
    )


def merge_databases(
    inputs: Iterable[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    *,
    query_indexes: bool = True,
) -> sqlite3.Connection:
    """Union compact inputs into a compact output and return its open connection."""
    input_paths = [Path(path).resolve() for path in inputs]
    for input_path in input_paths:
        validate_database(input_path)

    output = Path(output_path).resolve()
    if output in input_paths:
        raise ValueError(f"output database is also an input: {output}")
    connection = _new_database(output)
    try:
        for input_path in input_paths:
            connection.execute("ATTACH DATABASE ? AS source", (read_only_uri(input_path),))
            try:
                _merge_attached(connection)
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise
            finally:
                connection.execute("DETACH DATABASE source")
        if query_indexes:
            create_query_indexes(connection)
        connection.commit()
        return connection
    except (OSError, sqlite3.Error, ValueError):
        connection.close()
        output.unlink(missing_ok=True)
        raise
