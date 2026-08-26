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
"""Build and evaluate the one-shot flat-v2 compact-schema sizing experiment.

Production compact storage and hierarchical merging live in
``cbts.coverage.collection.compact_db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Union

import click

_SOURCE_REQUIRED_COLUMNS = {
    "touch": {"test", "file", "qualname", "stage"},
    "test_meta": {"test", "stage", "outcome", "expected_workers", "saved_procs"},
    "meta": {"key", "value"},
}


def _read_only_uri(path: Union[str, os.PathLike[str]]) -> str:
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _validate_source(path: Union[str, os.PathLike[str]]) -> None:
    connection = sqlite3.connect(_read_only_uri(path), uri=True)
    try:
        for table, required in _SOURCE_REQUIRED_COLUMNS.items():
            columns = _table_columns(connection, table)
            missing = required - columns
            if missing:
                raise ValueError(f"{table} is missing columns: {sorted(missing)}")
    finally:
        connection.close()


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
        CREATE TABLE touch (
            case_stage_id INTEGER NOT NULL REFERENCES case_stage(id),
            symbol_id INTEGER NOT NULL REFERENCES symbol(id),
            PRIMARY KEY(case_stage_id, symbol_id)
        ) WITHOUT ROWID;
        CREATE TABLE test_meta (
            case_stage_id INTEGER PRIMARY KEY REFERENCES case_stage(id),
            outcome TEXT,
            expected_workers INTEGER,
            saved_procs INTEGER
        );
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE compact_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )


def _populate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        INSERT INTO stage(name)
        SELECT stage FROM source.touch
        UNION
        SELECT stage FROM source.test_meta
        ORDER BY 1
        """
    )
    connection.execute(
        """
        INSERT INTO case_stage(test, stage_id)
        SELECT source_case.test, stage.id
        FROM (
            SELECT test, stage FROM source.touch
            UNION
            SELECT test, stage FROM source.test_meta
        ) AS source_case
        JOIN stage ON stage.name = source_case.stage
        ORDER BY source_case.test, source_case.stage
        """
    )
    connection.execute(
        """
        INSERT INTO file(path)
        SELECT DISTINCT file
        FROM source.touch
        ORDER BY file
        """
    )
    connection.execute(
        """
        INSERT INTO symbol(file_id, qualname)
        SELECT file.id, source_symbol.qualname
        FROM (
            SELECT DISTINCT file, qualname
            FROM source.touch
        ) AS source_symbol
        JOIN file ON file.path = source_symbol.file
        ORDER BY source_symbol.file, source_symbol.qualname
        """
    )
    connection.execute(
        """
        INSERT INTO touch(case_stage_id, symbol_id)
        SELECT case_stage.id, symbol.id
        FROM source.touch AS source_touch
        JOIN stage ON stage.name = source_touch.stage
        JOIN case_stage
          ON case_stage.test = source_touch.test
         AND case_stage.stage_id = stage.id
        JOIN file ON file.path = source_touch.file
        JOIN symbol
          ON symbol.file_id = file.id
         AND symbol.qualname = source_touch.qualname
        """
    )
    connection.execute(
        """
        INSERT INTO test_meta(case_stage_id, outcome, expected_workers, saved_procs)
        SELECT case_stage.id,
               source_meta.outcome,
               source_meta.expected_workers,
               source_meta.saved_procs
        FROM source.test_meta AS source_meta
        JOIN stage ON stage.name = source_meta.stage
        JOIN case_stage
          ON case_stage.test = source_meta.test
         AND case_stage.stage_id = stage.id
        """
    )
    connection.execute("INSERT INTO meta SELECT key, value FROM source.meta")
    connection.executemany(
        "INSERT INTO compact_meta VALUES (?, ?)",
        (("schema_version", "compact-spike-1"), ("source_schema", "touch-v2")),
    )


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX main.ix_case_test ON case_stage(test);
        CREATE INDEX main.ix_case_stage ON case_stage(stage_id, id);
        CREATE INDEX main.ix_touch_symbol ON touch(symbol_id, case_stage_id);
        ANALYZE main;
        """
    )


def build_compact_database(
    source_path: Union[str, os.PathLike[str]], output_path: Union[str, os.PathLike[str]]
) -> None:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    _validate_source(source_path)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    connection = sqlite3.connect(output_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("ATTACH DATABASE ? AS source", (_read_only_uri(source_path),))
        _create_schema(connection)
        _populate(connection)
        _create_indexes(connection)
        connection.commit()
    except Exception:
        connection.close()
        output_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()


def _count(connection: sqlite3.Connection, sql: str) -> int:
    return connection.execute(sql).fetchone()[0]


def _allocated_bytes(connection: sqlite3.Connection) -> int:
    page_size = _count(connection, "PRAGMA page_size")
    page_count = _count(connection, "PRAGMA page_count")
    return page_size * page_count


def _object_sizes(connection: sqlite3.Connection) -> dict[str, int]:
    try:
        rows = connection.execute(
            "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY SUM(pgsize) DESC"
        )
        return {name: size for name, size in rows}
    except sqlite3.OperationalError:
        return {}


def _run_queries(connection: sqlite3.Connection, sql: str, parameters: list) -> tuple[float, int]:
    row_count = 0
    started = time.perf_counter()
    for parameter in parameters:
        row_count += len(connection.execute(sql, parameter).fetchall())
    return time.perf_counter() - started, row_count


def _benchmark_pair(
    source: sqlite3.Connection,
    compact: sqlite3.Connection,
    source_sql: str,
    compact_sql: str,
    parameters: list,
    repeats: int,
) -> dict[str, float]:
    # Warm both connections before measuring their cached lookup cost.
    _run_queries(source, source_sql, parameters)
    _run_queries(compact, compact_sql, parameters)
    source_seconds = 0.0
    compact_seconds = 0.0
    source_rows = 0
    compact_rows = 0
    for _ in range(repeats):
        elapsed, rows = _run_queries(source, source_sql, parameters)
        source_seconds += elapsed
        source_rows += rows
        elapsed, rows = _run_queries(compact, compact_sql, parameters)
        compact_seconds += elapsed
        compact_rows += rows
    if source_rows != compact_rows:
        raise ValueError(f"benchmark row mismatch: source={source_rows}, compact={compact_rows}")
    query_count = len(parameters) * repeats
    return {
        "queries": query_count,
        "rows": source_rows,
        "source_seconds": round(source_seconds, 6),
        "compact_seconds": round(compact_seconds, 6),
        "source_ms_per_query": round(1000.0 * source_seconds / query_count, 6),
        "compact_ms_per_query": round(1000.0 * compact_seconds / query_count, 6),
        "compact_to_source_ratio": round(compact_seconds / source_seconds, 6),
    }


def benchmark_queries(
    source_path: Union[str, os.PathLike[str]],
    compact_path: Union[str, os.PathLike[str]],
    sample_size: int = 100,
    repeats: int = 5,
) -> dict[str, dict[str, float]]:
    source = sqlite3.connect(_read_only_uri(source_path), uri=True)
    compact = sqlite3.connect(_read_only_uri(compact_path), uri=True)
    try:
        tests = list(
            source.execute(
                "SELECT DISTINCT test FROM touch WHERE test != '' ORDER BY test LIMIT ?",
                (sample_size,),
            )
        )
        files = list(
            source.execute("SELECT DISTINCT file FROM touch ORDER BY file LIMIT ?", (sample_size,))
        )
        functions = list(
            source.execute(
                "SELECT DISTINCT file, qualname FROM touch ORDER BY file, qualname LIMIT ?",
                (sample_size,),
            )
        )
        return {
            "test_to_symbols": _benchmark_pair(
                source,
                compact,
                "SELECT file, qualname, stage FROM touch WHERE test = ?",
                """
                SELECT file.path, symbol.qualname, stage.name
                FROM case_stage
                JOIN stage ON stage.id = case_stage.stage_id
                JOIN touch ON touch.case_stage_id = case_stage.id
                JOIN symbol ON symbol.id = touch.symbol_id
                JOIN file ON file.id = symbol.file_id
                WHERE case_stage.test = ?
                """,
                tests,
                repeats,
            ),
            "file_to_tests": _benchmark_pair(
                source,
                compact,
                "SELECT DISTINCT test, stage FROM touch WHERE file = ?",
                """
                SELECT DISTINCT case_stage.test, stage.name
                FROM file
                JOIN symbol ON symbol.file_id = file.id
                JOIN touch ON touch.symbol_id = symbol.id
                JOIN case_stage ON case_stage.id = touch.case_stage_id
                JOIN stage ON stage.id = case_stage.stage_id
                WHERE file.path = ?
                """,
                files,
                repeats,
            ),
            "function_to_tests": _benchmark_pair(
                source,
                compact,
                "SELECT DISTINCT test, stage FROM touch WHERE file = ? AND qualname = ?",
                """
                SELECT DISTINCT case_stage.test, stage.name
                FROM file
                JOIN symbol
                  ON symbol.file_id = file.id
                 AND symbol.qualname = ?2
                JOIN touch ON touch.symbol_id = symbol.id
                JOIN case_stage ON case_stage.id = touch.case_stage_id
                JOIN stage ON stage.id = case_stage.stage_id
                WHERE file.path = ?1
                """,
                functions,
                repeats,
            ),
        }
    finally:
        compact.close()
        source.close()


def _verify_equivalence(
    source: Union[str, os.PathLike[str]], compact: sqlite3.Connection
) -> dict[str, Union[int, bool]]:
    compact.execute("ATTACH DATABASE ? AS source", (_read_only_uri(source),))
    compact.execute("PRAGMA temp_store=FILE")
    compact.execute("PRAGMA cache_size=-262144")
    missing_from_compact = _count(
        compact,
        """
        SELECT COUNT(*)
        FROM (
            SELECT test, file, qualname, stage FROM source.touch
            EXCEPT
            SELECT case_stage.test, file.path, symbol.qualname, stage.name
            FROM touch
            JOIN case_stage ON case_stage.id = touch.case_stage_id
            JOIN stage ON stage.id = case_stage.stage_id
            JOIN symbol ON symbol.id = touch.symbol_id
            JOIN file ON file.id = symbol.file_id
        )
        """,
    )
    extra_in_compact = _count(
        compact,
        """
        SELECT COUNT(*)
        FROM (
            SELECT case_stage.test, file.path, symbol.qualname, stage.name
            FROM touch
            JOIN case_stage ON case_stage.id = touch.case_stage_id
            JOIN stage ON stage.id = case_stage.stage_id
            JOIN symbol ON symbol.id = touch.symbol_id
            JOIN file ON file.id = symbol.file_id
            EXCEPT
            SELECT test, file, qualname, stage FROM source.touch
        )
        """,
    )
    missing_test_meta = _count(
        compact,
        """
        SELECT COUNT(*)
        FROM (
            SELECT test, stage, outcome, expected_workers, saved_procs FROM source.test_meta
            EXCEPT
            SELECT case_stage.test,
                   stage.name,
                   test_meta.outcome,
                   test_meta.expected_workers,
                   test_meta.saved_procs
            FROM test_meta
            JOIN case_stage ON case_stage.id = test_meta.case_stage_id
            JOIN stage ON stage.id = case_stage.stage_id
        )
        """,
    )
    extra_test_meta = _count(
        compact,
        """
        SELECT COUNT(*)
        FROM (
            SELECT case_stage.test,
                   stage.name,
                   test_meta.outcome,
                   test_meta.expected_workers,
                   test_meta.saved_procs
            FROM test_meta
            JOIN case_stage ON case_stage.id = test_meta.case_stage_id
            JOIN stage ON stage.id = case_stage.stage_id
            EXCEPT
            SELECT test, stage, outcome, expected_workers, saved_procs FROM source.test_meta
        )
        """,
    )
    return {
        "missing_touch_rows": missing_from_compact,
        "extra_touch_rows": extra_in_compact,
        "missing_test_meta_rows": missing_test_meta,
        "extra_test_meta_rows": extra_test_meta,
        "equivalent": not any(
            (missing_from_compact, extra_in_compact, missing_test_meta, extra_test_meta)
        ),
    }


def evaluate(
    source_path: Union[str, os.PathLike[str]],
    compact_path: Union[str, os.PathLike[str]],
    build_seconds: float,
) -> dict:
    source_path = Path(source_path).resolve()
    compact_path = Path(compact_path).resolve()
    source = sqlite3.connect(_read_only_uri(source_path), uri=True)
    compact = sqlite3.connect(_read_only_uri(compact_path), uri=True)
    try:
        verification = _verify_equivalence(source_path, compact)
        source_bytes = _allocated_bytes(source)
        compact_bytes = _allocated_bytes(compact)
        return {
            "source": {
                "path": os.fspath(source_path),
                "bytes": source_bytes,
                "touch_rows": _count(source, "SELECT COUNT(*) FROM touch"),
                "object_bytes": _object_sizes(source),
            },
            "compact": {
                "path": os.fspath(compact_path),
                "bytes": compact_bytes,
                "touch_rows": _count(compact, "SELECT COUNT(*) FROM touch"),
                "object_bytes": _object_sizes(compact),
            },
            "build_seconds": round(build_seconds, 3),
            "size_ratio": round(compact_bytes / source_bytes, 6),
            "reduction_pct": round(100.0 * (1.0 - compact_bytes / source_bytes), 3),
            "verification": verification,
        }
    finally:
        compact.close()
        source.close()


@click.command("compact-touch-db", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("source")
@click.argument("output")
@click.option("--report-json", default=None, help="optional path for the JSON result")
@click.option(
    "--reuse-output",
    is_flag=True,
    help="evaluate an existing compact output instead of rebuilding it",
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="benchmark cached forward and reverse lookup queries",
)
def main(source, output, report_json, reuse_output, benchmark):
    """Build and evaluate the one-shot flat-v2 compact-schema sizing experiment.

    Production compact storage and hierarchical merging live in
    ``cbts.coverage.collection.compact_db``.
    """
    started = time.perf_counter()
    if reuse_output:
        if not Path(output).is_file():
            raise click.UsageError(f"compact output does not exist: {output}")
    else:
        build_compact_database(source, output)
    result = evaluate(source, output, time.perf_counter() - started)
    if benchmark:
        result["query_benchmark"] = benchmark_queries(source, output)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if report_json:
        Path(report_json).write_text(rendered + "\n", encoding="utf-8")
    if not result["verification"]["equivalent"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
