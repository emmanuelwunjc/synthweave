"""Incremental chunk writing, so a long run is not lost at the end."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ChunkWriter:
    """Appends chunks to one file per table.

    Parquet writes a row group per chunk through a single open writer. CSV
    appends with the header written once. Either way nothing accumulates in
    memory and partial output survives a crash.
    """

    def __init__(
        self,
        directory: Path,
        table: str,
        *,
        format: str = "parquet",
        columns: list[str] | None = None,
    ) -> None:
        if format not in ("parquet", "csv"):
            raise ValueError(f"unsupported format {format!r}; use 'parquet' or 'csv'")
        self.format = format
        self.path = Path(directory) / f"{table}.{format}"
        # Declared up front so a table that turns out to have no rows still
        # leaves a file a reader can open. Without it the CSV was zero bytes
        # and the Parquet file was never created at all, while the run
        # reported success either way.
        self.columns = list(columns) if columns else []
        self._writer = None
        self._wrote_header = False
        self._csv_columns: list[str] | None = None

    def write(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        if self.format == "csv":
            if self._csv_columns is None:
                self._csv_columns = list(chunk.columns)
            elif list(chunk.columns) != self._csv_columns:
                chunk = self._reconcile_csv(chunk)
            chunk.to_csv(
                self.path,
                mode="a" if self._wrote_header else "w",
                header=not self._wrote_header,
                index=False,
            )
            self._wrote_header = True
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        batch = pa.Table.from_pandas(chunk, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, batch.schema)
        elif batch.schema != self._writer.schema:
            batch = self._reconcile(batch, pa)
        self._writer.write_table(batch)

    def _reconcile_csv(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Put a chunk onto the column order the file was opened with.

        A reordered set of the same columns is put back in order, silently,
        the same way Parquet's `_reconcile` widens a type silently. A missing
        or extra column is not something reordering can fix, so it stops the
        run with the column named rather than writing values under the wrong
        header.
        """
        if set(chunk.columns) != set(self._csv_columns):
            missing = [c for c in self._csv_columns if c not in chunk.columns]
            extra = [c for c in chunk.columns if c not in self._csv_columns]
            detail = "; ".join(
                filter(
                    None,
                    [
                        f"missing {missing}" if missing else "",
                        f"unexpected {extra}" if extra else "",
                    ],
                )
            )
            raise ValueError(
                f"{self.path.name}: a chunk's columns do not match the file's header "
                f"{self._csv_columns}: {detail}"
            )
        return chunk[self._csv_columns]

    def _reconcile(self, batch, pa):
        """Put a chunk onto the schema the file was opened with.

        Rules declare the type they produce, so a column's type does not
        normally vary between chunks. A rule that declares nothing, such as
        `Sequential` wrapping an arbitrary function, can still hand back whole
        numbers in one chunk and fractions in the next. Widening casts are
        safe and silent; a narrowing one would change values, so it stops the
        run with the column named rather than with pyarrow's schema dump.
        """
        try:
            return batch.cast(self._writer.schema)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            opened = {field.name: field.type for field in self._writer.schema}
            differing = [
                f"{field.name!r} is {field.type} here but the file was opened with "
                f"{opened.get(field.name)}"
                for field in batch.schema
                if opened.get(field.name) != field.type
            ]
            raise ValueError(
                f"{self.path.name}: a column changed type between chunks, so the value "
                f"cannot be written without changing it: {'; '.join(differing)}. "
                f"Give the rule that produces it a dtype(), or run with a chunk_size "
                f"large enough to hold the table."
            ) from exc

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self._wrote_header:
            self.write_empty()
        return self.path

    def write_empty(self) -> None:
        """Write a file holding the columns and no rows.

        An empty table is a real answer, so it gets a real file: a CSV with
        just its header, or a Parquet file with a schema and zero rows. Both
        read back as an empty frame rather than raising.
        """
        empty = pd.DataFrame({name: pd.Series(dtype=object) for name in self.columns})
        if self.format == "csv":
            empty.to_csv(self.path, index=False)
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        batch = pa.Table.from_pandas(empty, preserve_index=False)
        with pq.ParquetWriter(self.path, batch.schema) as writer:
            writer.write_table(batch)
