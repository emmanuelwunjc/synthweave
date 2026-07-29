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

    def __init__(self, directory: Path, table: str, *, format: str = "parquet") -> None:
        if format not in ("parquet", "csv"):
            raise ValueError(f"unsupported format {format!r}; use 'parquet' or 'csv'")
        self.format = format
        self.path = Path(directory) / f"{table}.{format}"
        self._writer = None
        self._wrote_header = False

    def write(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        if self.format == "csv":
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
        self._writer.write_table(batch)

    def close(self) -> Path:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif self.format == "csv" and not self._wrote_header:
            self.path.write_text("")
        return self.path
