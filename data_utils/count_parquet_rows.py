import pyarrow.parquet as pq


def count_parquet_rows(path: str) -> int:
  return pq.ParquetFile(path).metadata.num_rows
