import multiprocessing
from datetime import datetime

import polars as pl

from data_utils.ltm_aggregate import count, ltm_aggregate

multiprocessing.set_start_method("spawn")


if __name__ == "__main__":
  df = pl.DataFrame({'x': range(40_000)})
  df = df.select(pl.col('x').floor())
  lf = df.lazy()
  lf = lf.with_columns(
    pl.col("x")
      .map_elements(lambda x: range(x), pl.List(pl.Float64))
      .alias("y")
  )
  lf = lf.explode("y")
  x = datetime.now()
  with ltm_aggregate(lf, ["x"], count("count"), repartition_threshold=10_000_000) as result_paths:
    concat_result = pl.concat([pl.read_parquet(path) for path in result_paths])
    print(concat_result)
  y = datetime.now()
  print(y - x)
