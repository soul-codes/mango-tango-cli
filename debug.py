if __name__ == "__main__":
  import multiprocessing
  multiprocessing.set_start_method("spawn")
  # import analyzers.time_coordination.test

  import numpy.random
  import polars as pl
  import os
  os.makedirs("__private__/test/input", exist_ok=True)

  # for i in range(1000):
  #   print(f"Generating {i}")
  #   df = pl.DataFrame({
  #     'a': numpy.random.randint(0, 100, 1_000_000),
  #     'b': numpy.random.randint(0, 100, 1_000_000)
  #   })
  #   df.write_parquet(f"__private__/test/input/{i}.parquet")

  from data_utils.ltm_sort2 import ltm_sort
  from datetime import datetime
  before = datetime.now()
  result = ltm_sort(
    [os.path.join("__private__/test/input", path)
     for path in os.listdir("__private__/test/input")],
    ["a", "b"],
    descending=[True, False],
  )
  result.move("__private__/test/output", force=True)
  print(f"Time taken: {datetime.now() - before}")
