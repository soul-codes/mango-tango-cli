from .main import main
import os
import polars as pl
from datetime import datetime, timedelta
from analyzer_interface.context import PrimaryAnalyzerContext, InputTableReader, TableWriter
from .interface import COL_TIMESTAMP, COL_USER_ID, OUTPUT_TABLE, OUTPUT_COL_USER1, OUTPUT_COL_USER2, OUTPUT_COL_FREQ
from pydantic import BaseModel

window_size = 15*60
sliding_window = 5*60
assert window_size % sliding_window == 0

ts_row_count = 400
ts_row_digits = len(str(ts_row_count))


def create_test_input() -> pl.DataFrame:
  timestamps = [
    datetime(2020, 1, 1) + timedelta(seconds=sliding_window*row_index)
    for row_index in range(ts_row_count)
  ]
  users = [
    [
      f"user_{str(user_index+1).zfill(ts_row_digits)}"
      for user_index in range(row_index + 1)
    ]
    for row_index in range(ts_row_count)
  ]
  df = pl.DataFrame({
    COL_TIMESTAMP: timestamps,
    COL_USER_ID: users
  })
  df = df.explode("user_id")
  return df

def control(df: pl.DataFrame):
  def inner(df: pl.DataFrame, offset: int, period: int):
    df = df.select(
      pl.col(COL_USER_ID),
      (
        (df[COL_TIMESTAMP].dt.epoch("s") - offset)
        // period * period + offset
      ).alias("ts")
    )
    df = df.group_by(["ts", COL_USER_ID]).agg(pl.col(COL_USER_ID).alias("user_ids"))
    df = df.select(["ts", "user_ids"])
    df = df.explode("user_ids")
    df = df.join(df, on="ts", how="inner")
    df = df.rename({
      f"user_ids": OUTPUT_COL_USER1,
      f"user_ids_right": OUTPUT_COL_USER2
    })
    df = df.filter(pl.col(OUTPUT_COL_USER1) < pl.col(OUTPUT_COL_USER2))
    df = df.unique()
    return df

  df = pl.concat([
    inner(df, offset, window_size)
    for offset in range(0, window_size, sliding_window)
  ])
  df = df.group_by([OUTPUT_COL_USER1, OUTPUT_COL_USER2]).agg(pl.count().alias(OUTPUT_COL_FREQ))
  df = df.filter(pl.col(OUTPUT_COL_FREQ) > 1)
  return (
    df.sort([OUTPUT_COL_FREQ, OUTPUT_COL_USER1, OUTPUT_COL_USER2], descending=[True,False,False])
    .select([OUTPUT_COL_FREQ, OUTPUT_COL_USER1, OUTPUT_COL_USER2])
  )



root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "__private__")
os.makedirs(root_dir, exist_ok=True)

test_input_path = os.path.join(root_dir, f"test_input_{ts_row_count}.parquet")
if not os.path.exists(test_input_path):
  create_test_input().write_parquet(test_input_path)

input = pl.read_parquet(test_input_path)
control_output = control(input)

# input.write_csv(os.path.join(root_dir, f"test_input_{ts_row_count}.csv"))
control_output.write_csv(os.path.join(root_dir, f"output_control_{ts_row_count}.csv"))


class FakeInputTableReader(InputTableReader):
  def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
    return df

  @property
  def parquet_path(self) -> str:
    return test_input_path

class FakeTableWriter(TableWriter, BaseModel):
  output_id: str

  @property
  def parquet_path(self) -> str:
    return os.path.join(root_dir, f"{self.output_id}.parquet")


class FakePrimaryAnalyzerContext(PrimaryAnalyzerContext):
  def input(self) -> InputTableReader:
    return FakeInputTableReader()

  def output(self, output_id: str) -> TableWriter:
    return FakeTableWriter(output_id=output_id)

temp_dir = os.path.join(root_dir, "temp")

ctx = FakePrimaryAnalyzerContext(temp_dir=temp_dir)
main(ctx)

pl.read_parquet(ctx.output(OUTPUT_TABLE).parquet_path).write_csv(os.path.join(root_dir, f"output_perf_{ts_row_count}.csv"))

with open(os.path.join(root_dir, f"output_control_{ts_row_count}.csv")) as f:
  output_control_as_text = f.read()
with open(os.path.join(root_dir, f"output_perf_{ts_row_count}.csv")) as f:
  output_perf_as_text = f.read()

assert output_control_as_text == output_perf_as_text
print("Control and performance outputs are equal.")