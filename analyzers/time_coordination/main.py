import multiprocessing
import os.path
from datetime import datetime
from functools import partial
from typing import Iterable

import polars as pl

from analyzer_interface.context import PrimaryAnalyzerContext
from data_utils import AggregationResult, AggregationSpec, count, ltm_aggregate, ltm_sort

from .interface import (COL_TIMESTAMP, COL_USER_ID, OUTPUT_COL_FREQ,
                        OUTPUT_COL_USER1, OUTPUT_COL_USER2, OUTPUT_TABLE)


def transform_initialize(user_def_path: str):
  return pl.read_parquet(user_def_path)


def transform_partition(df: pl.DataFrame, df_users: pl.DataFrame):
  return (
    df
      .join(df_users, left_on=OUTPUT_COL_USER1, right_on=COL_USER_ID, how="inner")
      .drop(OUTPUT_COL_USER1)
      .rename({"user_identifier": OUTPUT_COL_USER1})
      .join(df_users, left_on=OUTPUT_COL_USER2, right_on=COL_USER_ID, how="inner")
      .drop(OUTPUT_COL_USER2)
      .rename({"user_identifier": OUTPUT_COL_USER2})
  )


def main(context: PrimaryAnalyzerContext):
  phases = [
    (0, 60 * 15),
    (60 * 5, 60 * 15),
    (60 * 10, 60 * 15)
  ]

  input_reader = context.input()
  df = input_reader.preprocess(
    pl.read_parquet(input_reader.parquet_path)
  )
  df = df.filter(
    pl.col(COL_USER_ID).is_not_null() &
    pl.col(COL_TIMESTAMP).is_not_null()
  )

  user_def_path = os.path.join(context.temp_dir, "users.parquet")
  df_users = df.select(COL_USER_ID).unique().sort(COL_USER_ID)
  df_users = df_users.select(
    pl.col(COL_USER_ID).alias("user_identifier"),
    pl.arange(0, df_users.height).alias(COL_USER_ID)
  )
  df_users.write_parquet(user_def_path)

  df = df.join(
    df_users, left_on=COL_USER_ID,
    right_on="user_identifier", how="inner").drop(COL_USER_ID).rename({f"{COL_USER_ID}_right": COL_USER_ID})

  timestamp_counts = df.select(
    pl.col(COL_TIMESTAMP).dt.epoch(f"s") // (5 * 60),
  ).n_unique()
  print(f"There are {timestamp_counts} unique 5-minute windows")

  phase_results = [
    ltm_aggregate(
      df.with_columns(
        ((pl.col(COL_TIMESTAMP).dt.epoch("s") - offset)
         // period
         * period
         + offset).alias("ts")
      ),
      ["ts"],
      AggregationSpec(
        lift=[pl.col(COL_USER_ID).unique().alias("user_ids")],
        fold=[pl.col("user_ids")],
        finish=[
          pl.col("user_ids").map_elements(
            flatten_unique,
            return_dtype=pl.List(df.schema.get(COL_USER_ID))
          )
        ]
      )
    )
    for offset, period in phases
  ]

  print(f"Starting explosion at {datetime.now()}")
  explode_base_path = os.path.join(context.temp_dir, "explode")
  os.makedirs(explode_base_path, exist_ok=True)

  with multiprocessing.Pool() as pool:
    partial_explode_job = partial(
      explode_job,
      phase_count=len(phase_results),
      explode_base_path=explode_base_path
    )
    pool.map(partial_explode_job, enumerate(phase_results))

  cooccurrence_result = ltm_aggregate(
    [
      os.path.join(explode_base_path, chunk_file)
      for chunk_file in os.listdir(explode_base_path)
    ],
    [OUTPUT_COL_USER1, OUTPUT_COL_USER2],
    count(OUTPUT_COL_FREQ),
    having=pl.col(OUTPUT_COL_FREQ).gt(timestamp_counts * 0.05),
    transform_initialize=transform_initialize,
    transform_initialize_arg=user_def_path,
    transform_partition=transform_partition
  )

  print("Generating histograms")
  (
    ltm_aggregate(
      cooccurrence_result,
      [OUTPUT_COL_FREQ],
      count("num_user_pairs")
    )
    .merge()
    .sort(OUTPUT_COL_FREQ, descending=True)
    .write_parquet(context.output("histogram").parquet_path)
  )

  print("Combining outputs")
  sorted_cooccurrence_result = ltm_sort(
    cooccurrence_result,
    by=[OUTPUT_COL_FREQ, OUTPUT_COL_USER1, OUTPUT_COL_USER2],
    descending=[True, False, False]
  )
  sorted_cooccurrence_result.sink_parquet(
    context.output(OUTPUT_TABLE).parquet_path)


def flatten_unique(nested_list):
  return list(set(_flatten(nested_list)))


def _flatten(nested_list):
  for item in nested_list:
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
      yield from _flatten(item)
    else:
      yield item


def explode_job(arg: tuple[int, AggregationResult], *, phase_count: int, explode_base_path: str):
  max_pairs_sum_per_chunk = 25_000_000
  phase_index, phase_result = arg
  with phase_result as paths:
    for path_index, path in enumerate(paths):
      df = pl.read_parquet(path)
      chunk_groups = df.with_columns(
        (
          pl.col("user_ids").list.len().pow(2).cum_sum()
          // 2
          // max_pairs_sum_per_chunk
        ).alias("chunk")
      ).group_by("chunk")
      for (chunk_index,), chunk_df in chunk_groups:
        print(f"Phase {phase_index + 1}/{phase_count} file {path_index +
              1}/{len(paths)} chunk {chunk_index + 1}")
        chunk_df = chunk_df.explode("user_ids")
        chunk_df = chunk_df.drop("chunk")
        chunk_df = chunk_df.join(chunk_df, on="ts", how="inner").rename({
          f"user_ids": OUTPUT_COL_USER1,
          f"user_ids_right": OUTPUT_COL_USER2
        })
        chunk_df = chunk_df.filter(
          pl.col(OUTPUT_COL_USER1).lt(pl.col(OUTPUT_COL_USER2)))
        chunk_df = chunk_df.select(OUTPUT_COL_USER1, OUTPUT_COL_USER2)
        chunk_path = os.path.join(
          explode_base_path, f"chunk_{phase_index}_{path_index}_{chunk_index}.parquet")
        chunk_df.write_parquet(chunk_path)
