import heapq
from functools import cmp_to_key
import math
import multiprocessing
import multiprocessing.managers
import multiprocessing.process
import os.path
import tempfile
import time
from contextlib import ExitStack
from functools import partial
from multiprocessing.sharedctypes import Synchronized
from shutil import rmtree
from typing import Union

import polars as pl
import psutil
import pyarrow.parquet as pq
import xxhash
from .ltm_aggregate import AggregationResult

HASH_BITS = 64
HASH_DTYPE = pl.UInt64
HASH_FN = xxhash.xxh3_64_intdigest


def get_default_partition_size():
  total_ram = psutil.virtual_memory().total
  total_ram_gb = total_ram / (1024**3)
  cpu_count = os.cpu_count() or 2
  return min(8_000_000, int(8_000_000 * pow(total_ram_gb / 64, 0.8) / (cpu_count / 12)))


def ltm_sort(
    input: Union[pl.DataFrame, list[str], AggregationResult],
    by: str | list[str], *,
    descending: bool | list[bool] = False,
    buffer_size=get_default_partition_size(),
) -> AggregationResult:
  if isinstance(by, str):
    by = [pl.col(by)]
  if isinstance(descending, bool):
    descending = [descending] * len(by)
  if len(descending) <= len(by):
    descending += [False] * (len(by) - len(descending))

  temp_dir = tempfile.TemporaryDirectory().name
  os.makedirs(temp_dir, exist_ok=True)
  print(f"temp dir is {temp_dir}")

  if isinstance(input, pl.DataFrame):
    output_path = os.path.join(temp_dir, "sorted.parquet")
    input.sort(by).write_parquet(output_path)
    return AggregationResult(result_paths=[output_path], temp_dir=temp_dir)

  if isinstance(input, AggregationResult):
    input_paths = input.result_paths
  else:
    input_paths = input

  presort_root_path = os.path.join(temp_dir, "presort")
  os.makedirs(presort_root_path, exist_ok=True)
  with multiprocessing.Pool() as pool:
    partial_presort_job = partial(
      presort_job,
      input_count=len(input_paths),
      by=by,
      descending=descending,
      out_dir=presort_root_path
    )
    presort_result = pool.map(
      partial_presort_job,
      (
        (input_path, input_index)
        for input_index, input_path in enumerate(input_paths)
      )
    )

  presorted_row_group_sizes = {
    input_index: row_group_size
    for input_index, _, row_group_size in presort_result
  }
  presorted_paths = {
    input_index: sorted_path
    for input_index, sorted_path, _ in presort_result
  }

  # we already have the row group sizes, we will now put each locally
  # sorted input into merge groups such that no merge group's sum of row group
  # size exceeds the buffer size
  merge_groups: list[tuple[int, list[str]]] = []
  merge_root_dir = os.path.join(temp_dir, "merge")
  current_merge_group: list[str] = []
  current_merge_group_size = 0
  merge_group_offset = 0
  for input_index, row_group_size in presorted_row_group_sizes.items():
    if current_merge_group_size + row_group_size > buffer_size:
      merge_groups.append((merge_group_offset, current_merge_group))
      current_merge_group = []
      current_merge_group_size = 0
      merge_group_offset += 1
    current_merge_group.append(presorted_paths[input_index])
    current_merge_group_size += row_group_size
  if current_merge_group:
    merge_groups.append((merge_group_offset, current_merge_group))
    merge_group_offset += 1

  merge_round_index = 0
  while True:
    merge_round_index += 1
    print(f"merge round {merge_round_index}")
    with multiprocessing.Pool() as pool:
      partial_merge_sort_job = partial(
        merge_sort_job,
        by=by,
        descending=descending,
        merge_root_dir=merge_root_dir,
        partition_size=buffer_size
      )
      max_row_group_sizes: list[tuple[int, int]] = pool.map(
        partial_merge_sort_job,
        merge_groups
      )

    if len(max_row_group_sizes) == 1:
      final_merge_group_id = merge_group_offset - 1
      final_merge_group_dir = os.path.join(
          merge_root_dir, f"merge_group_{final_merge_group_id}")
      for intermediate_merge_group_id in range(final_merge_group_id):
        rmtree(
          os.path.join(merge_root_dir, f"merge_group_{
            intermediate_merge_group_id}")
        )

      return AggregationResult(
        result_paths=[
          os.path.join(final_merge_group_dir, output_file)
          for output_file in os.listdir(final_merge_group_dir)
        ],
        temp_dir=temp_dir
      )

    merge_groups = []
    current_merge_group: list[str] = []
    current_merge_group_size = 0
    for merge_group_index, row_group_size in max_row_group_sizes:
      if current_merge_group_size + row_group_size > buffer_size:
        merge_groups.append(
          (merge_group_offset, current_merge_group)
        )
        merge_group_offset += 1
        current_merge_group = []
        current_merge_group_size = 0

      merge_group_output_dir = os.path.join(
        merge_root_dir, f"merge_group_{merge_group_index}")
      for output_file in os.listdir(merge_group_output_dir):
        current_merge_group.append(
          os.path.join(merge_group_output_dir, output_file)
        )

      current_merge_group_size += row_group_size
    if current_merge_group:
      merge_groups.append((merge_group_offset, current_merge_group))
      merge_group_offset += 1


def presort_job(arg: tuple[str, int], *, input_count: int, by: list[pl.Expr], descending: list[bool], out_dir: str):
  input_path, input_index = arg
  print(f"sorting input {input_path} {input_index + 1}/{input_count}")
  df = pl.read_parquet(input_path).sort(by, descending=descending)
  sorted_path = os.path.join(out_dir, f"sorted_{input_index}.parquet")
  df.write_parquet(sorted_path)

  df_arrow = df.to_arrow()
  arrow_schema = df_arrow.schema
  sorting_columns = [
    pq.SortingColumn(arrow_schema.get_field_index(
      by_expr), descending=by_descending)
    for by_expr, by_descending in zip(by, descending)
  ]
  with pq.ParquetWriter(sorted_path, arrow_schema, sorting_columns=sorting_columns) as writer:
    writer.write(df_arrow)

  with pq.ParquetFile(sorted_path) as pq_file:
    row_group_size: int = pq_file.metadata.row_group(0).num_rows
  return (input_index, sorted_path, row_group_size)


def compare_predicate(a: tuple, b: tuple, *, descending: list[bool]):
  for a_val, b_val, desc in zip(a, b, descending):
    if a_val < b_val:
      return 1 if desc else -1
    if a_val > b_val:
      return -1 if desc else 1
  return 0


def is_before_expr(by: list[str], other: tuple, descending: list[bool], or_at=False, df=None) -> pl.Expr:
  this_subj = pl.col(by[0])
  this_val = pl.lit(other[0])
  this_strict_expr = this_subj > this_val if descending[0] else this_subj < this_val
  if len(by) > 1:
    nested_expr = is_before_expr(by[1:], other[1:], descending[1:], or_at)
    return this_strict_expr | ((this_subj == this_val) & nested_expr)
  if or_at:
    return this_subj >= this_val if descending[0] else this_subj <= this_val
  return this_strict_expr


def merge_sort_job(arg: tuple[int, list[str]], *, by: list[str], descending: list[bool], merge_root_dir: str, partition_size: int):
  merge_group_id, input_paths = arg
  out_dir = os.path.join(merge_root_dir, f"merge_group_{merge_group_id}")
  os.makedirs(out_dir, exist_ok=True)

  print(f"merging {len(input_paths)} inputs into merge group {merge_group_id}")

  class MergeCandidate:
    def __init__(self, input_index: int, row_group_index: int, df: pl.DataFrame):
      self.input_index = input_index
      self.row_group_index = row_group_index
      self.df = df
      self.nearest_key = df.select(by).row(0)

    def __lt__(self, other: "MergeCandidate"):
      return compare_predicate(self.nearest_key, other.nearest_key, descending=descending) < 0

  output_index: int = 0
  buffer: list[pl.DataFrame] = []
  buffer_size = 0
  max_row_group_size: int = 0

  with ExitStack() as stack:
    pq_files = [
      stack.enter_context(pq.ParquetFile(input_path))
      for input_path in input_paths
    ]
    num_row_groups_by_input = [
      pq_file.metadata.num_row_groups for pq_file in pq_files
    ]
    merge_queue = [
      MergeCandidate(input_index, 0, pl.from_arrow(pq_file.read_row_group(0)))
      for input_index, pq_file in enumerate(pq_files)
    ]
    heapq.heapify(merge_queue)
    furthest_keys_by_candidate = [
      mc.df.select(by).row(-1) for mc in merge_queue]
    heapq.heapify(furthest_keys_by_candidate)

    while furthest_keys_by_candidate:
      nearest_furthest_key = heapq.heappop(furthest_keys_by_candidate)
      merge_candidates: list[MergeCandidate] = []
      while merge_queue:
        if compare_predicate(merge_queue[0].nearest_key, nearest_furthest_key, descending=descending) > 0:
          break
        merge_candidates.append(heapq.heappop(merge_queue))

      if not merge_candidates:
        continue

      merge_dfs = [
        mc.df.with_columns(
          is_before_expr(by, nearest_furthest_key, descending, True, mc.df)
            .alias("__merge__")
        )
        for mc in merge_candidates
      ]

      merged_df = pl.concat(
        df.filter("__merge__").drop("__merge__")
        for df in merge_dfs
      )
      sorted_df = merged_df.sort(by, descending=descending)

      if (sorted_df.height + buffer_size >= partition_size):
        slice_size = min(partition_size - buffer_size, sorted_df.height)
        buffer.append(sorted_df.head(slice_size))
        buffer_df = pl.concat(buffer)
        output_path = os.path.join(out_dir, f"output_{output_index}.parquet")
        buffer_df.write_parquet(output_path)
        with pq.ParquetFile(output_path) as pq_file:
          max_row_group_size = max(
            max_row_group_size, pq_file.metadata.row_group(0).num_rows
          )

        unflushed_row_count = sorted_df.height - slice_size
        if unflushed_row_count > 0:
          unflushed_df = sorted_df.tail(unflushed_row_count)
          buffer = [unflushed_df]
          buffer_size = unflushed_row_count
        else:
          buffer = []
          buffer_size = 0

        output_index += 1
      else:
        buffer.append(sorted_df)
        buffer_size += sorted_df.height

      for merge_df, mc in zip(merge_dfs, merge_candidates):
        unmerged_df = merge_df.filter(pl.col("__merge__") == False)
        if unmerged_df.height > 0:
          heapq.heappush(merge_queue, MergeCandidate(
            mc.input_index, mc.row_group_index, unmerged_df))
          continue

        if mc.row_group_index == num_row_groups_by_input[mc.input_index] - 1:
          continue

        next_row_group_index = mc.row_group_index + 1
        next_df = pl.from_arrow(
          pq_files[mc.input_index].read_row_group(next_row_group_index))
        heapq.heappush(
          furthest_keys_by_candidate, next_df.select(by).row(-1))
        heapq.heappush(
          merge_queue,
          MergeCandidate(mc.input_index, next_row_group_index, next_df))

  if buffer:
    buffer_df = pl.concat(buffer)
    output_path = os.path.join(out_dir, f"output_{output_index}.parquet")
    buffer_df.write_parquet(output_path)
    with pq.ParquetFile(output_path) as pq_file:
      max_row_group_size = max(
        max_row_group_size, pq_file.metadata.row_group(0).num_rows
      )

  return (merge_group_id, max_row_group_size)
