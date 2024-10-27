import math
import multiprocessing
import multiprocessing.managers
import multiprocessing.process
import os.path
import tempfile
import time
from functools import partial
from multiprocessing.sharedctypes import Synchronized
from shutil import rmtree
from typing import Union

import polars as pl
import pyarrow.parquet as pq
import xxhash
from pydantic import BaseModel

from .count_parquet_rows import count_parquet_rows


class AggregationSpec(BaseModel):
  lift: list[pl.Expr]
  fold: list[pl.Expr]
  finish: list[pl.Expr]

  class Config:
    arbitrary_types_allowed = True


def count(alias: str):
  return AggregationSpec(
    lift=[pl.len().alias(alias)],
    fold=[pl.sum(alias)],
    finish=[pl.col(alias)]
  )


def sum_agg(target: str, alias: str):
  return AggregationSpec(
    lift=[pl.sum(target).alias(alias)],
    fold=[pl.sum(alias)],
    finish=[pl.col(alias)]
  )


class PartitionContext(BaseModel):
  id: int
  bits: int
  root_dir: str
  repartition_threshold: int
  spec: AggregationSpec
  by: list[str]


class AggregationResult(BaseModel):
  temp_dir: str
  result_paths: list[str]

  def __enter__(self):
    return self.result_paths

  def __exit__(self, exc_type, exc_val, exc_tb):
    rmtree(self.temp_dir)


def get_partition_id(bits: int, current_index: int, parent_id: int, parent_bits: int):
  return parent_id + ((current_index * (1 << (32 - bits))))


def get_partition_modulo(partition_bits: int):
  return 1 << partition_bits


def get_partition_path(root_dir: str, id: int):
  return os.path.join(root_dir, "partitions", format(id, "08x"))


def ltm_aggregate(lf: Union[pl.LazyFrame, pl.DataFrame, list[str]], by: list[str], spec: AggregationSpec, *, repartition_threshold=10_000_000):
  temp_dir = tempfile.TemporaryDirectory().name
  os.makedirs(temp_dir, exist_ok=True)

  if isinstance(lf, list):
    input_paths = lf
  else:
    lf = lf.lazy()

    print(f"temp dir is {temp_dir}")
    print(f"persisting parquet")

    input_path = os.path.join(temp_dir, "input.parquet")
    lf.sink_parquet(input_path)
    input_paths = [input_path]

  total_rows = sum(
    count_parquet_rows(input_path)
    for input_path in input_paths
  )

  if total_rows < repartition_threshold:
    df = pl.concat([
      pl.read_parquet(input_path)
      for input_path in input_paths
    ])
    df = df.group_by(by).agg(spec.lift).select([*by, *spec.finish])
    output_path = os.path.join(temp_dir, "finished.parquet")
    df.write_parquet(output_path)

    return AggregationResult(
      result_paths=[output_path],
      temp_dir=temp_dir
    )

  total_row_groups = sum(
    pq.ParquetFile(input_path).metadata.num_row_groups
    for input_path in input_paths
  )

  partition_bits = min(max(
    math.ceil(
      math.log2(total_rows / repartition_threshold)
    ),
    0
  ), 4)

  print(f"Input has {total_rows} rows in {total_row_groups} row groups")
  print(f"Using {get_partition_modulo(partition_bits)} initial partitions")

  for partition_index in range(get_partition_modulo(partition_bits)):
    partition_id = get_partition_id(partition_bits, partition_index, 0, 0)
    os.makedirs(get_partition_path(
      temp_dir,
      partition_id
    ), exist_ok=True)

  with multiprocessing.Pool() as pool:
    partial_lift_job = partial(
      lift_job,
      input_count=len(input_paths),
      by=by,
      spec=spec,
      partition_bits=partition_bits,
      temp_dir=temp_dir
    )
    partitions_with_data: set[int] = set.union(set(), *pool.map(
      partial_lift_job,
      (
        (input_path, input_index, row_group_index)
        for input_index, input_path in enumerate(input_paths)
        for row_group_index in
        range(pq.ParquetFile(input_path).metadata.num_row_groups)
      )
    ))

  job_queue = multiprocessing.Queue()
  for partition_id in partitions_with_data:
    job_queue.put(PartitionContext(
      id=partition_id,
      bits=partition_bits,
      root_dir=get_partition_path(temp_dir, partition_id),
      repartition_threshold=repartition_threshold,
      spec=spec,
      by=by
    ))

  active_task_count = multiprocessing.Value("i", 0)
  processes: list[multiprocessing.Process] = []
  with multiprocessing.Manager() as manager:
    live_result_paths = manager.list()

    for worker_index in range(os.cpu_count()):
      process = multiprocessing.Process(
        target=fold_job_worker,
        args=(job_queue, active_task_count, live_result_paths)
      )
      process.start()
      processes.append(process)

    for process in processes:
      process.join()

    result_paths = list(live_result_paths)

  return AggregationResult(
    result_paths=result_paths,
    temp_dir=temp_dir
  )


def lift_job(arg: tuple[str, int, int], *, input_count: int, by: list[pl.Expr], spec: AggregationSpec, partition_bits: int, temp_dir: str):
  input_path, input_index, row_group_index = arg
  print(f"lifting input {input_index +
        1}/{input_count} group {row_group_index}")
  partitions_with_data = partition_and_lift_row_group(
    row_group_index,
    input_path=input_path, by=by, spec=spec,
    root_partition_bits=partition_bits, root_dir=temp_dir
  )
  return partitions_with_data


def fold_job_worker(queue: multiprocessing.Queue, active_task_count: Synchronized, result_paths: multiprocessing.managers.ListProxy):
  while True:
    try:
      job = queue.get(timeout=0.05)
    except multiprocessing.queues.Empty:
      job = None

    if job is None:
      with active_task_count.get_lock():
        if active_task_count.value == 0:
          break

      time.sleep(0.05)
      continue

    with active_task_count.get_lock():
      active_task_count.value += 1

    try:
      job_result = fold_or_finish_partition(job)
      if isinstance(job_result, str):
        result_paths.append(job_result)
      elif job_result is not None:
        for next_job in job_result:
          queue.put(next_job)

    finally:
      with active_task_count.get_lock():
        active_task_count.value -= 1


very_large_prime_numbers = [
  106013, 106019, 106031, 106033, 106087, 106103, 106109, 106121
]


def get_hash_expr(df: pl.DataFrame, by: list[str]):
  is_every_column_number = all(
    df.schema.get(col).is_numeric()
    for col in by
  )

  if is_every_column_number:
    first = by[0]
    rest = by[1:]
    hash_expr = (pl.col(first) * very_large_prime_numbers[0])
    for i, col in enumerate(rest):
      hash_expr += (pl.col(col) * very_large_prime_numbers[i + 1])
    return hash_expr % (1 << 32)

  first = by[0]
  rest = by[1:]
  hash_expr = pl.col(first).cast(pl.String)
  for col in rest:
    hash_expr += pl.col(col).cast(pl.String)
  return hash_expr.map_elements(xxhash.xxh32_intdigest, return_dtype=pl.UInt32)


def partition_and_lift_row_group(row_group_index: int, *, input_path: str, by: list[str], spec: AggregationSpec, root_partition_bits: int, root_dir: str):
  parquet_file = pq.ParquetFile(input_path)
  df = pl.from_arrow(parquet_file.read_row_group(row_group_index))
  parquet_file.close(force=True)

  groups = (
    df
      .with_columns(get_hash_expr(df, by).alias("__hash__"))
      .with_columns(
        (pl.col("__hash__") % get_partition_modulo(root_partition_bits))
          .alias("__partition__")
      )
      .group_by("__partition__")
  )

  partitions_with_data: set[int] = set()
  for (partition_index,), partition_df in groups:
    partition_id = get_partition_id(root_partition_bits, partition_index, 0, 0)
    partitions_with_data.add(partition_id)

    lifted_partition_df = (
      partition_df
        .group_by(by)
        .agg([pl.first("__hash__"), *spec.lift])
    )

    lifted_partition_df.write_parquet(os.path.join(
      get_partition_path(
        root_dir,
        partition_id
      ),
      f"group_{row_group_index}.parquet"
    ))

  return partitions_with_data


def fold_or_finish_partition(ctx: PartitionContext):
  partition_dir = ctx.root_dir

  files = os.listdir(partition_dir)
  sum_nonunique_groups = sum(
    count_parquet_rows(os.path.join(partition_dir, file))
    for file in files
  )

  if sum_nonunique_groups == 0:
    print(f"partition {ctx.id:08x} is empty")
    return None

  if sum_nonunique_groups < ctx.repartition_threshold:
    print(f"finishing partition {ctx.id:08x}")
    df_concat = pl.concat([
      pl.read_parquet(os.path.join(partition_dir, file))
      for file in files
    ])

    df_result = (
      df_concat
        .group_by(ctx.by)
        .agg(ctx.spec.fold)
        .select([*ctx.by, *ctx.spec.finish])
    )
    finished_path = os.path.join(partition_dir, "finished.parquet")
    df_result.write_parquet(finished_path)
    return finished_path

  print(f"folding partition {ctx.id:08x}")
  subpartition_bits_gain: int = min(max(math.ceil(math.log2(
    sum_nonunique_groups / ctx.repartition_threshold
  )), 0), 4)
  subpartition_bits = subpartition_bits_gain + ctx.bits
  for index in range(get_partition_modulo(subpartition_bits_gain)):
    os.makedirs(get_partition_path(
      partition_dir,
      get_partition_id(subpartition_bits, index, ctx.id, ctx.bits)
    ), exist_ok=True)

  subpartitions_with_data: set[int] = set()
  for file in files:
    df = pl.read_parquet(os.path.join(partition_dir, file))
    groups = df.with_columns(
      (
        pl.col("__hash__")
          % get_partition_modulo(subpartition_bits)
          // get_partition_modulo(ctx.bits)
      ).alias("__partition__")
    ).group_by("__partition__")

    for (subpartition_index,), subpartition_df in groups:
      subpartition_id = get_partition_id(
        subpartition_bits, subpartition_index, ctx.id, ctx.bits)
      subpartitions_with_data.add(subpartition_id)
      subpartition_df = (
        subpartition_df
          .group_by("__group__")
          .agg([pl.first("__hash__"), *ctx.spec.fold])
      )
      subpartition_df.write_parquet(os.path.join(
        get_partition_path(
          partition_dir,
          subpartition_id
        ),
        f"folded_{ctx.id:08x}.parquet"
      ))

  return [
    PartitionContext(
      id=subpartition_id,
      bits=subpartition_bits,
      root_dir=get_partition_path(partition_dir, subpartition_id),
      repartition_threshold=ctx.repartition_threshold,
      spec=ctx.spec,
      by=ctx.by
    )
    for subpartition_id in subpartitions_with_data
  ]
