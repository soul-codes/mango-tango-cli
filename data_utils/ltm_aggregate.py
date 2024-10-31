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
from typing import Union, Optional, Callable, Any
import psutil
import polars as pl
import pyarrow.parquet as pq
import xxhash
from pydantic import BaseModel

from .count_parquet_rows import count_parquet_rows

HASH_BITS = 64
HASH_DTYPE = pl.UInt64
HASH_FN = xxhash.xxh3_64_intdigest


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


class PartitionContext(BaseModel):
  id: int
  bits: int
  root_dir: str
  partition_size: int
  spec: Optional[AggregationSpec]
  by: list[str]
  having: Optional[pl.Expr]
  transform: Optional[Callable[[pl.DataFrame], pl.DataFrame]]

  class Config:
    arbitrary_types_allowed = True


class AggregationResult(BaseModel):
  temp_dir: str
  result_paths: list[str]

  def __enter__(self):
    return self.result_paths

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.clean()

  def clean(self):
    rmtree(self.temp_dir)

  def merge(self, *, clean=True):
    try:
      return pl.concat([
        pl.read_parquet(result_path)
        for result_path in self.result_paths
      ])
    finally:
      if clean:
        self.clean()

  def sink_parquet(self, output_path: str, *, clean=True):
    schema = pq.read_schema(self.result_paths[0])
    try:
      with pq.ParquetWriter(output_path, schema) as writer:
        for result_path in self.result_paths:
          with pq.ParquetFile(result_path) as reader:
            for batch in reader.iter_batches():
              writer.write_batch(batch)
    finally:
      if clean:
        self.clean()


def get_partition_id(bits: int, current_index: int, parent_id: int, parent_bits: int):
  return parent_id + ((current_index * (1 << (HASH_BITS - bits))))


def get_partition_modulo(partition_bits: int):
  return 1 << partition_bits


def get_partition_path(root_dir: str, id: int):
  return os.path.join(root_dir, "partitions", format(id, "016x"))


def get_default_partition_size():
  total_ram = psutil.virtual_memory().total
  total_ram_gb = total_ram / (1024**3)
  cpu_count = os.cpu_count() or 2
  return int(3_000_000 * math.sqrt(total_ram_gb / 16) / (cpu_count / 12))


def ltm_aggregate(
    lf: Union[pl.LazyFrame, pl.DataFrame, list[str], AggregationResult],
    by: list[str],
    spec: Optional[AggregationSpec],
    *,
    having: Optional[pl.Expr] = None,
    partition_size=8_000_000,
    transform_partition: Optional[
      Callable[[pl.DataFrame, Any], pl.DataFrame]] = None,
    transform_initialize: Optional[Callable[[Any], Any]] = None,
    transform_initialize_arg: Optional[Any] = None
  ) -> AggregationResult:
  temp_dir = tempfile.TemporaryDirectory().name
  os.makedirs(temp_dir, exist_ok=True)
  print(f"temp dir is {temp_dir}")

  if isinstance(lf, AggregationResult):
    input_paths = lf.result_paths
  elif isinstance(lf, list):
    input_paths = lf
  else:
    lf = lf.lazy()
    print(f"persisting parquet")

    input_path = os.path.join(temp_dir, "input.parquet")
    lf.sink_parquet(input_path)
    input_paths = [input_path]

  total_rows = sum(
    count_parquet_rows(input_path)
    for input_path in input_paths
  )

  if total_rows < partition_size:
    df = pl.concat([
      pl.read_parquet(input_path)
      for input_path in input_paths
    ])

    if spec:
      df = df.group_by(by).agg(spec.lift).select([*by, *spec.finish])
    else:
      df = df.select(by).unique()

    output_path = os.path.join(temp_dir, "finished.parquet")
    df.write_parquet(output_path)

    return AggregationResult(
      result_paths=[output_path],
      temp_dir=temp_dir
    )

  metadata_by_input: list = [
    pq.ParquetFile(input_path).metadata
    for input_path in input_paths
  ]

  row_group_count_by_input: list[int] = [
    metadata.num_row_groups
    for metadata in metadata_by_input
  ]

  total_row_groups = sum(row_group_count_by_input)

  cardinality_ratio = sum(
    sample.n_unique() / sample.height
    for input_path in input_paths[:10]
    if (pa_table := pq.ParquetFile(input_path)) is not None
    if (sample := pl.from_arrow(pa_table.read_row_group(0, columns=by))) is not None
  ) / len(input_paths[:10])
  print(f"Average cardinality ratio is {cardinality_ratio}")

  partition_bits = min(max(
    math.ceil(
      math.log2(total_rows * cardinality_ratio / partition_size)
    ),
    0
  ), 10)

  print(f"Input has {total_rows} rows in {
      total_row_groups} row groups across {len(input_paths)} file(s)")
  print(f"Using {get_partition_modulo(partition_bits)} initial partitions")

  for partition_index in range(get_partition_modulo(partition_bits)):
    partition_id = get_partition_id(partition_bits, partition_index, 0, 0)
    os.makedirs(get_partition_path(
      temp_dir,
      partition_id
    ), exist_ok=True)

  def get_row_group_slices(metadata):
    num_row_groups: int = metadata.num_row_groups
    num_rows: int = metadata.num_rows
    row_group_size = num_rows / num_row_groups
    num_row_groups_per_processing_group = max(1, math.floor(
      partition_size / row_group_size
    ))

    return (
      list(range(i, min(i + num_row_groups_per_processing_group, num_row_groups)))
      for i in range(0, num_row_groups, num_row_groups_per_processing_group)
    )

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
        (input_path, input_index, row_group_indices)
        for input_index, input_path in enumerate(input_paths)
        for row_group_indices in get_row_group_slices(metadata_by_input[input_index])
      )
    ))

  job_queue = multiprocessing.Queue()
  for partition_id in partitions_with_data:
    job_queue.put(PartitionContext(
      id=partition_id,
      bits=partition_bits,
      root_dir=get_partition_path(temp_dir, partition_id),
      partition_size=partition_size,
      spec=spec,
      by=by,
      having=having,
      transform=transform_partition
    ))

  active_task_count = multiprocessing.Value("i", 0)
  processes: list[multiprocessing.Process] = []
  with multiprocessing.Manager() as manager:
    live_result_paths = manager.list()

    for worker_index in range(os.cpu_count()):
      process = multiprocessing.Process(
        target=fold_job_worker,
        args=(
          job_queue, active_task_count,
          live_result_paths, transform_initialize,
          transform_initialize_arg
        )
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


def lift_job(arg: tuple[str, int, list[int]], *, input_count: int, by: list[pl.Expr], spec: Optional[AggregationSpec], partition_bits: int, temp_dir: str):
  input_path, input_index, row_group_indices = arg
  print(f"lifting input {input_index +
        1}/{input_count} group(s) {min(row_group_indices)}-{max(row_group_indices)}")
  partitions_with_data = partition_and_lift_row_group(
    row_group_indices,
    input_index=input_index,
    input_path=input_path, by=by, spec=spec,
    root_partition_bits=partition_bits, root_dir=temp_dir
  )
  return partitions_with_data


def fold_job_worker(
  queue: multiprocessing.Queue,
  active_task_count: Synchronized,
  result_paths: multiprocessing.managers.ListProxy,
  transform_initialize: Optional[Callable[[Any], Any]],
  transform_initialize_arg: Optional[Any]
):
  transform_arg = transform_initialize(
    transform_initialize_arg) if transform_initialize else None

  while True:
    try:
      job = queue.get(timeout=0.05)
    except multiprocessing.queues.Empty:
      job = None

    if job is None:
      with active_task_count.get_lock():
        if active_task_count.value == 0:
          break

      time.sleep(0.01)
      continue

    with active_task_count.get_lock():
      active_task_count.value += 1

    try:
      job_result = fold_or_finish_partition(job, transform_arg)
      if isinstance(job_result, str):
        result_paths.append(job_result)
      elif job_result is not None:
        for next_job in job_result:
          queue.put(next_job)

    finally:
      with active_task_count.get_lock():
        active_task_count.value -= 1


INTEGER_HASH_PRIMES = [
  106013, 106019, 106031, 106033, 106087, 106103, 106109, 106121
]
PRIME_MAX_BITS = 17


def get_hash_expr(df: pl.DataFrame, by: list[str]):
  is_every_column_integer = all(
    df.schema.get(col).is_integer()
    for col in by
  )

  if is_every_column_integer:
    first = by[0]
    rest = by[1:]
    prime_multiplier_modulo = (1 << (HASH_BITS - PRIME_MAX_BITS))
    hash_expr = (pl.col(first).cast(HASH_DTYPE, wrap_numerical=True) %
                 prime_multiplier_modulo * INTEGER_HASH_PRIMES[0])
    for i, col in enumerate(rest):
      hash_expr = (hash_expr.cast(HASH_DTYPE, wrap_numerical=True) ^ pl.col(
        col).cast(pl.UInt64)) % prime_multiplier_modulo * INTEGER_HASH_PRIMES[i + 1]
    return hash_expr

  first = by[0]
  rest = by[1:]
  hash_expr = pl.col(first).cast(pl.String)
  for col in rest:
    hash_expr += pl.col(col).cast(pl.String)
  return hash_expr.map_elements(HASH_FN, return_dtype=HASH_DTYPE)


def partition_and_lift_row_group(row_group_indices: list[int], *, input_index: int, input_path: str, by: list[str], spec: Optional[AggregationSpec], root_partition_bits: int, root_dir: str):
  parquet_file = pq.ParquetFile(input_path)
  df = pl.from_arrow(parquet_file.read_row_groups(row_group_indices))
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

    if spec:
      lifted_partition_df = (
        partition_df
          .group_by(by)
          .agg([pl.first("__hash__"), *spec.lift])
      )
    else:
      lifted_partition_df = partition_df.select(["__hash__", *by]).unique(by)

    lifted_partition_df.write_parquet(os.path.join(
      get_partition_path(
        root_dir,
        partition_id
      ),
      f"input_{input_index}_groups_{
        min(row_group_indices)}-{max(row_group_indices)}.parquet"
    ))

  return partitions_with_data


def fold_or_finish_partition(ctx: PartitionContext, transform_arg: Any):
  partition_dir = ctx.root_dir

  files = os.listdir(partition_dir)
  sum_nonunique_groups = sum(
    count_parquet_rows(os.path.join(partition_dir, file))
    for file in files
  )

  if sum_nonunique_groups == 0:
    print(f"partition {ctx.id:016x} is empty")
    return None

  if sum_nonunique_groups < ctx.partition_size:
    print(f"finishing partition {ctx.id:016x} ({
          sum_nonunique_groups} items in {len(files)} files)")
    df_concat = pl.concat([
      pl.read_parquet(os.path.join(partition_dir, file))
      for file in files
    ])

    if ctx.spec:
      df_result = (
        df_concat
          .group_by(ctx.by)
          .agg(ctx.spec.fold)
          .select([*ctx.by, *ctx.spec.finish])
      )
    else:
      df_result = df_concat.select(ctx.by).unique()

    if ctx.having is not None:
      df_result = df_result.filter(ctx.having)

    if ctx.transform is not None:
      df_result = ctx.transform(df_result, transform_arg)

    finished_path = os.path.join(partition_dir, "finished.parquet")
    df_result.write_parquet(finished_path)
    return finished_path

  print(f"folding partition {ctx.id:016x} ({
        sum_nonunique_groups} items in {len(files)} files)")
  subpartition_bits_gain: int = min(max(math.ceil(math.log2(
    sum_nonunique_groups / ctx.partition_size
  )), 0), 4)
  subpartition_bits = subpartition_bits_gain + ctx.bits
  for index in range(get_partition_modulo(subpartition_bits_gain)):
    os.makedirs(get_partition_path(
      partition_dir,
      get_partition_id(subpartition_bits, index, ctx.id, ctx.bits)
    ), exist_ok=True)

  subpartitions_with_data: set[int] = set()
  subpartition_dfs: dict[int, list[pl.DataFrame]] = {}
  subpartition_heights: dict[int, int] = {}
  for file_index, file in enumerate(files):
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

      if ctx.spec:
        subpartition_df = (
          subpartition_df
            .group_by(ctx.by)
            .agg([pl.first("__hash__"), *ctx.spec.fold])
        )
      else:
        subpartition_df = subpartition_df.select(
          ["__hash__", *ctx.by],).unique()

      if subpartition_heights.get(subpartition_id, 0) + subpartition_df.height > ctx.partition_size:
        subpartition_df_list = subpartition_dfs.pop(subpartition_id, [])
        concat_subpartition_df = pl.concat(subpartition_df_list)
        concat_subpartition_df.write_parquet(os.path.join(
          get_partition_path(
            partition_dir,
            subpartition_id
          ),
          f"folded_{ctx.id:016x}_{file_index - 1}.parquet"
        ))
        subpartition_heights[subpartition_id] = 0
        print(f"flushed {subpartition_id:016x} at file {file_index - 1} with {
              concat_subpartition_df.height} items from {len(subpartition_df_list)} file(s)")

      subpartition_dfs.setdefault(subpartition_id, []).append(subpartition_df)
      subpartition_heights[subpartition_id] = subpartition_heights.get(
        subpartition_id, 0) + subpartition_df.height

  # flush remaining subpartition buffers
  for subpartition_id, subpartition_df_list in subpartition_dfs.items():
    concat_subpartition_df = pl.concat(subpartition_df_list)
    concat_subpartition_df.write_parquet(os.path.join(
      get_partition_path(
        partition_dir,
        subpartition_id
      ),
      f"folded_{ctx.id:016x}_final.parquet"
    ))
    print(f"flushed {subpartition_id:016x} at end with {
          concat_subpartition_df.height} items from {len(subpartition_df_list)} file(s)")

  return [
    PartitionContext(
      id=subpartition_id,
      bits=subpartition_bits,
      root_dir=get_partition_path(partition_dir, subpartition_id),
      partition_size=ctx.partition_size,
      spec=ctx.spec,
      by=ctx.by,
      having=ctx.having,
      transform=ctx.transform
    )
    for subpartition_id in subpartitions_with_data
  ]
