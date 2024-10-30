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
from typing import Union, Callable, Optional

import polars as pl
import pyarrow.parquet as pq
import xxhash
from pydantic import BaseModel

from .utils import count_parquet_rows


class AggregationSpec(BaseModel):
  lift: list[pl.Expr]
  fold: list[pl.Expr]
  finish: list[pl.Expr]

  class Config:
    arbitrary_types_allowed = True


class AggregationExpressionDefinition(BaseModel):
  default_alias: str
  get_spec: Callable[[Optional[str], Optional[str]], AggregationSpec]


class AggregationExpression(BaseModel):
  alias: Optional[str] = None
  prefix: Optional[str] = None
  definition: AggregationExpressionDefinition

  def alias(self, alias: str):
    return AggregationExpression(
      alias=alias,
      definition=self.definition
    )


def count_all():
  return AggregationExpression(
    definition=AggregationExpressionDefinition(
      default_alias="count",
      get_spec=lambda alias: AggregationSpec(
        lift=[pl.len().alias(alias or "count")],
        fold=[pl.sum(alias)],
        finish=[pl.col(alias)]
      )
    )
  )

def count(alias: str):
  return AggregationSpec(
    lift=[pl.len().alias(alias)],
    fold=[pl.sum(alias)],
    finish=[pl.col(alias)]
  )
