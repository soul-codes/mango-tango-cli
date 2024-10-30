import polars as pl
from typing import Union
from .context import Context

class DataFrame:
  def __init__(self, input: Union[pl.DataFrame, str, list[str]],*,context: Context):
    self.context = context
    self.input = input
