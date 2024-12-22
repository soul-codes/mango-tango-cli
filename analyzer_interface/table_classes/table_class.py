from pydantic import BaseModel
from .dtypes import DataType


class ColumnSchema(BaseModel):
    id: str
    display_name: str
    data_type: DataType
    name_hints: list[str]
    """
    Specifies a list of space-separated words that are likely to be found in the
    column name of the user-provided data. This is used to help the user map the
    input columns to the expected columns.

    Any individual hint matching is sufficient for a match to be called. The hint
    in turn is matched if every word matches some part of the column name.
    """


class TableSchema(BaseModel):
    id: str
    display_name: str
    description: str = ""
    columns: list[ColumnSchema]
