from functools import cached_property
from typing import Literal

import polars as pl
from pydantic import BaseModel

from analyzer_interface import UserInputColumn as BaseUserInputColumn
from analyzer_interface.schema import ObjectSchema, Attribute
from preprocessing.series_semantic import SeriesSemantic, infer_series_semantic


from .app_context import AppContext
from .project_context import ProjectContext


type Assignment = VerbatimAssignment | DateTimeConcatAssignment | ArrayDelimitedAssignment


class ProjectShapeModel(BaseModel):
    base_object_schema_id: str
    attribute_assignments: list[Assignment] = []


class VerbatimAssignment(BaseModel):
    type: Literal["verbatim"] = "verbatim"
    column: str
    attribute: Attribute


class DateTimeConcatAssignment(BaseModel):
    type: Literal["datetime_from_separate_date_and_time"] = (
        "datetime_from_separate_date_and_time"
    )
    date_column: str
    time_column: str
    attribute: Attribute


class ArrayDelimitedAssignment(BaseModel):
    type: Literal["array_delimited"]
    column: str
    delimiter: str
    quote_char: str
    trim_left: str
    trim_right: str
    attribute: Attribute


class ProjectShapeContext(BaseModel):
    model: ProjectShapeModel
    project_context: ProjectContext
    app_context: AppContext
