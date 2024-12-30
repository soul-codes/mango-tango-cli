from typing import Literal

from pydantic import BaseModel

from analyzer_interface.schema import Attribute

type Assignment = VerbatimAssignment | DateTimeConcatAssignment | ArrayDelimitedAssignment


class VerbatimAssignmentSettings(BaseModel):
    type: Literal["verbatim"] = "verbatim"
    column: str


class VerbatimAssignment(VerbatimAssignmentSettings):
    attribute: Attribute


class DateTimeConcatAssignmentSettings(BaseModel):
    type: Literal["datetime_from_separate_date_and_time"] = (
        "datetime_from_separate_date_and_time"
    )
    date_column: str
    time_column: str


class DateTimeConcatAssignment(DateTimeConcatAssignmentSettings):
    attribute: Attribute


class ArrayDelimitedAssignmentSettings(BaseModel):
    type: Literal["array_delimited"]
    column: str
    delimiter: str
    quote_char: str
    trim_left: str
    trim_right: str


class ArrayDelimitedAssignment(ArrayDelimitedAssignmentSettings):
    attribute: Attribute
