from pydantic import BaseModel
from typing import Literal


type DataType = Identifier | Text


class Identifier(BaseModel):
    type_id: Literal["identifier"] = "identifier"


class Text(BaseModel):
    type_id: Literal["text"] = "text"


class DateTime(BaseModel):
    type_id: Literal["datetime"] = "datetime"


class Tags(BaseModel):
    type_id: Literal["tags"] = "tags"
