from typing import Literal

from pydantic import BaseModel

type DataType = Identifier | Text | Text | DateTime | Tags | Integer


class BaseDataType(BaseModel):
    type_id: str

    def __hash__(self):
        return hash(self.type_id)


class Identifier(BaseDataType):
    type_id: Literal["identifier"] = "identifier"


class Text(BaseDataType):
    type_id: Literal["text"] = "text"


class DateTime(BaseDataType):
    type_id: Literal["datetime"] = "datetime"


class Tags(BaseDataType):
    type_id: Literal["tags"] = "tags"


class Integer(BaseDataType):
    type_id: Literal["integer"] = "integer"
