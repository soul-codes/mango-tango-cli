from .table_class import TableSchema, ColumnSchema
from .dtypes import Identifier, Text, DateTime, Tags


class StaticDict:
    __cache__ = None

    @classmethod
    def __as_dict__(cls):
        """Discover static properties automatically."""
        if cls.__cache__ is None:
            cls.__cache__ = {
                key: value
                for key, value in vars(cls).items()
                if not key.startswith("_") and not callable(value)
            }
        return cls.__cache__

    @classmethod
    def __iter__(cls):
        return iter(cls.__as_dict__().items())

    @classmethod
    def __getitem__(cls, key):
        return cls.__as_dict__()[key]

    @classmethod
    def __len__(cls):
        return len(cls.__as_dict__())

    @classmethod
    def __contains__(cls, key):
        return key in cls.__as_dict__()

    @classmethod
    def __repr__(cls):
        return f"StaticDict [{cls.__name__}]"


class Foo(StaticDict):
    a = 1
    b = 2
    c = 3


class Message:
    id = "id"
    text = "text"

    def __init__(self):
        super().__init__(
            id="message",
            display_name="Message",
            columns=(
                [
                    ColumnSchema(
                        id="message_id",
                        display_name="Message ID",
                        data_type="identifier",
                        name_hints=["message", "id"],
                    ),
                    ColumnSchema(
                        id="message_text",
                        display_name="Message Text",
                        data_type="text",
                        name_hints=["message", "text"],
                    ),
                    ColumnSchema(
                        id="message_time",
                        display_name="Message Time",
                        data_type="datetime",
                        name_hints=["message", "time"],
                    ),
                ],
            ),
        )
