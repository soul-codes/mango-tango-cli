from functools import cached_property
from typing import Generic, Optional, TypeVar, Iterator

T = TypeVar("TypeVarT")
V = TypeVar("TypeVarV")


class StaticDict(Generic[T]):
    __cache__ = None

    @cached_property
    @classmethod
    def __as_dict__(cls) -> dict[str, T]:
        """Discover static properties automatically."""
        if cls.__cache__ is None:
            cls.__cache__ = {
                key: value
                for key, value in vars(cls).items()
                if not key.startswith("_") and not callable(value)
            }
        return cls.__cache__

    @classmethod
    def __iter__(cls) -> Iterator[tuple[str, T]]:
        return iter(cls.__as_dict__().items())

    @classmethod
    def __getitem__(cls, key: str) -> Optional[T]:
        return cls.__as_dict__().get(key)

    @classmethod
    def __len__(cls):
        return len(cls.__as_dict__())

    @classmethod
    def __contains__(cls, key: str):
        return key in cls.__as_dict__()

    @classmethod
    def __repr__(cls):
        return f"StaticDict [{cls.__name__}]"

    @classmethod
    def __hash__(cls):
        return hash(cls.__as_dict__())

    @classmethod
    def __eq__(cls, other):
        return cls.__as_dict__() == other.__as_dict__()
