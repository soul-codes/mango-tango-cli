from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, Literal, Optional, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator
from typing_extensions import Annotated

from utils.static_dict import StaticDict

from .data_type import DataType, Identifier

type ObjectSchema = AtomicObjectSchema | CompositeObjectSchema

type AttributeContext = AtomicObjectAttributeContext | CompositeObjectAttributeContext


class BaseObjectAttributeContext(ABC, BaseModel):
    attribute_id: str

    @abstractmethod
    def get_column_name(self) -> str:
        pass


class AtomicObjectAttributeContext(BaseObjectAttributeContext):
    type: Literal["atomic"] = "atomic"
    object_schema: "AtomicObjectSchema"

    def get_column_name(self) -> str:
        return f"{self.object_schema.schema_id}_{self.attribute_id}"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, AtomicObjectAttributeContext):
            return (
                self.attribute_id == other.attribute_id
                and self.object_schema == other.object_schema
            )
        if isinstance(other, CompositeObjectAttributeContext):
            return other == self
        return False


class CompositeObjectDimensionContext(BaseModel):
    object_schema: "CompositeObjectSchema"
    dim_id: str

    def is_natural_dimension(self, atomic_schema: "AtomicObjectSchema") -> bool:
        return (
            self.object_schema.schema_id is None
            and atomic_schema.schema_id == self.dim_id
            and atomic_schema == self.object_schema.dims[self.dim_id]
        )

    def __repr__(self):
        return f"CompositeObjectDimensionContext({self.object_schema} [{self.dim_id}])"

    def __eq__(self, other):
        return (
            isinstance(other, CompositeObjectDimensionContext)
            and self.dim_id == other.dim_id
            and self.object_schema == other.object_schema
        )


class CompositeObjectAttributeContext(
    BaseObjectAttributeContext, CompositeObjectDimensionContext
):
    type: Literal["composite"] = "composite"

    def get_column_name(self) -> str:
        if self.object_schema.schema_id is not None:
            return f"{self.object_schema.schema_id}_{self.dim_id}_{self.attribute_id}"
        else:
            return f"{self.dim_id}_{self.attribute_id}"

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, CompositeObjectAttributeContext):
            return (
                self.attribute_id == other.attribute_id
                and self.dim_id == other.dim_id
                and self.object_schema == other.object_schema
            )
        if isinstance(other, AtomicObjectAttributeContext):
            return self.is_natural_dimension(other.object_schema)
        return False


class Attribute(BaseModel):
    type: DataType
    """
    The data type of the attribute.
    """

    display_name: str
    """
    The human readable, formatted name of the attribute.
    """

    description: str = ""
    """
    The description of the attribute. This is used to help the user understand
    what it's supposed to represent within the object schema.
    """

    user_column_name_hints: list[str] = []
    """
    Specifies a list of space-separated words that are likely to be found in the
    column name of the user-provided data. This is used to help the user map the
    input columns to the expected columns.

    Any individual hint matching is sufficient for a match to be called. The hint
    in turn is matched if every word matches some part of the column name.
    """

    _context: AttributeContext
    """
    The context in which the attribute lives in. The object schema
    assigns this.
    """

    @property
    def colname(self):
        """
        The globally unique column name for this attribute. Requires the attribute
        to be part of an object schema instantation.
        """
        return self._context.get_column_name()

    @property
    def col(self):
        """Shorthand for the column name"""
        return self.colname

    @property
    def pl(self):
        """
        The polars expression selecting the column for this attribute. Requires the attribute
        to be part of an object schema instantation.
        """
        from polars import col

        return col(self.colname)

    @property
    def context(self):
        """
        The attribute's object schema context.
        """
        return self._context

    def with_context(self, context: AttributeContext) -> "Attribute":
        attr = self.model_copy()
        attr._context = context
        return attr

    def __repr__(self):
        return f"Attribute({self.type} {self.colname})"

    def __eq__(self, value):
        return self is value or (
            isinstance(value, Attribute)
            and self._context.attribute_id == value._context.attribute_id
            and self.type == value.type
        )


class AttributeDict(StaticDict[Attribute]):
    pass


AttrDictType = TypeVar("AttrDictType", bound="AttributeDict", default="AttributeDict")


def _validate_attr_dict(value: AttrDictType):
    for key, attr in value.items():
        if key != attr.id:
            raise ValueError(f"Key {key} does not match attribute ID {attr.id}")
    return value


class BaseObjectSchema(BaseModel, Generic[AttrDictType]):
    display_name: str
    """
    The human readable, formatted name of the object schema.
    """

    description: str = ""
    """
    The description of the object schema. This is used to help the user understand
    what the object represents.
    """

    attrs: Annotated[AttrDictType, AfterValidator(_validate_attr_dict)]
    """
    The object schema's attributes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AtomicObjectSchema(BaseObjectSchema[AttrDictType], Generic[AttrDictType]):
    """
    Describes an object class whose primary key is derived from a single object.
    """

    kind: Literal["atomic"] = "atomic"

    schema_id: str
    """
    A stable, globally unique ID for the object schema.
    """

    @cached_property
    def id(self) -> Attribute:
        """
        The ID attribute of the object schema. This is taken to be unique.
        """
        attr = Attribute(
            display_name=f"{self.display_name} ID",
            type=Identifier(),
            description=f"The unique identifier of the {self.display_name}",
        ).with_context(
            AtomicObjectAttributeContext(attribute_id="id", object_schema=self)
        )

    def __init__(
        self, *, schema_id: str, display_name: str = "", description: str = ""
    ):
        data = {}
        data["schema_id"] = schema_id
        data["display_name"] = display_name or schema_id
        data["description"] = description
        data["attributes"] = self.__class__.__parameters__[0]
        super().__init__(**data)

    def model_post_init(self, __context):
        self.attrs = self.attrs.__class__()
        for attr_id, attr_spec in self.attrs:
            attr_spec = attr_spec.with_context(
                AtomicObjectAttributeContext(attribute_id=attr_id, object_schema=self)
            )

    def __eq__(self, value):
        return self is value or (
            isinstance(value, AtomicObjectSchema)
            and self.schema_id == value.schema_id
            and self.attrs == value.attrs
        )

    def __repr__(self):
        return (
            f"Object {self.schema_id} ("
            + ", ".join(map(lambda attr: str(attr), self.attrs))
            + f")"
        )


class CompositeDimension(AtomicObjectSchema[AttrDictType], Generic[AttrDictType]):
    component_description: str
    component_display_name: str
    cardinal_margin: bool
    """
    Defines whether the dimension is a cardinal margin. In a composite object schema,
    the collection's cardinality is determined by the product of the cardinalities of
    all dimensions marked as cardinal margins. Essentially, this ensures there is a
    single row in the composite object for each unique combination of cardinal margins.

    Examples:

    - A model where each message has exactly one author, but a user can author
      multiple messages. Here, the message is a cardinal margin.
    - A model where each message can have multiple tags, and each tag can be
      associated with multiple messages. In this case, there are no cardinal margins.

    This concept helps determine whether a composite object has a one-to-one relationship
    with a non-composite object that is one of its dimensions. This relationship defines
    whether the composite object can be represented as a single value or if it must be
    displayed as a collection in a table view:

    - In a message table, where each row represents a single message, the author can be
      represented as a single value.
    - In a user table, where each row represents a single user, messages must be represented
      as a collection.
    """

    _context: CompositeObjectDimensionContext

    def __init__(
        self,
        object_schema: AtomicObjectSchema[AttrDictType],
        *,
        component_description: str = "",
        component_display_name: str = "",
        cardinal_margin: bool = False,
    ):
        data = object_schema.model_dump()
        data["component_description"] = component_description
        data["component_display_name"] = component_display_name
        data["cardinal_margin"] = cardinal_margin
        attrs: AttributeDict = object_schema.attrs.__class__()
        for attr_id, attr_spec in attrs:
            setattr(attrs, attr_id, Attribute(**attr_spec.model_dump()))
        data["attrs"] = attrs
        super().__init__(**data)

    @property
    def context(self):
        """
        The dimension's object schema context.
        """
        return self._context

    def with_context(
        self, context: CompositeObjectDimensionContext
    ) -> "CompositeDimension":
        dim = self.model_copy()
        dim._context = context
        dim.attrs = dim.attrs.__class__()
        for attr_id, attr_spec in dim.attrs:
            attr_spec = attr_spec.with_context(
                CompositeObjectAttributeContext(
                    attribute_id=attr_id,
                    object_schema=context.object_schema,
                    dim_id=context.dim_id,
                )
            )
            setattr(dim.attrs, attr_id, attr_spec)
        return dim


class CompositeDimensionDict(StaticDict[CompositeDimension]):
    pass


CompositeDimensionDictType = TypeVar(
    "CompositeDimensionDictType",
    bound=CompositeDimensionDict,
    default=CompositeDimensionDict,
)


class CompositeObjectSchema(
    BaseObjectSchema[AttrDictType], Generic[CompositeDimensionDictType, AttrDictType]
):
    """
    Describes an object class whose primary key is dervied from multiple
    objects.
    """

    kind: Literal["composite"] = "composite"

    schema_id: Optional[str] = None
    """
    A stable, globally unique ID for the object schema. This is not needed,
    in which case the object class is taken to represent a natural/intuitive/
    "default" composition of the classes whose interpretation is obvious.

    When the schema ID is not provided, the component's attribute namespaces
    begin with the component ID itself, rather than the schema ID. In this case,
    each dimension ID is expected to match the schema ID of the the object
    it links exactly.
    """

    dims: CompositeDimensionDictType
    """
    The dimensions that make up the composite object.
    """

    def __init__(
        self,
        *,
        schema_id: Optional[str] = None,
        display_name: str = "",
        description: str = "",
    ):
        data = {}
        data["schema_id"] = schema_id
        data["display_name"] = display_name or schema_id or ""
        data["description"] = description
        data["components"] = self.__class__.__parameters__[0]
        data["attributes"] = self.__class__.__parameters__[1]
        super().__init__(**data)

    @model_validator(mode="after")
    def _validate_idless_dimensions(self):
        if self.schema_id:
            return self

        for dim_id, dim_spec in self.dims:
            dim_spec: CompositeDimension = dim_spec
            if dim_spec.schema_id != dim_id:
                raise ValueError(
                    f"Composite dimension {dim_id} does not match its schema ID {dim_spec.schema_id} "
                    "Either match them or provide a schema ID for the composite object."
                )

    def model_post_init(self, __context):
        self.dims = self.dims.__class__()
        for dim_id, dim_spec in self.dims:
            dim_spec = dim_spec.with_context(
                CompositeObjectDimensionContext(object_schema=self, dim_id=dim_id)
            )
            setattr(self.dims, dim_id, dim_spec)

        self.display_name = self.display_name or (
            "("
            + ", ".join(
                dim_spec.component_display_name or dim_spec.display_name
                for _, dim_spec in self.dims
            )
            + ")"
        )

    def __eq__(self, value):
        return self is value or (
            isinstance(value, CompositeObjectSchema)
            and self.schema_id == value.schema_id
            and self.dims == value.dims
            and self.attrs == value.attrs
        )

    def __repr__(self):
        return (
            (
                f"Composite Object {self.schema_id}"
                if self.schema_id
                else "Composite Object"
            )
            + " ["
            + ", ".join([f"{dim_id}: {dim_spec}" for dim_id, dim_spec in self.dims])
            + "] ("
            + ", ".join(
                [f"{attr_id}: {attr_spec}" for attr_id, attr_spec in self.attrs]
            )
            + ")"
        )
