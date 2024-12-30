from pydantic import BaseModel

from .schema import (
    AtomicObjectSchema,
    Attribute,
    AttributeDict,
    CompositeDimension,
    CompositeObjectSchema,
    CompositeObjectDimensionContext,
    ObjectSchema,
)

from typing import Literal

type RelatedAttributeCardinality = Literal[""]


class KeyRelation(BaseModel):
    attributes: list[Attribute]


class NonKeyRelation(BaseModel):
    attribute: Attribute


class PrimaryKeyAttribute(KeyRelation):
    relation_kind: Literal["primary_key"]


class DependentAttribute(NonKeyRelation):
    relation_kind: Literal["dependent"]


class ForeignKeyAttributeOneToMany(KeyRelation):
    relation_kind: Literal["foreign_key_one_to_many"]


class ForeignDependentAttributeOneToMany(NonKeyRelation):
    relation_kind: Literal["foreign_dependent_one_to_many"]


class ForeignKeyAttributeManyToOne(KeyRelation):
    relation_kind: Literal["foreign_key_many_to_one"]


class ForeignDependentAttributeManyToOne(NonKeyRelation):
    relation_kind: Literal["foreign_dependent_many_to_one"]


class SchemaResolver:
    def __init__(self, object_schemas: list[ObjectSchema]):
        (
            atomic_schemas_by_id,
            composite_schemas,
            composite_dimension_contexts_by_atomic_id,
        ) = _validate_unique_object_schemas(object_schemas)
        self._atomic_schemas_by_id = atomic_schemas_by_id
        self._composite_schemas = composite_schemas
        self._composite_dimension_contexts_by_atomic_id = (
            composite_dimension_contexts_by_atomic_id
        )
        self._attributes_by_colname = _validate_unique_attributes(
            atomic_schemas_by_id, composite_schemas
        )
        _validate_distinct_natural_compositions(
            self._composite_dimension_contexts_by_atomic_id
        )

    def get_related_attributes(
        self, object_schema: ObjectSchema
    ) -> dict[str, Attribute]:
        self_attrs = {attr.colname: attr for _, attr in object_schema.attrs}
        if object_schema.kind == "composite":
            for _, dim_spec in object_schema.dims:
                self_attrs.update({attr.colname: attr for _, attr in dim_spec.attrs})
        else:
            composite_dimension_contexts = (
                self._composite_dimension_contexts_by_atomic_id.get(
                    object_schema.schema_id, []
                )
            )
            for composite_dimension_context in composite_dimension_contexts:
                composite_schema = composite_dimension_context.object_schema
                for _, attr in composite_schema.attrs:
                    self_attrs[attr.colname] = attr
                for _, dim_spec in composite_schema.dims:
                    self_attrs.update(
                        {attr.colname: attr for _, attr in dim_spec.attrs}
                    )


def _validate_unique_object_schemas(object_schemas: list[ObjectSchema]):
    atomic_schemas_by_id: dict[str, AtomicObjectSchema] = dict()
    composite_schemas: list[CompositeObjectSchema] = []
    composite_dimension_contexts_by_atomic_id: dict[
        str, list[CompositeObjectDimensionContext]
    ] = dict()

    for object_schema in object_schemas:
        if object_schema.kind == "atomic":
            existing_atomic_schema = atomic_schemas_by_id.get(object_schema.schema_id)
            if existing_atomic_schema:
                if existing_atomic_schema != object_schema:
                    raise ValueError(
                        f"Non-identical atomic schemas with the same ID `{object_schema.schema_id}`: {existing_atomic_schema} vs {object_schema}"
                    )
            else:
                atomic_schemas_by_id[object_schema.schema_id] = object_schema

    for object_schema in object_schemas:
        if object_schema.kind == "composite":
            composite_schemas.append(object_schema)
            for _, dim_spec in object_schema.dims:
                existing_atomic_schema = atomic_schemas_by_id.get(dim_spec.schema_id)
                if existing_atomic_schema:
                    if existing_atomic_schema != dim_spec:
                        raise ValueError(
                            f"Non-identical atomic schemas with the same ID `{dim_spec.schema_id}`: {existing_atomic_schema} vs {dim_spec} defined in composite schema {object_schema}"
                        )
                else:
                    atomic_schemas_by_id[dim_spec.schema_id] = dim_spec

                composite_dimension_contexts_by_atomic_id.setdefault(
                    dim_spec.schema_id, []
                ).append(dim_spec.context)

    return (
        atomic_schemas_by_id,
        composite_schemas,
        composite_dimension_contexts_by_atomic_id,
    )


def _validate_unique_attributes(
    atomic_schemas: dict[str, AtomicObjectSchema],
    composite_schemas: list[CompositeObjectSchema],
):
    attributes_by_colname: dict[str, Attribute] = dict()
    for atomic_schema in atomic_schemas.values():
        attrs = [
            *(attr for _, attr in atomic_schema.attrs),
            atomic_schema.id,
        ]
        for attr in attrs:
            existing_attr = attributes_by_colname.get(attr.colname)
            if existing_attr:
                if existing_attr != attr:
                    raise ValueError(
                        f"Non-identical attributes with the same column name `{attr.colname}`: {existing_attr} vs {attr}"
                    )
            else:
                attributes_by_colname[attr.colname] = attr

    for composite_schema in composite_schemas:
        for _, dim_spec in composite_schema.dims:
            attrs = [
                *(attr for _, attr in dim_spec.attrs),
                dim_spec.id,
            ]
            for attr in attrs:
                existing_attr = attributes_by_colname.get(attr.colname)
                if existing_attr:
                    if existing_attr != attr:
                        raise ValueError(
                            f"Non-identical attributes with the same column name `{attr.colname}`: {existing_attr} vs {attr}"
                        )
                else:
                    attributes_by_colname[attr.colname] = attr

    return attributes_by_colname


def _validate_distinct_natural_compositions(
    composite_dimension_contexts_by_atomic_id: dict[
        str, list[CompositeObjectDimensionContext]
    ]
):
    for (
        atomic_id,
        composite_dimension_contexts,
    ) in composite_dimension_contexts_by_atomic_id.items():
        related_atomic_id_contexts: dict[str, CompositeObjectDimensionContext] = {}
        for composite_dimension_context in composite_dimension_contexts:

            composite_schema = composite_dimension_context.object_schema
            if composite_schema.schema_id is not None:
                continue
            for dim_id, dim_spec in composite_schema.dims:
                assert dim_spec.schema_id == dim_id
                if dim_spec.schema_id == atomic_id:
                    continue
                existing_related_atomic_id_context = related_atomic_id_contexts.get(
                    dim_spec.schema_id
                )
                if existing_related_atomic_id_context:
                    if (
                        existing_related_atomic_id_context
                        != composite_dimension_context
                    ):
                        raise ValueError(
                            f"Atomic schema `{atomic_id}` is implicitly related to {dim_spec.schema_id} in multiple ways: {related_atomic_id_contexts[dim_spec.schema_id]} and {composite_dimension_context}"
                        )
                else:
                    related_atomic_id_contexts[dim_spec.schema_id] = (
                        composite_dimension_context
                    )

        if len(composite_dimension_contexts) > 1:
            raise ValueError(
                f"Atomic schema `{atomic_id}` is part of multiple composite schemas: {composite_dimension_contexts}"
            )

    return composite_dimension_contexts_by_atomic_id
