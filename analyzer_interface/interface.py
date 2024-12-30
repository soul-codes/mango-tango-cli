from typing import Literal, Optional

import polars as pl
from pydantic import BaseModel, model_validator

from .schema import Attribute, ObjectClass


class BaseAnalyzerInterface(BaseModel):
    id: str
    """
  The static ID for the analyzer that, with the version, uniquely identifies the
  analyzer and will be stored as metadata as part of the output data.
  """

    version: str
    """
  The version ID for the analyzer. In future, we may choose to support output
  migration between versions of the same analyzer.
  """

    name: str
    """
  The short human-readable name of the analyzer.
  """

    short_description: str
    """
  A short, one-liner description of what the analyzer does.
  """

    long_description: Optional[str] = None
    """
  A longer description of what the analyzer does that will be shown separately.
  """

    view_presets: Optional[list["TableView"]] = None
    """
    Contributes a view preset to the system that can be used to export data
    that has run this analyzer.
    """


class AnalyzerInput(BaseModel):
    object_class: ObjectClass
    required_attributes: list[Attribute]

    @model_validator(mode="after")
    def _validate_attributes(self):
        object_attributes = set(value for _, value in self.object_class.attrs)
        for attr in self.required_attributes:
            if attr not in object_attributes:
                raise ValueError(
                    f"Attribute {attr} is specified by the analyzer input but not found in the object schema"
                )
        return self


class AnalyzerOutput(BaseModel):
    id: str
    """
    Uniquely identifies the output data schema for the analyzer. The analyzer
    must include this key in the output dictionary.
    """

    name: str
    """The human-friendly for the output."""

    description: Optional[str] = None

    object_class: ObjectClass
    output_attributes: list[Attribute]

    internal: bool = False

    def get_column_by_name(self, name: str):
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def transform_output(self, output_df: pl.LazyFrame | pl.DataFrame):
        output_columns = output_df.lazy().collect_schema().names()
        return output_df.select(
            [
                pl.col(col_name).alias(
                    output_spec.human_readable_name_or_fallback()
                    if output_spec
                    else col_name
                )
                for col_name in output_columns
                if (output_spec := self.get_column_by_name(col_name)) or True
            ]
        )


class TableViewColumn(BaseModel):
    attribute: Attribute
    """
    The attributes that will be included in the export.

    If the table view's primary object class is itse composite, and the attribute
    is one of the components, the attribute will simply be repeated to correlate
    with the primary object.

    Currently, attribute included in the view must either be
    - a direct attribute of the primary object class, where it matches
      one-to-one, or
    - an attribute of a composite dimension of the primary object class, if
      the primary object class is composite, where it matches one-to-one.
    - an attribute of an atomic object class that is a composite dimension of
      the primary object class, if the primary object class is composite, where
      it is repeated for each matching composite instance.

    The case where the primary object class is atomic and the attribute is
    one of a composite object class containing the primary object class is
    not yet supported.
    """

    column_id: Optional[str]
    column_human_readable_name: Optional[str]

    def __init__(
        self,
        attribute: Attribute,
        *,
        column_id: Optional[str] = None,
        column_human_readable_name: Optional[str] = None,
    ):
        super().__init__(
            attribute=attribute,
            column_id=column_id,
            column_human_readable_name=column_human_readable_name,
        )


class TableView(BaseModel):
    id: str
    name: str

    primary_object: ObjectClass
    """The object class that will be represented as a single row in the export."""

    columns: list[TableViewColumn]


class AnalyzerInterface(BaseAnalyzerInterface):
    input: AnalyzerInput
    """
    Specifies the input data schema for the analyzer.
    """

    outputs: list["AnalyzerOutput"]
    """
    Specifies the output data schema for the analyzer.
    """

    kind: Literal["primary"] = "primary"


class DerivedAnalyzerInterface(BaseAnalyzerInterface):
    base_analyzer: AnalyzerInterface
    """
  The base analyzer that this secondary analyzer extends. This is always a primary
  analyzer. If your module depends on other secondary analyzers (which must have
  the same base analyzer), you can specify them in the `depends_on` field.
  """

    depends_on: list["SecondaryAnalyzerInterface"] = []
    """
  A dictionary of secondary analyzers that must be run before the current analyzer
  secondary analyzer is run. These secondary analyzers must have the same
  primary base.
  """


class SecondaryAnalyzerInterface(DerivedAnalyzerInterface):
    outputs: list[AnalyzerOutput]
    """
  Specifies the output data schema for the analyzer.
  """

    kind: Literal["secondary"] = "secondary"


class WebPresenterInterface(DerivedAnalyzerInterface):
    kind: Literal["web"] = "web"
