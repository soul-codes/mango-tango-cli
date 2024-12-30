from analyzer_interface.schema import (
    AttributeDict,
    CompositeDimension,
    CompositeDimensionDict,
    CompositeObjectSchema,
    Integer,
    Attribute,
    AtomicObjectSchema,
    Text,
)
from object_schemas import Message


class NgramAttributes(AttributeDict):
    words = Attribute(
        display_name="N-gram Words",
        type=Text(),
        description="The words that make up the n-gram",
    )
    length = Attribute(
        display_name="N-gram Length",
        type=Integer(),
        description="The number of words in the n-gram",
    )
    total_repetition_count = Attribute(
        display_name="Total Repetition Count",
        type=Integer(),
        description="The total number of times this ngram is repeated across all messages by all users.",
    )
    distinct_poster_count = Attribute(
        display_name="Distinct Poster Count",
        type=Integer(),
        description="The number of users who have posted this ngram.",
    )


Ngram = AtomicObjectSchema[NgramAttributes](schema_id="ngrams")


class MessageNgramAttributes(AttributeDict):
    occurrence_count = Attribute(type=Integer())


class MessageNgramComponents(CompositeDimensionDict):
    message = CompositeDimension(Message)
    ngram = CompositeDimension(Ngram)


MessageNgram = CompositeObjectSchema[MessageNgramComponents, MessageNgramAttributes]()
