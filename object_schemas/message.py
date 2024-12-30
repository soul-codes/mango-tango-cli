from analyzer_interface.schema import (
    AttributeDict,
    DateTime,
    Attribute,
    AtomicObjectSchema,
    CompositeObjectSchema,
    CompositeDimension,
    CompositeDimensionDict,
    Text,
)
from .user import User
from .hashtag import Hashtag


class MessageAttributes(AttributeDict):
    text = Attribute(
        display_name="Message Text",
        type=Text(),
        description="The text content of the message",
        user_column_name_hints=[
            "message",
            "text",
            "comment",
            "post",
            "body",
            "content",
            "tweet",
        ],
    )
    timestamp = Attribute(
        display_name="Message Timestamp",
        type=DateTime(),
        description="The time at which the message was posted",
        user_column_name_hints=["time", "timestamp", "date", "ts"],
    )


Message = AtomicObjectSchema[MessageAttributes](schema_id="message")
"""Models social media messages"""


class MessageAuthorDimensions(CompositeDimensionDict):
    user = CompositeDimension(User)
    """The user that authors the message"""

    message = CompositeDimension(Message, cardinal_margin=True)
    """The message that the user authored"""


MessageAuthor = CompositeObjectSchema[MessageAuthorDimensions]()


class MessageHashtagDimensions(CompositeDimensionDict):
    message = CompositeDimension(Message)
    """The message that the user authored"""

    hashtag = CompositeDimension(Hashtag)
    """The tags associated with the message"""


MessageHashtag = CompositeObjectSchema[MessageHashtagDimensions]()
