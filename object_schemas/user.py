from analyzer_interface.schema import (
    AtomicObjectSchema,
    Attribute,
    AttributeDict,
    DateTime,
    Text,
)


class UserAttribute(AttributeDict):
    username = Attribute(
        display_name="Username",
        type=Text(),
        description="The username of the user",
        user_column_name_hints=[
            "username",
            "user",
            "poster",
            "screen name",
            "user name",
            "name",
            "email",
        ],
    )
    email = Attribute(
        display_name="Email",
        type=Text(),
        description="The email address of the user",
        user_column_name_hints=["email"],
    )
    join_date = Attribute(
        display_name="Join Date",
        type=DateTime(),
        description="The date the user joined the platform",
        user_column_name_hints=["join date", "join time", "join"],
    )


User = AtomicObjectSchema[UserAttribute](schema_id="user")
