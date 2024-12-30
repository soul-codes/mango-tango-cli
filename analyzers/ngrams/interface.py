from analyzer_interface import AnalyzerInput, AnalyzerInterface, AnalyzerOutput
from object_schemas import Message, MessageAuthor

from .schema import MessageNgram, Ngram

OUTPUT_MESSAGE_NGRAMS = "message_ngrams"
OUTPUT_NGRAMS = "ngrams"


interface = AnalyzerInterface(
    id="ngrams",
    version="0.1.0",
    name="N-gram Analysis",
    short_description="Extracts n-grams from text data",
    long_description="""
The n-gram analysis extract n-grams (sequences of n words) from the text data
in the input and counts the occurrences of each n-gram in each message, linking
the message author to the ngram frequency.

The result can be used to see if certain word sequences are more common in
the corpus of text, and whether certain authors use these sequences more often.
  """,
    input=AnalyzerInput(
        object_class=Message,
        required_attributes=[Message.attrs.text, MessageAuthor.dims.user.id],
    ),
    outputs=[
        AnalyzerOutput(
            id=OUTPUT_MESSAGE_NGRAMS,
            name="N-gram count per message",
            internal=True,
            object_class=MessageNgram,
            output_attributes=[MessageNgram.attrs.occurrence_count],
        ),
        AnalyzerOutput(
            id=OUTPUT_NGRAMS,
            name="N-gram definitions",
            internal=True,
            object_class=Ngram,
            output_attributes=[
                Ngram.attrs.words,
                Ngram.attrs.length,
                Ngram.attrs.total_repetition_count,
                Ngram.attrs.distinct_poster_count,
            ],
            description="The word compositions of each unique n-gram",
        ),
    ],
)
