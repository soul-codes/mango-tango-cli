from analyzer_interface import (
    AnalyzerOutput,
    SecondaryAnalyzerInterface,
    TableView,
    TableViewColumn,
)


from ..ngrams import interface as ngrams_interface
from ..ngrams.schema import Ngram, MessageNgram


OUTPUT_NGRAM_STATS = "ngram_stats"


interface = SecondaryAnalyzerInterface(
    id="ngram_stats",
    version="0.1.0",
    name="Copy-Pasta Detector",
    short_description="",
    base_analyzer=ngrams_interface,
    outputs=[
        AnalyzerOutput(
            id=OUTPUT_NGRAM_STATS,
            name="N-gram repetition statistics",
            object_class=Ngram,
            columns=[
                Ngram.attrs.total_repetition_count,
                Ngram.attrs.distinct_poster_count,
            ],
        ),
    ],
    view_presets=[
        TableView(
            id="ngram_report",
            name="N-gram Report",
            primary_object=MessageNgram,
            columns=[
                TableViewColumn(MessageNgram.dims.ngram.id),
                TableViewColumn(MessageNgram.dims.ngram.attrs.words),
                TableViewColumn(MessageNgram.dims.ngram.attrs.total_repetition_count),
                TableViewColumn(MessageNgram.dims.ngram.attrs.distinct_poster_count),
                TableViewColumn(MessageNgram.dims.message.id),
                TableViewColumn(MessageNgram.dims.message.attrs.author.type),
                TableViewColumn(MessageNgram.dims.message.attrs.text),
                TableViewColumn(MessageNgram.dims.message.attrs.timestamp),
                TableViewColumn(MessageNgram.attrs.occurrence_count),
            ],
        ),
    ],
)
