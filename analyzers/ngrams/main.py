import re

import polars as pl

from analyzer_interface.context import PrimaryAnalyzerContext
from analyzer_interface.schema import Attribute
from object_schemas import MessageAuthor, Message
from terminal_tools import ProgressReporter
from utils.static_dict import StaticDict

from .interface import OUTPUT_MESSAGE_NGRAMS, OUTPUT_NGRAMS
from .schema import MessageNgram, Ngram


class Attrs(StaticDict[Attribute]):
    MessageId = Message.id
    NgramId = Ngram.id
    MessageNgramCount = MessageNgram.attrs.occurrence_count
    Text = Message.attrs.text
    Author = MessageAuthor.dims.user.id


def main(context: PrimaryAnalyzerContext):
    input_reader = context.input()
    df_input = input_reader.preprocess(pl.read_parquet(input_reader.parquet_path))
    with ProgressReporter("Preprocessing messages"):
        df_input = df_input.filter(
            Attrs.Text.pl.is_not_null()
            & (Attrs.Text.pl != "")
            & Attrs.Author.pl.is_not_null()
            & (Attrs.Author.pl != "")
        )

    with ProgressReporter("Generating n-grams") as progress:

        def get_ngram_rows(ngrams_by_id: dict[str, int]):
            nonlocal progress
            num_rows = df_input.height
            current_row = 0
            for row in df_input.iter_rows(named=True):
                tokens = tokenize(row[Attrs.MessageId.nam])
                for ngram in ngrams(tokens, 3, 5):
                    serialized_ngram = serialize_ngram(ngram)
                    if serialized_ngram not in ngrams_by_id:
                        ngrams_by_id[serialized_ngram] = len(ngrams_by_id)
                    ngram_id = ngrams_by_id[serialized_ngram]
                    yield {
                        Attrs.MessageId.col: row[Attrs.MessageId.col],
                        Attrs.NgramId.col: ngram_id,
                    }
                current_row = current_row + 1
                if current_row % 100 == 0:
                    progress.update(current_row / num_rows)

        ngrams_by_id: dict[str, int] = {}
        df_ngram_instances = pl.DataFrame(get_ngram_rows(ngrams_by_id))

    with ProgressReporter("Computing n-gram occurrence count per message"):
        df_ngram_message = (
            pl.DataFrame(df_ngram_instances)
            .group_by(Attrs.MessageId.col, Attrs.MessageId.col)
            .agg(pl.count().alias(MessageNgram.attrs.occurrence_count.col))
            .rename(
                {
                    Attrs.MessageId.col: MessageNgram.dims.message.id.col,
                    Attrs.NgramId.col: MessageNgram.dims.ngram.id.col,
                }
            )
        )
        df_ngram_message.write_parquet(
            context.output(OUTPUT_MESSAGE_NGRAMS).parquet_path
        )

    with ProgressReporter("Computing ngram statistics"):
        dict_authors_by_message = {
            row[MessageAuthor.dims.message.id.col]: row[MessageAuthor.dims.user.id.col]
            for row in df_input.iter_rows(named=True)
        }

        df_ngrams = pl.DataFrame(
            {
                Ngram.id.col: list(ngrams_by_id.values()),
                Ngram.attrs.words.col: list(ngrams_by_id.keys()),
            }
        ).with_columns(
            [
                Ngram.attrs.words.pl.str.split(" ")
                .list.len()
                .alias(Ngram.attrs.length.col)
            ]
        )

        df_ngrams = (
            df_ngram_message.with_columns(
                MessageNgram.attrs.occurrence_count.pl.sum()
                .over([MessageNgram.dims.ngram.id.col])
                .alias(Ngram.attrs.total_repetition_count.col)
            )
            .filter(Ngram.attrs.total_repetition_count.col > 1)
            .group_by(Ngram.id.col)
            .agg(
                Ngram.attrs.total_repetition_count.pl.first().alias(
                    Ngram.attrs.total_repetition_count.col
                ),
                MessageAuthor.dims.message.id.pl.replace_strict(dict_authors_by_message)
                .n_unique()
                .alias(Ngram.attrs.distinct_poster_count.col),
            )
            .with_columns(
                Ngram.id.pl.replace_strict(ngrams_by_id).alias(Ngram.attrs.words.col)
            )
            .with_columns(
                Ngram.attrs.words.pl.str.split(" ")
                .list.len()
                .alias(Ngram.attrs.length.col)
            )
        )

        df_ngrams.write_parquet(context.output(OUTPUT_NGRAMS).parquet_path)


def tokenize(input: str) -> list[str]:
    """Generate words from input string."""
    return re.split(r"\W+", input.lower())


def ngrams(tokens: list[str], min: int, max: int):
    """Generate n-grams from list of tokens."""
    for i in range(len(tokens) - min + 1):
        for n in range(min, max + 1):
            if i + n > len(tokens):
                break
            yield tokens[i : i + n]


def serialize_ngram(ngram: list[str]) -> str:
    """Generates a string that uniquely represents an ngram"""
    return " ".join(ngram)
