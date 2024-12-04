import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa

from analyzer_interface.context import SecondaryAnalyzerContext

from ..ngrams.interface import (COL_AUTHOR_ID, COL_MESSAGE_ID, COL_MESSAGE_SURROGATE_ID,
                                COL_MESSAGE_NGRAM_COUNT, COL_NGRAM_ID,
                                COL_NGRAM_LENGTH, COL_NGRAM_WORDS,
                                OUTPUT_MESSAGE, OUTPUT_MESSAGE_NGRAMS,
                                OUTPUT_NGRAM_DEFS, COL_MESSAGE_TEXT, COL_MESSAGE_TIMESTAMP)
from .interface import (COL_NGRAM_DISTINCT_POSTER_COUNT, COL_NGRAM_TOTAL_REPS,
                        OUTPUT_NGRAM_FULL, OUTPUT_NGRAM_STATS)


def main(context: SecondaryAnalyzerContext):
  df_message_ngrams = pl.read_parquet(
    context.base.table(OUTPUT_MESSAGE_NGRAMS).parquet_path
  )
  df_ngrams = pl.read_parquet(
    context.base.table(OUTPUT_NGRAM_DEFS).parquet_path
  )
  df_messages = pl.read_parquet(
    context.base.table(OUTPUT_MESSAGE).parquet_path
  )

  # Avoid an intermediate cardinality explosion by processing each ngram
  # group separately. There is no point in materializing the entire
  # cross-join of ngrams and messages since we're only interested in the
  # statistics.
  df_ngram_stats = pl.DataFrame([
    {
      COL_NGRAM_ID: ngram_id,
      COL_NGRAM_TOTAL_REPS: ngram_group[COL_MESSAGE_NGRAM_COUNT].sum(),
      COL_NGRAM_DISTINCT_POSTER_COUNT: df_messages.filter(
        pl.col(COL_MESSAGE_SURROGATE_ID)
          .is_in(ngram_group[COL_MESSAGE_SURROGATE_ID])
      )[COL_AUTHOR_ID].n_unique()
    }
    for (ngram_id,), ngram_group in df_message_ngrams.group_by(COL_NGRAM_ID)
  ]).filter(
    pl.col(COL_NGRAM_TOTAL_REPS) > 1
  )

  df_ngram_summary = df_ngrams.join(
    df_ngram_stats,
    on=COL_NGRAM_ID,
    how="inner"
  ).sort([
    COL_NGRAM_LENGTH,
    COL_NGRAM_TOTAL_REPS,
    COL_NGRAM_DISTINCT_POSTER_COUNT
  ], descending=True)

  df_ngram_summary.write_parquet(
    context.output(OUTPUT_NGRAM_STATS).parquet_path
  )

  df_messages_schema = df_messages.to_arrow().schema
  df_message_ngrams_schema = df_message_ngrams.to_arrow().schema
  df_ngram_summary_schema = df_ngram_summary.to_arrow().schema

  with pq.ParquetWriter(
    context.output(OUTPUT_NGRAM_FULL).parquet_path,
    schema=pa.schema([
      df_message_ngrams_schema.field(COL_NGRAM_ID),
      df_ngram_summary_schema.field(COL_NGRAM_LENGTH),
      df_ngram_summary_schema.field(COL_NGRAM_WORDS),
      df_ngram_summary_schema.field(COL_NGRAM_TOTAL_REPS),
      df_ngram_summary_schema.field(COL_NGRAM_DISTINCT_POSTER_COUNT),
      df_messages_schema.field(COL_AUTHOR_ID),
      df_message_ngrams_schema.field(COL_MESSAGE_NGRAM_COUNT),
      df_messages_schema.field(COL_MESSAGE_SURROGATE_ID),
      df_messages_schema.field(COL_MESSAGE_ID),
      df_messages_schema.field(COL_MESSAGE_TEXT),
      df_messages_schema.field(COL_MESSAGE_TIMESTAMP)
    ])
  ) as writer:
    for ngram_row in df_ngram_summary.iter_rows(named=True):
      df_output = (
        df_message_ngrams
          .filter(pl.col(COL_NGRAM_ID) == pl.lit(ngram_row[COL_NGRAM_ID]))
          .join(df_messages, on=COL_MESSAGE_SURROGATE_ID)
          .with_columns([
            pl.lit(
              ngram_value,
              dtype=df_ngram_summary.schema.get(ngram_column)
            ).alias(ngram_column)
            for ngram_column, ngram_value in ngram_row.items()
          ])
      ).select([
        COL_NGRAM_ID,
        COL_NGRAM_LENGTH,
        COL_NGRAM_WORDS,
        COL_NGRAM_TOTAL_REPS,
        COL_NGRAM_DISTINCT_POSTER_COUNT,
        COL_AUTHOR_ID,
        COL_MESSAGE_NGRAM_COUNT,
        COL_MESSAGE_SURROGATE_ID,
        COL_MESSAGE_ID,
        COL_MESSAGE_TEXT,
        COL_MESSAGE_TIMESTAMP
      ]).sort(
        [COL_MESSAGE_NGRAM_COUNT],
        descending=True
      )
      writer.write_table(df_output.to_arrow())
