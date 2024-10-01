import csv
import os
from typing import Optional

import polars as pl

from analyzer_interface import (AnalyzerInterface, InputColumn,
                                UserInputColumn, column_automap,
                                get_data_type_compatibility_score)
from analyzers import all_analyzers
from preprocessing.series_semantic import infer_series_semantic
from storage import Project, Storage
from terminal_tools import draw_box, prompts, wait_for_key
from terminal_tools.inception import TerminalContext


def new_project(context: TerminalContext, storage: Storage):
  with context.nest(draw_box("1. Data Source", padding_lines=0)):
    print("Select a file for your analysis")
    selected_file = prompts.file_selector("Select a file")
    if selected_file is None:
      print("Canceled")
      return wait_for_key(True)

    print(f"Selected file: {selected_file}")
    confirm_file = prompts.confirm("Is this correct?", default=True)
    if not confirm_file:
      print("Canceled")
      return wait_for_key(True)

    file_extension: str = os.path.splitext(selected_file)[1].lower()
    if file_extension == ".csv":
      try:
        with context.nest("Reading CSV file..."):
          print("Opening file...")
          with open(selected_file, "r", encoding="utf8") as file:
            dialect = csv.Sniffer().sniff(file.read(65536))

          df = pl.read_csv(
            selected_file,
            separator=dialect.delimiter,
            quote_char=dialect.quotechar,
            ignore_errors=True,
            has_header=True,
            truncate_ragged_lines=True,
          )
      except Exception as e:
        print(f"Error reading CSV file: {e}")
        wait_for_key(True)
        return

    else:
      print(
        f"Unsupported file type: {
          file_extension or '(file with no extension)'}"
      )
      wait_for_key(True)
      return

  with context.nest(draw_box("2. Data preview", padding_lines=0)):
    input_preview(df)
    wait_for_key(True)

  with context.nest(draw_box("3. Project Setup", padding_lines=0)):
    print("Choose a project name. You can reload the project later.")
    suggested_project_name = os.path.splitext(
      os.path.basename(selected_file))[0]
    project_name = prompts.text(
      "Project name", default=suggested_project_name
    )

    project = storage.init_project(display_name=project_name, input=df)
    print("Project created!")
    wait_for_key(True)
    return ProjectInstance(
      id=project.id,
      display_name=project.display_name,
      input=df
    )


def select_project(context: TerminalContext, storage: Storage):
  while True:
    with context.nest(draw_box("Choose a project", padding_lines=0)):
      projects = storage.list_projects()
      if not projects:
        print("There are no previously created projects.")
        wait_for_key(True)
        return None

      project = prompts.list_input(
        "Which project?",
        choices=[
          (project.display_name, project)
          for project in projects
        ],
      )

      if project is None:
        return None

    with context.nest(draw_box(f"Project: {project.display_name}", padding_lines=0)):
      df = storage.load_project_input(project.id)
      input_preview(df)
      confirm_load = prompts.confirm("Load this project?", default=True)
      if confirm_load:
        return ProjectInstance(
          id=project.id,
          display_name=project.display_name,
          input=df
        )


class ProjectInstance(Project):
  input: pl.DataFrame

  class Config:
    arbitrary_types_allowed = True


def project_main(context: TerminalContext, storage: Storage, project: ProjectInstance):
  while True:
    with context.nest(draw_box(f"CIB Mango Tree/Project: {project.display_name}", padding_lines=0)):
      action = prompts.list_input(
        "What would you like to do?",
        choices=[
          ("New analysis", "new_analysis"),
          ("(Back)", "back"),
        ],
      )

    if action is None:
      return

    if action == "back":
      return

    if action == "new_analysis":
      new_analysis(context, storage, project)


def new_analysis(context: TerminalContext, storage: Storage, project: ProjectInstance):
  df = project.input
  with context.nest(draw_box("Choose an analysis", padding_lines=0)):
    analyzer: Optional[AnalyzerInterface] = prompts.list_input(
      "Which analysis?",
      choices=[
        ("Exit", None),
        *(
          (f"{analyzer.name} ({analyzer.short_description})", analyzer)
          for analyzer in all_analyzers
        ),
      ],
    )

  if analyzer is None:
    return

  with context.nest(draw_box(analyzer.name, padding_lines=0)):
    with context.nest(analyzer.long_description or analyzer.short_description):
      wait_for_key(True)

    user_columns = get_user_columns(df)
    draft_column_mapping = column_automap(
      user_columns,
      analyzer.input.columns
    )

    want_to_change_mapping = False
    final_column_mapping = None
    while True:
      with context.nest("Column mapping") as column_mapping_scope:
        has_unmapped_column = any(
          draft_column_mapping.get(input_column.name, None) is None
          for input_column in analyzer.input.columns
        )

        if has_unmapped_column or want_to_change_mapping:
          selected_analyzer_column: Optional[InputColumn] = prompts.list_input(
            "Choose the column mapping to change",
            choices=[
              (f"{input_column.name} -> {user_column or '(no mapping)'}", input_column)
              for input_column in analyzer.input.columns
              if (user_column := draft_column_mapping.get(input_column.name, None)) or True
            ],
          )

          column_mapping_scope.refresh()
          if selected_analyzer_column is None:
            break

          print(f"Mapping {selected_analyzer_column.name}")
          print(f"Type: {selected_analyzer_column.data_type}")
          print(selected_analyzer_column.description or "")
          print("")

          selected_user_column: Optional[UserInputColumn] = prompts.list_input(
            "Choose the user column",
            choices=[
              (f"{user_column.name} [{user_column.data_type}]", user_column)
              for user_column in user_columns
              if get_data_type_compatibility_score(
                selected_analyzer_column.data_type, user_column.data_type
              ) is not None
            ],
          )

          if selected_user_column is not None:
            draft_column_mapping[selected_analyzer_column.name] = selected_user_column.name

          want_to_change_mapping = False
          continue

        # At this point, every column is mapped
        for input_column in analyzer.input.columns:
          user_column = draft_column_mapping.get(input_column.name, None)
          print(f"{input_column.name} -> {user_column}")

        want_to_change_mapping = not prompts.confirm(
          "Are you happy with this mapping?",
          default=False
        )
        if not want_to_change_mapping:
          final_column_mapping = draft_column_mapping
          break

    if final_column_mapping is None:
      return

    try:
      result = analyzer.entry_point(
        df.select(
          pl.col(user_col).alias(input_col)
          for input_col, user_col in final_column_mapping.items()
        )
      )

      for output in analyzer.outputs:
        output_df = result[output.id]
        print(f"Output: {output.name}")
        print(output_df)

      os.makedirs("analysis_outputs", exist_ok=True)
      for output in analyzer.outputs:
        print(f"Saving {output.name} to 'analysis_outputs/{output.id}.csv'")
        output_df: pl.DataFrame = result[output.id]
        output_df.write_csv(f"analysis_outputs/{output.id}.csv")
      print("Analysis results saved to 'analysis_outputs' folder")

    except KeyboardInterrupt:
      print("Canceled")

    wait_for_key(True)


def input_preview(df: pl.DataFrame):
  user_columns = get_user_columns(df)
  print(df)
  print("Inferred column semantics:")
  for col in user_columns:
    print(f"  {col.name}: {col.data_type}")


def get_user_columns(df: pl.DataFrame):
  return [
    UserInputColumn(name=col, data_type=semantic.data_type)
    for col in df.columns
    if (semantic := infer_series_semantic(df[col])) is not None
  ]
