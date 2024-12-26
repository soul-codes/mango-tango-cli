from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Optional

import polars as pl
from pydantic import BaseModel

from analyzer_interface import AnalyzerOutput


class ProjectModel(BaseModel):
    class_: Literal["project"] = "project"
    id: str
    display_name: str


class SettingsModel(BaseModel):
    class_: Literal["settings"] = "settings"
    export_chunk_size: Optional[int | Literal[False]] = None


class StatesModel(BaseModel):
    class_: Literal["states"] = "states"
    last_path: Optional[str] = None


class AnalysisModel(BaseModel):
    class_: Literal["analysis"] = "analysis"
    analysis_id: str
    project_id: str
    display_name: str
    primary_analyzer_id: str
    path: str
    column_mapping: Optional[dict[str, str]] = None
    create_timestamp: Optional[float] = None
    is_draft: bool = False

    def create_time(self):
        return (
            datetime.fromtimestamp(self.create_timestamp)
            if self.create_timestamp
            else None
        )


SupportedOutputExtension = Literal["parquet", "csv", "xlsx", "json"]


class StorageBackend(ABC):
    @abstractmethod
    def init_project(self, *, display_name: str, input_temp_file: str) -> ProjectModel:
        pass

    @abstractmethod
    def list_projects(self) -> list[ProjectModel]:
        pass

    @abstractmethod
    def get_project(self, project_id: str) -> Optional[ProjectModel]:
        pass

    @abstractmethod
    def delete_project(self, project_id: str) -> None:
        pass

    @abstractmethod
    def save_project(self, project: ProjectModel) -> None:
        pass

    @abstractmethod
    def load_project_input(
        self, project_id: str, *, n_records: Optional[int] = None
    ) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_project_input_stats(self, project_id: str) -> "TableStats":
        pass

    @abstractmethod
    def get_primary_output_parquet_path(
        self, analysis: AnalysisModel, output_id: str
    ) -> str:
        pass

    @abstractmethod
    def get_secondary_output_parquet_path(
        self, analysis: AnalysisModel, secondary_id: str, output_id: str
    ) -> str:
        pass

    @abstractmethod
    def export_project_primary_output(
        self,
        analysis: AnalysisModel,
        output_id: str,
        *,
        extension: SupportedOutputExtension,
        spec: AnalyzerOutput,
        export_chunk_size: Optional[int] = None,
    ) -> str:
        pass

    @abstractmethod
    def export_project_secondary_output(
        self,
        analysis: AnalysisModel,
        secondary_id: str,
        output_id: str,
        *,
        extension: SupportedOutputExtension,
        spec: AnalyzerOutput,
        export_chunk_size: Optional[int] = None,
    ):
        pass

    @abstractmethod
    def list_project_analyses(self, project_id: str) -> list[AnalysisModel]:
        pass

    @abstractmethod
    def init_analysis(
        self,
        project_id: str,
        display_name: str,
        primary_analyzer_id: str,
        column_mapping: dict[str, str],
    ) -> AnalysisModel:
        pass

    @abstractmethod
    def save_analysis(self, analysis: AnalysisModel) -> None:
        pass

    @abstractmethod
    def delete_analysis(self, analysis: AnalysisModel) -> None:
        pass

    @abstractmethod
    def list_secondary_analyses(self, analysis: AnalysisModel) -> list[str]:
        pass

    @abstractmethod
    def get_project_primary_output_root_path(self, analysis: AnalysisModel) -> str:
        pass

    @abstractmethod
    def get_project_secondary_output_root_path(
        self, analysis: AnalysisModel, secondary_id: str
    ) -> str:
        pass

    @abstractmethod
    def get_project_exports_root_path(self, analysis: AnalysisModel) -> str:
        pass

    @abstractmethod
    def get_web_presenter_state_path(
        self, analysis: AnalysisModel, presenter_id: str
    ) -> str:
        pass

    @abstractmethod
    def get_settings(self) -> SettingsModel:
        pass

    @abstractmethod
    def save_settings(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_states(self) -> StatesModel:
        pass

    @abstractmethod
    def set_states(self, **kwargs) -> None:
        pass


class TableStats(BaseModel):
    num_rows: int
