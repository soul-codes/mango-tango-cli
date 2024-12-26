from tempfile import NamedTemporaryFile

from pydantic import BaseModel

from importing import ImporterSession
from terminal_tools.prompts import FileSelectorStateManager

from .app_context import AppContext
from .project_context import ProjectContext


class App(BaseModel):
    context: AppContext

    def list_projects(self):
        return [
            ProjectContext(model=project, app_context=self.context)
            for project in self.context.storage.list_projects()
        ]

    def create_project(self, name: str, importer_session: ImporterSession):
        with NamedTemporaryFile(delete=False) as temp_file:
            importer_session.import_as_parquet(temp_file.name)
        project_model = self.context.storage.init_project(
            display_name=name, input_temp_file=temp_file.name
        )
        return ProjectContext(model=project_model, app_context=self.context)

    @property
    def file_selector_state(self):
        return AppFileSelectorStateManager(self.context)


class AppFileSelectorStateManager(FileSelectorStateManager):
    def __init__(self, context: AppContext):
        self.storage = context.storage

    def get_current_path(self):
        return self.storage.get_states().last_path

    def set_current_path(self, path: str):
        self.storage.set_states(last_path=path)
