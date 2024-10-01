import os
import re
from typing import Callable, Literal

import platformdirs
import polars as pl
from filelock import FileLock
from tinydb import Query, TinyDB
from pydantic import BaseModel, Field

STORAGE_VERSION = 1


class Project(BaseModel):
  object_class: Literal["project"] = Field("project", alias='@class')
  id: str
  display_name: str


class Storage:
  def __init__(self, *, app_name: str, app_author: str):
    self.user_data_dir = platformdirs.user_data_dir(
      appname=app_name, appauthor=app_author, ensure_exists=True
    )
    self.temp_dir = platformdirs.user_cache_dir(
      appname=app_name, appauthor=app_author, ensure_exists=True
    )
    self.db = TinyDB(self._get_db_path())
    with self._lock_database():
      self._ensure_database_version()

  def init_project(self, *, display_name: str, input: pl.DataFrame):
    with self._lock_database():
      project_id = self._find_unique_project_id(display_name)
      project = Project(id=project_id, display_name=display_name)
      self.db.insert(project.model_dump())
      return project

    project_dir = self._get_project_path(project_id)
    os.makedirs(project_dir, exist_ok=True)

    input_path = self._get_project_input_path(project_id)
    input.write_parquet(input_path)

  def list_projects(self):
    q = Query()
    projects = self.db.search(q["@class"] == "project")
    return sorted(
      (Project(**project) for project in projects),
      key=lambda project: project.display_name
    )

  def load_project_input(self, project_id: str):
    input_path = self._get_project_input_path(project_id)
    return pl.read_parquet(input_path)

  def _ensure_database_version(self):
    """
    Makes sure that we have the correct database version

    This function will in future be extended to include any migration necessary
    to allow interoperability between versions.
    """
    q = Query()
    if self.db.search(q["@class"] == "version"):
      stored_version = self.db.search(q["@class"] == "version")[0]["version"]
      if stored_version != STORAGE_VERSION:
        raise Exception(
          f"Storage version mismatch: expected {
            STORAGE_VERSION}, got {stored_version}"
        )
    self.db.insert({"@class": "version", "version": STORAGE_VERSION})

  def _find_unique_project_id(self, display_name: str):
    """Turn the display name into a unique project ID"""
    return self._get_unique_name(self._slugify_name(display_name), self._is_project_id_unique)

  def _is_project_id_unique(self, project_id: str):
    """Check the database if the project ID is unique"""
    q = Query()
    return not self.db.search(q["@class"] == "project" and q["id"] == project_id)

  def _get_db_path(self):
    return os.path.join(self.user_data_dir, "db.json")

  def _get_project_path(self, project_id: str):
    return os.path.join(self.user_data_dir, "projects", project_id)

  def _get_project_input_path(self, project_id: str):
    return os.path.join(self._get_project_path(project_id), "input.parquet")

  def _lock_database(self):
    """
    Locks the database to prevent concurrent access, in case multiple instances
    of the application are running.
    """
    lock_path = os.path.join(self.temp_dir, "db.lock")
    return FileLock(lock_path)

  @staticmethod
  def _slugify_name(name: str):
    return re.sub(r'\W+', '_', name.lower()).strip("_")

  @staticmethod
  def _get_unique_name(base_name: str, validator: Callable[[str], bool]):
    if validator(base_name):
      return base_name
    i = 1
    while True:
      candidate = f"{base_name}_{i}"
      if validator(candidate):
        return candidate
      i += 1
