import tempfile
import shutil

class Context:
  def __init__(self):
    pass

  def __enter__(self):
    self.temp_dir = tempfile.mkdtemp(prefix="multipolars_")

  def __exit__(self, exc_type, exc_val, exc_tb):
    shutil.rmtree(self.temp_dir)
