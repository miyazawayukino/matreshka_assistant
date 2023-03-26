from src.utils import Model
class Error(Model):
  def __init__(self, error: str):
    self.error = error