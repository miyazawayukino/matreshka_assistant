from src.utils import Model

# {
#   "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#   "message": "string",
#   "createdAt": "2023-03-10T10:13:05.836Z",
#   "state": "WAIT_HANDLING",
#   "fromAssistant": true
# }
class Message(Model):
  def __init__(self, message: str):
    self.message = message