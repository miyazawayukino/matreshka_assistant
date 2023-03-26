import json

class Model():
  def __json__(self):
    return json.dumps(self.__dict__)
