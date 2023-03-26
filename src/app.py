import os
from flask import Flask



class App:
  def __init__(self, app, assistant, host: str = os.environ.get("HOST"), port: int = int(os.environ.get("PORT"))):
    self.host = host
    self.port = port
    self.app = Flask(app)
    self.app.config["assistant"] = assistant

  def run(self):
    self.app.run(host=self.host, port=self.port, debug=True, threaded=True)

  def add_repo(self, methods=None, endpoint=None, name=None, handler=None):
    self.app.add_url_rule(endpoint=name, rule=endpoint, view_func=handler, methods=methods)
