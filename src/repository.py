
from flask import Response, request
from src.app import App
from src.models import Message, Error



class Repository:
  def __init__(self, app: App):
    self.app = app

  def message(self):
    try:
      message = request.get_json()["message"]
    except:
      return Response(status=400, response=Error("Parameter message is required.").__json__(), mimetype="application/json")
    try:
      lastMessages = request.get_json()["lastMessages"]
    except:
      return Response(status=400, response=Error("Parameter message is required.").__json__(), mimetype="application/json")



    try:
      assistant = self.app.app.config["assistant"]

      userMessage = Message(message["id"], message["message"], message["createdAt"], message["state"], message["fromAssistant"])

      print(userMessage)

      response = ""# Message(assistant.ask(message, assistant.dataset_path, assistant.document_embeddings)).__json__()

      return Response(status=200, response=response, mimetype="application/json")
    except:
      return Response(status=500, response=Error("Something went wrong, try again later.").__json__(), mimetype="application/json")
