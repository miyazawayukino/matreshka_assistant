from src import MatreshkaAssistant, App, Repository


if __name__ == "__main__":

  assistant = MatreshkaAssistant(dataset_path="document_library/dataset_001.csv")
  assistant.load_data_frame(keys=["title", "heading"], sample=5)
  assistant.compute_doc_embeddings()

  app = App(app='MatreshkaAssistant', assistant=assistant)
  repository = Repository(app=app)

  app.add_repo(methods=["POST"], endpoint='/message', name='message', handler=repository.message)
  app.run()

