import openai

class AnswersWithFT:
  def __init__(self):
    pass

  def create_context(self, question: str, search_file_id: str, max_len=1800, search_model="ada", max_rerank=10):
    results = openai.Engine(search_model).search(
      search_model=search_model,
      query=question,
      max_rerank=max_rerank,
      file=search_file_id,
      return_metadata=True,
    )
    returns = []
    cur_len = 0

    for result in results["data"]:
      cur_len += int(result["metadata"]) + 4
      if cur_len > max_len:
        break
      returns.append(result["text"])
    return "\n\n###\n\n".join(returns)

  def answer_question(
          self,
          search_file_id="<SEARCH_FILE_ID>",
          fine_tuned_qa_model="<FT_QA_MODEL_ID>",
          question="Which country won the European Football championship in 2021?",
          max_len=1800,
          search_model="ada",
          max_rerank=10,
          debug=False,
          stop_sequence=["\n", "."],
          max_tokens=100,
  ):
    context = self.create_context(
      question,
      search_file_id,
      max_len=max_len,
      search_model=search_model,
      max_rerank=max_rerank,
    )

    if debug:
      print("Context:\n" + context)
      print("\n\n")

    try:
      # fine-tuned models requires model parameter, whereas other models require engine parameter
      model_param = (
        {"model": fine_tuned_qa_model}
        if ":" in fine_tuned_qa_model
           and fine_tuned_qa_model.split(":")[1].startswith("ft")
        else {"engine": fine_tuned_qa_model}
      )
      response = openai.Completion.create(
        prompt=f"Answer the question based on the context below\n\nText: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_sequence,
        **model_param,
      )
      return response["choices"][0]["text"]
    except Exception as e:
      print(e)
      return ""

