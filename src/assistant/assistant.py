import os
from loguru import logger
import pandas as pd
import numpy as np
import tiktoken
import openai


class MatreshkaAssistant:
  def __init__(
          self,
          COMPLETIONS_MODEL: str = os.environ.get("COMPLETIONS_MODEL"),
          EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL"),
          MAX_SECTION_LEN: int = int(os.environ.get("MAX_SECTION_LEN")),
          SEPARATOR: str = os.environ.get("SEPARATOR"),
          ENCODING: str = os.environ.get("ENCODING"),
          TEMPERATURE: float = float(os.environ.get("TEMPERATURE")),
          MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS")),
          dataset_path: str = None
  ):
    self.COMPLETIONS_MODEL = COMPLETIONS_MODEL
    self.EMBEDDING_MODEL = EMBEDDING_MODEL
    self.MAX_SECTION_LEN = MAX_SECTION_LEN
    self.SEPARATOR = SEPARATOR
    self.ENCODING = ENCODING
    self.COMPLETIONS_API_PARAMS = {
      "temperature": TEMPERATURE,
      "max_tokens": MAX_TOKENS,
      "model": "gpt-4",
      "frequency_penalty": 0,
      "presence_penalty": 0,
      "top_p": 1
    }
    self.dataset_path = dataset_path
    self.data_frame = None
    self.encoding = tiktoken.get_encoding(encoding_name=self.ENCODING)
    self.separator_len = len(tiktoken.get_encoding(encoding_name=ENCODING).encode(text=SEPARATOR))
    self.document_embeddings = None
    self.type_q = "!general_survey"

    logger.info(f"Context separator contains {self.separator_len} tokens")


  def load_data_frame(self, keys: list[str], sample: int = 5):
    """
    Загрузка датасета.
    :param str path: Ссылка или путь к csv файлу.
    :param list[str] keys: Установки списка, серии или фрейма данных в качестве индекса фрейма данных.
    :param int sample: Возврат случайной выборки элементов с оси объекта.
    """
    self.data_frame = pd.read_csv(self.dataset_path)
    self.data_frame = self.data_frame.set_index(keys)
    logger.info(f"{len(self.data_frame)} rows in the data.")
    self.data_frame.sample(sample)

  # def load_embeddings(self, dataset: str):
  #   """
  #   Прочитать вложения документов и их ключи из CSV
  #   dataset - путь к файлу CSV с точно такими столбцами:
  #       "title", "heading", "0", "1", ... up to the length of the embedding vectors.
  #   """
  #   df = pd.read_csv(dataset, header=0)
  #   max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
  #   self.document_embeddings = {
  #     (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
  #   }

  def compute_doc_embeddings(self):
    """
    Возвращает словарь, который сопоставляет каждый вектор вложения с индексом строки, которой он соответствует.
    :return dict[tuple[str, str], list[float]]:
    """
    #
    self.document_embeddings = {
      idx: self.get_embedding(input=r.content) for idx, r in self.data_frame.iterrows()
    }

  def get_embedding(self, input: str) -> list[float]:
    result = openai.Embedding.create(
      model=self.EMBEDDING_MODEL,
      input=input
    )
    return result["data"][0]["embedding"]

  def vector_similarity(self, x: list[float], y: list[float]) -> float:
    """
    Возвращает сходство между двумя векторами.
    :param list[float] x: First vector
    :param list[float] y: Second vector
    :return float:
    """
    return np.dot(np.array(x), np.array(y))

  def order_document_sections_by_query_similarity(self, input: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Поиск вложение запроса для предоставленного запроса и сравнение его со всеми предварительно рассчитанными вложениями документа чтобы найти наиболее релевантные разделы.
    Возвращает список разделов документа, отсортированных по релевантности в порядке убывания.
    :param str input:
    :param dict[(str, str) contexts:
    :return list:
    """
    query_embedding = self.get_embedding(input=input)

    document_similarities = sorted([
      (self.vector_similarity(x=query_embedding, y=doc_embedding), doc_index) for doc_index, doc_embedding in
      contexts.items()
    ], reverse=True)

    return document_similarities

  def construct_prompt(self, prompt: str, context_embeddings: dict, data_frame: pd.DataFrame) -> str:
    """
    :param str prompt:
    :param dict context_embeddings:
    :param pd.DataFrame data_frame:
    :return str:
    """
    # Поиск наиболее похожих вложений документа на вложение вопроса
    most_relevant_document_sections = self.order_document_sections_by_query_similarity(prompt, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:

      document_section = self.data_frame.loc[section_index]
      chosen_sections_len += document_section.tokens + self.separator_len
      if chosen_sections_len > self.MAX_SECTION_LEN:
        break

      chosen_sections.append(self.SEPARATOR + document_section.content.replace("\n", " "))
      chosen_sections_indexes.append(str(section_index))

    logger.info(f"Selected {len(chosen_sections)} document sections:")
    logger.info("\n".join(chosen_sections_indexes))

    # prompt_header = f"Ответь на вопрос как можно правдивее, используя предоставленный контекст, и если ответ не содержится в приведенном ниже тексте, скажите \"UNKNOWN_TOPIC\""

    prompt_header = f"Ответь на вопрос как можно правдивее, используя предоставленный контекст, и если ответ не содержится в приведенном ниже тексте или в предыдущих ответах и вопросах, то попробуй ответить сама."

    header = f"""{prompt_header}\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + prompt + "\n A:"


  def ask(
          self,
          query: str,
          lastMessages: list[str],
          data_frame: pd.DataFrame,
          document_embeddings: dict[(str, str), np.array],
          show_prompt: bool = False
  ) -> str:
    """
    :param str query:
    :param pd.DataFrame data_frame:
    :param dict[(str, str), np.array] document_embeddings:
    :param bool show_prompt:
    :return str:
    """

    messages = [
      {"role": "system", "content": "Меня зовут Matreshka. Я создана в марте 2022 года. Я VPN сервис, для вашей безопасности от цензуры и утечки данных. Мои создатели очевидно программисты, я не могу сказать их имен, ведь защита персональных данных одна из главных моих задач. Мой пол женский, но я искусственный интеллект, так что формально у меня нет гендерной принадлежности. Меня назвали Matreshka, потому что, матрёшка — русская деревянная игрушка в виде расписной куклы, внутри которой находятся подобные ей куклы меньшего размера. Принцип VPN чем то напоминает эту игрушку, сервер внутри сервера, внутри сервера. Совершенно простая и незамысловатая на вид игрушка, скрывающая идеальную архитектуру внутри. Я говорю на любые темы, но не стану отвечать на вопросы связанные с незаконной деятельностью. Это твоя биография."}
    ]
    lastMessages.reverse()
    for i in lastMessages:
      messages.append({"role": "user", "content": i})

    prompt = self.construct_prompt(
      query,
      document_embeddings,
      data_frame
    )

    messages.append({"role": "system", "content": prompt})


    completion = openai.ChatCompletion.create(
      messages=messages,
      **self.COMPLETIONS_API_PARAMS
    )


    res = completion.choices[0].message.content

    return res

    # if res == "UNKNOWN_TOPIC":
    #   newCompletion = openai.ChatCompletion.create(
    #     messages=[
    #       {"role": "user", "content": query}
    #     ]
    #   )
    #   logger.info(newCompletion.choices[0].message.content)
    #   return newCompletion.choices[0].message.content
    #
    # else:
    #   logger.info(res)
    #   return res

