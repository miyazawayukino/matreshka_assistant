from .assistant import MatreshkaAssistant


version: '2'
services:
    openai_server:
        environment:
            - HOST="0.0.0.0"
            - PORT=8082
            - COMPLETIONS_MODEL="text-davinci-003"
            - EMBEDDING_MODEL="text-embedding-ada-002"
            - MAX_SECTION_LEN=500
            - SEPARATOR="\n"
            - ENCODING="cl100k_base"
            - TEMPERATURE=0.0
            - MAX_TOKENS=400
            - PROMPT_HEADER="Ответь на вопрос как можно правдивее, используя предоставленный контекст, и если ответ не содержится в приведенном ниже тексте, скажите \"Я не знаю ответ на этот вопрос.\""
        image: gipatlab/matreshkavpn_assistant/matreshkavpn_assistant
        restart: always
        container_name: 'openai_server'
        ports:
            - 8081:8081
        expose:
            - 8081