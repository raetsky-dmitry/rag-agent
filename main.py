from app.config import get_settings

from agent.agent import get_agent

from rag.embeddings_models import get_embeddings_llm
from rag.doc_reader import get_documents
from rag.chunking import get_chunks
from rag.vector_indexer import get_vectorstore
from rag.retriever import get_retriever

from chat.session import get_session_id
from chat.chat import start_chat

# Получаем настройки
settings = get_settings()
print(" -- загрузили настройки")

# Создаем массив документов БЗ
doc_list = get_documents(settings.docs_path)
print(" -- создали массив документов")

# Разбиваем документы на чанки
chunks = get_chunks(doc_list, settings.chunk_size, settings.chunk_overlap)
print(" -- разбили документы на чанки")

# Создаем или загружаем векорный индекс базы знаний
vectorstore = get_vectorstore(settings.index_path, chunks, settings.is_local_embeddings_model)
print(" -- создал/занрузили векторный индекс базы знаний")

# Создаем  ретривер
retriever = get_retriever(vectorstore, chunks, settings.retriver_weights, settings.retriever_k)
print(" -- создали ретривер")

# Получаем ID сессии
session_id = get_session_id()
print(" -- получили ID сессии")

#  Создаем агента
agent = get_agent(session_id, retriever)
print(" -- создали агента")
 
# Запускаем чат
# stop_commands = ["stop","стоп", "пока", "до свидания"]
# start_chat(agent, session_id, stop_commands)

if __name__ == "__main__":
    session_id = get_session_id()
    stop_commands = ["stop","стоп", "пока", "до свидания"]
    start_chat(agent, session_id, stop_commands)