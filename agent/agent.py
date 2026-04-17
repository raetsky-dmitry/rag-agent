from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver  # create_agent строится на LangGraph, checkpointer из него
from langchain.agents.middleware import ToolCallLimitMiddleware, SummarizationMiddleware

from agent.prompts import get_system_prompt
from agent.models import get_llm
from agent.tools import get_tools

from app.config import get_settings

from rag.doc_reader import get_documents
from rag.chunking import get_chunks
from rag.vector_indexer import get_vectorstore
from rag.retriever import get_retriever

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
print(" -- создали/загрузили векторный индекс базы знаний")

# Создаем  ретривер
retriever = get_retriever(vectorstore, chunks, settings.retriver_weights, settings.retriever_k)
print(" -- создали ретривер")

def get_agent() -> create_agent.AgentExecutor:
    tools = get_tools(retriever)
    return create_agent(
		model=get_llm("qwen"), 
		tools=tools, 
		system_prompt=get_system_prompt(),
		checkpointer=MemorySaver(),
		middleware=[
			# Глобальный лимит: не более 10 вызовов за всю беседу, не более 5 за один запрос
			ToolCallLimitMiddleware(
				thread_limit=10, # всего за беседу
				run_limit=5, # всего за один запрос
			),
			# Суммаризация истории при привышении лимита
			SummarizationMiddleware(
				model=get_llm("qwen"),
				trigger=("tokens", 4000),
				keep=("messages", 20),
			),
		]
	)