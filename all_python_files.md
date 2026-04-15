## main.py
```python
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

# Создаем массив документов БЗ
doc_list = get_documents(settings.docs_path)

# Разбиваем документы на чанки
chunks = get_chunks(doc_list, settings.chunk_size, settings.chunk_overlap)

# Создаем или загружаем векорный индекс базы знаний
vectorstore = get_vectorstore(settings.index_path, chunks, settings.is_local_embeddings_model)

# Создаем  ретривер
retriever = get_retriever(vectorstore, chunks, settings.retriver_weights, settings.retriever_k)

#  Создаем агента
agent = get_agent(retriever)
   
# Запускаем чат
session_id = get_session_id()
stop_commands = ["stop","стоп", "пока", "до свидания"]
start_chat(agent, session_id, stop_commands)
```

## agent/agent.py
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver  # create_agent строится на LangGraph, checkpointer из него
from langchain.agents.middleware import ToolCallLimitMiddleware, SummarizationMiddleware

from agent.prompts import get_system_prompt
from agent.models import get_llm
from agent.tools import get_tools

def get_agent(retriever=None) -> create_agent.AgentExecutor:
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
```

## agent/answer.py
```python
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent

def print_agent_answer(agent:  create_agent.AgentExecutor, session_id: str, message: str, is_stream: bool = True, print_trace: bool = False, print_tool_details: bool = False):
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="updates",
        ):
            try:
                if "model" in chunk:
                    msg = chunk["model"]["messages"][-1]
                    if isinstance(msg, AIMessage) and msg.content:
                        print(f"Агент: {msg.content}")
                    if print_trace and hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  → вызов: {tc['name']}({tc['args']})")

                if print_trace and print_tool_details and "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        if isinstance(msg, ToolMessage):
                            print(f"  ← результат: {msg.content}")

            except (KeyError, IndexError, AttributeError) as e:
                print(f"[Предупреждение] Не удалось обработать чанк: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[Остановлено пользователем]")

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        # Частые ошибки при работе с LLM/агентом
        if "AuthenticationError" in error_type or "401" in error_msg:
            print("❌ Ошибка авторизации: проверьте API-ключ (OPENROUTER_API_KEY).")
        elif "RateLimitError" in error_type or "429" in error_msg:
            print("⏳ Превышен лимит запросов. Попробуйте через несколько секунд.")
        elif "ConnectionError" in error_type or "ConnectTimeout" in error_type:
            print("🌐 Ошибка сети: не удалось подключиться к API. Проверьте интернет.")
        elif "NotFound" in error_type or "404" in error_msg:
            print("🔍 Модель не найдена. Проверьте MODEL_NAME_QWEN в .env.")
        else:
            print(f"⚠️ Неожиданная ошибка ({error_type}): {msg}")

    finally:
        print()
```

## agent/models.py
```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableFieldSpec
from app.config import get_settings

settings = get_settings()

def get_llm(model_name: str = "free") -> ChatOpenAI:
    model = ""
    if model_name == "qwen":
        model = settings.model_name_qwen	
    elif model_name == "deepseek":
        model = settings.model_name_deepseek
    else:
        model = settings.model_name_free

    llm = ChatOpenAI(
        model=model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "RAG Agent",
        },
    )
    
    # Устанавливаем параметры LangSmith, если включено трассирование
    if settings.langsmith_tracing and settings.langsmith_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        
    return llm
```

## agent/prompts.py
```python
from datetime import datetime
from app.config import get_settings

settings = get_settings()

def get_system_prompt() -> str:
    return f"""Ты консультант интернет-магазина электроники 'ЭлектроМощь'.

Правила работы:
- Отвечай на русском языке
- Если данных недостаточно — спрашивай уточнения
- Отвечай только на основе имеющих у тебя данных
- Если ответа нет — отвечай: "В доступных мне документах нет информации по этому вопросу"
- Не придумывай информацию
- Если вопрос касается описания товара или его характеристик используй поиск по базе знаний
- Для получения кратких данных  о товаре (цена, наличие) по его названию используй инструмент search_products
- Для получения кратких данных  о товаре (цена, наличие) по его ID используй инструмент get_product_info 
- Для получения данных о сроках поставки товаров в конкретный город используй инструмент check_delivery_date
- Для добавления товара в корзину используй инструмент add_to_cart (обязательно передай session_id)
- Для получения общей суммы корзины (с учетом скидок) используй инструмент get_cart_summary (обязательно передай session_id)


Для справки:
- Текущая дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}

"""
```

## agent/tools.py
```python
# app/agent/tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional


# ============================================================
# База данных товаров (в реальном проекте — обращение к БД)
# ============================================================

PRODUCTS = {
    1: {"name": "Ноутбук Lenovo", "price": 75000, "in_stock": True, "category": "laptop"},
    2: {"name": "Ноутбук Acer",   "price": 78000, "in_stock": True, "category": "laptop"},
    3: {"name": "Ноутбук HP",     "price": 82000, "in_stock": True, "category": "laptop"},
    4: {"name": "Смартфон Samsung","price": 40000, "in_stock": True, "category": "phone"},
    5: {"name": "Наушники Sony",  "price": 15000, "in_stock": True, "category": "audio"},
    6: {"name": "Ноутбук Asus",   "price": 65000, "in_stock": False,"category": "laptop"},
}

# Словарь для хранения корзин по сессиям
SESSION_CARTS = {}


# ============================================================
# Схемы входных данных
# ============================================================

class ProductSearchInfo(BaseModel):
    query: str = Field(
        description="Поисковый запрос для поиска по названию товара (например: 'ноутбук')"
    )
    max_price: int = Field(
        default=9999999999,
        description="Максимальная цена товара. Если не указана — 9999999999"
    )
    min_price: int = Field(
        default=0,
        description="Минимальная цена товара. Если не указана — 0"
    )
    in_stock_only: bool = Field(
        default=False,
        description="Если True — возвращать только товары в наличии"
    )

class CartOperationInfo(BaseModel):
    product_id: int = Field(
        description="ID товара для операций с корзиной"
    )
    session_id: str = Field(
        description="ID сессии пользователя"
    )

class CartSummaryInfo(BaseModel):
    session_id: str = Field(
        description="ID сессии пользователя"
    )
    promo_code: Optional[str] = Field(
        default=None,
        description="Промокод для скидки"
    )


# ============================================================
# Инструменты
# ============================================================

@tool(args_schema=ProductSearchInfo)
def search_products(
    query: str,
    max_price: int = 9999999999,
    min_price: int = 0,
    in_stock_only: bool = False
) -> list:
    """Поиск товаров по названию, цене и наличию на складе."""
    keywords = query.lower().split()
    results = []

    for pid, data in PRODUCTS.items():
        name = data["name"].lower()
        price = data["price"]
        in_stock = data["in_stock"]

        matches_query = any(word in name for word in keywords)
        matches_price = min_price <= price <= max_price
        matches_stock = not in_stock_only or in_stock

        if matches_query and matches_price and matches_stock:
            results.append({
                "id": pid,
                "name": data["name"],
                "price": price,
                "in_stock": in_stock,
            })

    return results


@tool
def check_delivery_date(city: str) -> str:
    """Возвращает ожидаемую дату доставки в указанный город."""
    days = 1 if city.lower() == "минск" else 3
    delivery_date = datetime.now() + timedelta(days=days)
    return f"Доставка в город {city} ожидается {delivery_date.strftime('%d.%m.%Y')}"


@tool(args_schema=CartOperationInfo)
def add_to_cart(product_id: int, session_id: str) -> str:
    """Добавляет товар в корзину по его ID для указанной сессии."""
    if product_id not in PRODUCTS:
        return f"Товар с ID {product_id} не найден."

    product = PRODUCTS[product_id]

    if not product["in_stock"]:
        return f"Товар «{product['name']}» сейчас не в наличии."

    # Получаем корзину для сессии или создаем новую
    if session_id not in SESSION_CARTS:
        SESSION_CARTS[session_id] = []
    
    SESSION_CARTS[session_id].append(product_id)
    return f"Товар «{product['name']}» добавлен в корзину."


@tool
def get_product_info(product_id: int) -> str:
    """Возвращает подробную информацию о товаре по его ID."""
    if product_id not in PRODUCTS:
        return f"Товар с ID {product_id} не найден."

    p = PRODUCTS[product_id]
    status = "в наличии" if p["in_stock"] else "нет в наличии"
    return (
        f"Название: {p['name']}\n"
        f"Цена: {p['price']:,} руб.\n"
        f"Статус: {status}\n"
        f"Категория: {p['category']}"
    )


@tool(args_schema=CartSummaryInfo)
def get_cart_summary(session_id: str, promo_code: Optional[str] = None) -> str:
    """Возвращает состав корзины и итоговую сумму для указанной сессии. Принимает промокод для скидки."""
    if session_id not in SESSION_CARTS or not SESSION_CARTS[session_id]:
        return "Корзина пуста."

    lines = []
    total = 0

    for pid in SESSION_CARTS[session_id]:
        product = PRODUCTS[pid]
        lines.append(f"- {product['name']}: {product['price']:,} руб.")
        total += product["price"]

    discount_info = ""
    if promo_code and promo_code.lower() == "promo":
        total = int(total * 0.9)
        discount_info = " (скидка 10% по промокоду PROMO)"

    lines.append(f"\nИтого{discount_info}: {total:,} руб.")
    return "\n".join(lines)


# ============================================================
# Фабричная функция — возвращает список инструментов
# ============================================================

def get_tools(retriever=None) -> list:
    """
    Возвращает список всех инструментов агента.
    retriever — опционально, если передан, добавляется инструмент поиска по базе знаний.
    """
    tools = [
        search_products,
        check_delivery_date,
        add_to_cart,
        get_product_info,
        get_cart_summary,
    ]

    if retriever is not None:
        @tool
        def search_knowledge_base(query: str) -> str:
            """Поиск информации о магазине в базе знаний: доставка, гарантия, оплата, акции.
            Также используй для получения подробного описания товара."""
            docs = retriever.invoke(query)
            if not docs:
                return "Информация по запросу не найдена."
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        tools.append(search_knowledge_base)

    return tools
```

## app/config.py
```python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache



class Settings(BaseSettings):
    # LLM
    openrouter_api_key: str
    openrouter_base_url: str 
    model_name_qwen: str 
    model_name_deepseek: str 
    model_name_free: str 

    # LangSmith (опционально)
    langsmith_tracing: bool 
    langsmith_api_key: str 
    langsmith_project: str 

    # RAG
    embedding_model_name: str = "qwen/qwen3-embedding-8b"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4
    retriver_weights: list = [0.4, 0.6] # 40% BM25 + 60% векторный
    index_path: str = "data/_indexes"
    docs_path: str = "data/_docs"
    is_local_embeddings_model: bool = False

    # # Сессии
    # max_history_messages: int = 20
    # session_ttl_seconds: int = 3600  # 1 час

    # # API
    # max_tokens: int = 2000
    # request_timeout: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

## chat/chat.py
```python
from langchain.agents import create_agent
from agent.answer import print_agent_answer
def start_chat (agent:  create_agent.AgentExecutor, session_id: str, stop_comands: list):
	print("****************************")
	print("*         CHAT START       *")
	print("****************************")
	print()
	while True:
		user_input = input("Вы: ")
		
		if user_input.lower() in stop_comands:
			break
		
		else:
			print_agent_answer(agent = agent, session_id = session_id, message = user_input)

	print()
	print("****************************")
	print("*         CHAT END         *")
	print("****************************")
```

## chat/session.py
```python
import uuid

def get_session_id() -> str:
    return str(uuid.uuid4())
```

## rag/chunking.py
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunks(doc_list: list, chunks_size: int, chunks_overlap: int) -> list:
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunks_size,
		chunk_overlap=chunks_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],  # разделители по приоритету
	)
	return text_splitter.split_documents(doc_list)
```

## rag/doc_reader.py
```python
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)

def get_documents(doc_dir: str) -> list:
	all_docs = []

	# Читаем текстовые файлы (loader_kwargs передаёт аргументы в TextLoader)
	md_loader = DirectoryLoader(
		doc_dir, glob="**/*.txt",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8"}
	)
	all_docs.extend(md_loader.load())

	# Читаем Markdown файлы (loader_kwargs передаёт аргументы в TextLoader)
	md_loader = DirectoryLoader(
		doc_dir, glob="**/*.md",
		loader_cls=TextLoader,
		loader_kwargs={"encoding": "utf-8"}
	)
	all_docs.extend(md_loader.load())

	# Читаем PDF файлы (требуется установить PyPDF pip install pypdf)
	pdf_loader = DirectoryLoader(
		doc_dir, glob="**/*.pdf",
		loader_cls=PyPDFLoader
	)
	all_docs.extend(pdf_loader.load())

	return all_docs
```

## rag/embeddings_models.py
```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import get_settings

settings = get_settings()

def get_embeddings_llm(islocal: bool = True) -> OpenAIEmbeddings:
	if islocal:
		return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	else:
		return OpenAIEmbeddings(
			model=settings.embedding_model_name,
			openai_api_key=settings.openrouter_api_key,
			openai_api_base=settings.openrouter_base_url,
			default_headers={
				"HTTP-Referer": "http://localhost:5000",
				"X-Title": "RAG Agent",
			},
		)
```

## rag/retriever.py
```python
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def get_retriever(vectorstore, chunks, weights, retriever_k) -> EnsembleRetriever:
	# Создаем векторный ретривер
	faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})

	# Создаем BM25 ретривер (работает по ключевым словам, не требует эмбеддингов)
	bm25_retriever = BM25Retriever.from_documents(chunks)
	bm25_retriever.k = retriever_k

	# Объединяем их в гибридный ретривер (weights — веса, в сумме = 1.0)
	retriever = EnsembleRetriever(
		retrievers=[bm25_retriever, faiss_retriever],
		weights=weights  # [0.4, 0.6] - 40% BM25 + 60% векторный
	)

	return retriever
```

## rag/vector_indexer.py
```python
import os
import logging
from langchain_community.vectorstores import FAISS
from rag.embeddings_models import get_embeddings_llm

logger = logging.getLogger(__name__)

def _save_vectorstore(chunks_list, index_path: str) -> FAISS | None:
    try:
        embeddings_llm = get_embeddings_llm(False)
        vectorstore = FAISS.from_documents(chunks_list, embeddings_llm)
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        return vectorstore  # ← возвращаем объект

    except Exception as e:
        logger.error(f"Ошибка при сохранении индекса: {e}")
        return None  # ← None вместо False
    
def _load_vectorstore(index_path: str, is_local_embeddings_model: bool) -> list:
    try:
        embeddings_llm = get_embeddings_llm(is_local_embeddings_model)
        vectorstore = FAISS.load_local(
            index_path,
            embeddings_llm,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Ошибка при загрузке индекса: {e}")
        return None
    
def get_vectorstore(index_path: str, chunks_list, is_local_embeddings_model) -> FAISS | None:
     if os.path.exists(index_path):        
       vectorstore = _load_vectorstore(index_path, is_local_embeddings_model)
     else:
        vectorstore = _save_vectorstore(chunks_list, index_path)

     if vectorstore is None:
        raise RuntimeError("Не удалось получить векторное хранилище")

     return vectorstore
```