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

