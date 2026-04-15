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