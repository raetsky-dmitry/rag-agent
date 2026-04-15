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