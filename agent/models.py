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