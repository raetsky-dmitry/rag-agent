from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver  # create_agent строится на LangGraph, checkpointer из него
from langchain.agents.middleware import ToolCallLimitMiddleware, SummarizationMiddleware

from agent.prompts import get_system_prompt
from agent.models import get_llm
from agent.tools import get_tools

def get_agent(session_id: str, retriever=None) -> create_agent.AgentExecutor:
    tools = get_tools(retriever)
    return create_agent(
		model=get_llm("qwen"), 
		tools=tools, 
		system_prompt=get_system_prompt(session_id),
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