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
            print(f"⚠️ Неожиданная ошибка ({error_type}): {error_msg}")

    finally:
        print()

# Ответ ИИ без команды print (для API чата)
from langchain_core.messages import AIMessage

def get_agent_answer(agent, session_id: str, message: str) -> str:
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": session_id}},
        stream_mode="updates",
    ):
        if "model" in chunk:
            msg = chunk["model"]["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content:
                final_answer += msg.content

    return final_answer.strip()