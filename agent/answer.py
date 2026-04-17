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
            print("🔍 Модель не найдена. Проверьте MODEL_NAME_ в .env.")
        else:
            print(f"⚠️ Неожиданная ошибка ({error_type}): {error_msg}")

    finally:
        print()

# Ответ ИИ без команды print (для API чата)

def get_agent_answer(agent, session_id: str, message: str, print_trace: bool = False, print_tool_details: bool = False) -> str:
    user_msg = f"{message} \n- session_id: {session_id}"
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": session_id}},
    )
    # return agent_response
    
    final_answer = ""

    try:
        # получаем все сообщения
        messages = agent_response.get("messages", [])

        # находим индекс последнего пользовательского сообщения
        human_msg_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
        last_human_msg_index = human_msg_indices[-1] if human_msg_indices else None

        if last_human_msg_index is not None:
            # Если нужно выводить ответы инструментов, собираем их в отдельный массив
            if print_tool_details:
                tool_results = {}
                for msg in messages[last_human_msg_index + 1:]:
                    if isinstance(msg, ToolMessage):
                        tool_results[msg.tool_call_id] = msg

            # проходимся по сообщениям начиная с последнего вопроса пользователя
            for msg in messages[last_human_msg_index + 1:]:
                # msg = messages[i]
                if isinstance(msg, AIMessage): 
                    if msg.content:
                        final_answer += (f"\n{msg.content}")
                    elif  msg.tool_calls and print_trace:
                        for tc in msg.tool_calls:
                            final_answer += f"\n  → вызов: {tc['name']}({tc['args']})"
                            if print_tool_details and tc['id'] in tool_results:
                                tool_result = tool_results[tc['id']]
                                final_answer += f"\n  ← результат {tool_result.name}: {tool_result.content}"
                

    except KeyboardInterrupt:
        final_answer += "\n[Остановлено пользователем]"

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        # Частые ошибки при работе с LLM/агентом
        if "AuthenticationError" in error_type or "401" in error_msg:
            final_answer += ("❌ Ошибка авторизации: проверьте API-ключ (OPENROUTER_API_KEY).")
        elif "RateLimitError" in error_type or "429" in error_msg:
            final_answer += ("⏳ Превышен лимит запросов. Попробуйте через несколько секунд.")
        elif "ConnectionError" in error_type or "ConnectTimeout" in error_type:
            final_answer += ("🌐 Ошибка сети: не удалось подключиться к API. Проверьте интернет.")
        elif "NotFound" in error_type or "404" in error_msg:
            final_answer += ("🔍 Модель не найдена. Проверьте MODEL_NAME_ в .env.")
        else:
            final_answer += (f"⚠️ Неожиданная ошибка ({error_type}): {error_msg}")

    return final_answer.strip()