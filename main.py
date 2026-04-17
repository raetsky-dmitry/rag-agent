from agent.agent import get_agent
from chat.session import get_session_id
from chat.chat import start_chat

#  Создаем агента
agent = get_agent()
print(" -- создали агента")
 
# Запускаем чат
if __name__ == "__main__":
    session_id = get_session_id()
    stop_commands = ["stop","стоп", "пока", "до свидания"]
    start_chat(agent, session_id, stop_commands)
    