from langchain.agents import create_agent
from agent.answer import print_agent_answer, get_agent_answer
def start_chat (agent:  create_agent.AgentExecutor, session_id: str, stop_comands: list):
	print("****************************")
	print("*         CHAT START       *")
	print("****************************")
	# print(f"session_id: {session_id}")
	print()
	while True:
		user_input = input("Вы: ")
		
		if user_input.lower() in stop_comands:
			break
		
		else:
			agent_answer = get_agent_answer(agent = agent, session_id = session_id, message = user_input, print_trace = True, print_tool_details = True)
			print (f"Агент: {agent_answer}")

	print()
	print("****************************")
	print("*         CHAT END         *")
	print("****************************")