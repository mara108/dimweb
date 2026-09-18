from generate_live import make_clients, generate_diminutives, print_result
from dotenv import load_dotenv

load_dotenv()
gigachat_client, gpt_llm = make_clients()  # один раз при старте приложения

# на каждый запрос пользователя:
result = generate_diminutives(gigachat_client, gpt_llm, "рука")
print_result(result)