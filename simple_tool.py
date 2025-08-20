from ollama import chat, ChatResponse 

def add_two_numbers(a: int, b: int) -> int:
    """
    Суммирует два числа и возвращает результат
    ARGS:
    a - int
    b - int
    """
    return a + b


TOOLS = {
    "add_two_numbers": add_two_numbers
}


messages = [{
    'role': 'user',
    'content': 'банан это 2. Опиши, что такое банан. Можно ли к банану применить функцию из tools? Если да, вызови tools для двух бананов.'
}]


response: ChatResponse = chat(
    model='qwen3:4b-q4_K_M',
    messages=messages,
    tools=[add_two_numbers],
    stream=True
)


for chunk in response:
    if chunk.message.content:
        print(chunk.message.content, end='', flush=True)

    if chunk.message.tool_calls:
        for call in chunk.message.tool_calls:
            func_name = call.function.name
            args = call.function.arguments
            print(f"\n🔧 Модель хочет вызвать {func_name}({args})")

            # Вызов функции напрямую
            if func_name:
                print(TOOLS.get(func_name)(**args))
