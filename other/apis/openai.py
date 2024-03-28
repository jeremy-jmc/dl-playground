from openai import OpenAI
import tiktoken
from typing import Callable
import os
import sys

dummy_check_callback = lambda val: True

SEED = 42

def make_question(prompt: str,
                  check_callback: Callable = dummy_check_callback,
                  model_name: str = 'gpt-3.5-turbo', 
                  **kwargs) -> str:
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
    )
    
    encoding = tiktoken.get_encoding('cl100k_base')
    print('# prompt tokens:', len(encoding.encode(prompt)))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        # try: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
        #   gpt-3.5-turbo-16k
        model=model_name,
        temperature=0.0,
        seed=SEED,
        **kwargs
    )
    
    # https://platform.openai.com/docs/guides/text-generation/managing-tokens
    print(f'{chat_completion.usage}')
    if model_name == 'gpt-3.5-turbo':
        price = chat_completion.usage.prompt_tokens * 3/1e6 + chat_completion.usage.completion_tokens * 6/1e6
        print('	total price: USD {:.10f}'.format(price))
    else:
        print('	check pricing in the OpenAI API documentation!')

    # print(chat_completion)
    # print(dir(chat_completion))
    # print('chat_completion to dict')
    # print(chat_completion.__dict__['choices'][0].__dict__['message'].content)

    if hasattr(chat_completion, 'choices') and chat_completion.choices:
        message = chat_completion.choices[0].message
        if hasattr(message, 'content'):
            # print(message.content)
            if check_callback(message.content):
                print('# response tokens:', len(encoding.encode(message.content)))
                return message.content
            else:
                print('Warning: check_callback failed.')
                return None
        else:
            print('Warning: "content" attribute not found in message.')
            return None
    else:
        print('Warning: "choices" attribute not found in chat_completion.')
        return None
