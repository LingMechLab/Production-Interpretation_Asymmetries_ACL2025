from openai import OpenAI
import numpy as np

client = OpenAI(api_key="YOUR_API_KEY")


def openai_api(model, prompt, use_prob=False):
    messages = [{"role": "user", "content": prompt}]
    if use_prob:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
            max_completion_tokens=1,
            temperature=0,
        )
        content = completion.choices[0].message.content
        prob = completion.choices[0].logprobs.content[0].logprob
        return [content, np.exp(prob)]
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return completion.choices[0].message.content
