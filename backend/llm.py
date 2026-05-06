import os
from openai import OpenAI

client = None

def _get_client():
    global client
    if client is None:
        # GLM_API_URL is the full endpoint (e.g. https://aihubmix.com/v1/chat/completions)
        # OpenAI SDK base_url must be the root without the path suffix
        base_url = os.environ["LLM_API_URL"]
        client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=base_url,
        )
    return client
