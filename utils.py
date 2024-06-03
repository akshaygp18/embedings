import openai
import os
from dotenv import load_dotenv
load_dotenv()

def get_embedding(text, model="text-embedding-3-small"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']
