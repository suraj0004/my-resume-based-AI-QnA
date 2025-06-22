import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
