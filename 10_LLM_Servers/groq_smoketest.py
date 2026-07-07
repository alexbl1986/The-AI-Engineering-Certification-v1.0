import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads 10_LLM_Servers/.env

llm = ChatOpenAI(
    model=os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
    openai_api_key=os.environ["GROQ_API_KEY"],
    openai_api_base="https://api.groq.com/openai/v1",
)

resp = llm.invoke("say hi in 3 words")
print(resp.content)