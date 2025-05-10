from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

from .storage import get_messages_for_event, summaries

llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")
prompt = PromptTemplate(
    input_variables=["chat"],
    template="""
You are an assistant summarizing a chat event. Write a 2–3 paragraph summary capturing main topics, names, and insights.

Chat:
{chat}

Summary:
"""
)
chain = LLMChain(llm=llm, prompt=prompt)

def summarize_event_chat(event_id: str):
    messages = get_messages_for_event(event_id)
    combined = "\n".join(messages)
    summary = chain.run(chat=combined)
    summaries[event_id] = summary
    print(f"[Summary Updated] eventId={event_id}")
