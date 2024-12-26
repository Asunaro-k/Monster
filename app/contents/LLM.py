from langchain_ollama import ChatOllama

def init_ollama():
    llm = ChatOllama(
        base_url="http://host.docker.internal:11434/",
        #llama3.1:8b
        #llama3.2:latest
        #7shi/tanuki-dpo-v1.0:latest
        model="gemma2:9b",
        temperature=0.7
    )
    return llm