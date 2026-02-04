from fastapi import FastAPI
import chromadb
import ollama
import os

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

# Configure Ollama to connect to host machine
ollama_host = os.getenv("OLLAMA_HOST", "http://host.orb.internal:11434")
client = ollama.Client(host=ollama_host)

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    answer = client.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    return {"answer": answer["response"]}
