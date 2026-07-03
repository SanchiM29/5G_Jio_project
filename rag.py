import pandas as pd
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

#Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# global cache
df_cache = None
index_cache = None

# LOAD + EMBED 
def load_data():
    global df_cache, index_cache

    # If already loaded → reuse
    if df_cache is not None and index_cache is not None:
        return df_cache, index_cache

    df = pd.read_csv("anomaly_events.csv")

    def create_text(row):
        return (
            f"Anomaly from {row['start_time']} to {row['end_time']}. "
            f"Duration {row['duration_min']} minutes. "
            f"Throughput dropped to {row['min_throughput']} Mbps vs expected {row['expected_lower']} Mbps. "
            f"Drop {row['percent_drop']}%. Severity {row['severity']}."
        )

    df["text"] = df.apply(create_text, axis=1)

    # Faster encoding
    embeddings = embed_model.encode(
        df["text"].tolist(),
        show_progress_bar=False
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Cache it
    df_cache = df
    index_cache = index

    return df, index
# RETRIEVE
def retrieve(query, df, index, k=2):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)

    return df.iloc[indices[0]]

# CALL OLLAMA (LIGHT MODEL)
def call_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",  
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        return response.json().get("response", "No response from model")

    except Exception as e:
        return f"Ollama Error: {e}"

# query
def explain(query):
    df, index = load_data()
    context = retrieve(query, df, index)

    context_text = "\n".join(

        context["text"].tolist()

    )

    prompt = f"""
You are a telecom network analyst.

Your task is to answer the user's selected analysis question using ONLY the retrieved anomaly context provided below.

Retrieved Context:
{context_text}

Instructions:
- Answer ONLY the user question.
- Use the retrieved context as supporting evidence.
- Do NOT invent information, causes, or metrics.
- Keep the explanation concise, realistic, and data-driven.
- Ensure the response is understandable for both technical and non-technical users.
- Maintain a professional but simple tone.
- Mention exact throughput values, percentage drops, severity, and timings whenever available.
- Any decimal value must be rounded to a maximum of 2 decimal places.
- If the context does not contain enough evidence, clearly mention that.

Answering Guidelines:
- If the question asks "What happened?" → summarize the anomaly event clearly.
- If the question asks about the cause → infer the most likely operational/network-related reason from the context.
- If the question asks about impact → explain expected effect on users or network performance in simple language.
- If the question asks about actions → provide practical mitigation or troubleshooting steps.
- If the question asks about severity → explain severity using the available metrics and anomaly behavior.

Generate a direct answer to the selected question only.

User Question:
{query}

"""
    # SEND PROMPT TO LLM
    return call_ollama(prompt)
