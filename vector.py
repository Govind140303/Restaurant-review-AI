from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load your review dataset
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize embeddings using Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define persistent Chroma database location
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Prepare documents for storage if DB doesn't exist
if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=f"{row['Title']} {row['Review']}",
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create or load the Chroma vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents only the first time
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever (for top 5 most relevant reviews)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Export DataFrame for other scripts
def get_dataframe():
    return df
