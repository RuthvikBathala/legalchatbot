import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_text_files_from_folder(folder_path):
    documents=[]
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename),"r",encoding="utf-8") as f:
                text=f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents

def ingest_country_laws(country):    
    folder_path=os.path.join("data", country)
    save_path=os.path.join("embeddings", country)
    documents=load_text_files_from_folder(folder_path)

    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks=splitter.split_documents(documents)

    embedder=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(chunks, embedder)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"[{country.upper()}] Saved FAISS index to: {save_path}")

if __name__ == "__main__":
    countries = ["india", "usa", "uk", "canada", "australia", "eu"]
    for country in countries:
        ingest_country_laws(country)
