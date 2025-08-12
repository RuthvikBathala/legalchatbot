import os
from typing import Dict,Any,List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_NAME="all-MiniLM-L6-v2"
BASE_EMBED_PATH=os.getenv("EMBEDDING_PATH","embeddings")

def load_vectorstore(country:str)->FAISS:
    embedder=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    index_path=os.path.join(BASE_EMBED_PATH, country.lower())
    return FAISS.load_local(index_path,embedder,allow_dangerous_deserialization=True)

def build_domain_query(domain_data:Dict[str,Any])->str:
    facts=domain_data.get("facts",[])
    return " ".join(facts).strip()

def retrieve_relevant_laws(intake:Dict[str,Any],top_k:int=5)->Dict[str,List[Document]]:
    country=intake.get("country")
    if not country:
        raise ValueError("Intake missing 'country'. Cannot retrieve laws.")
    domain_specific=intake.get("domain_specific",{})
    if not domain_specific:
        raise ValueError("No domains found in intake for retrieval.")

    vectorstore=load_vectorstore(country)
    results={}

    for domain, data in domain_specific.items():
        query=build_domain_query(data)
        if not query:
            print(f"Skipping domain '{domain}'. No facts found to query.")
            results[domain]=[]
            continue
        retrieved_docs=vectorstore.similarity_search(query,k=top_k)
        results[domain]=retrieved_docs
    return results
