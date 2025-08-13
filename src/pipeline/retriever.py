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

def retrieve_relevant_laws(intake:Dict[str,Any],top_k:int=5)->Dict[str,List[Dict[str,str]]]:
    country=intake.get("country")
    if not country:
        raise ValueError("Intake missing 'country'. Cannot retrieve laws.")
    domain_specific=intake.get("domain_specific") or {}
    if not domain_specific:
        domains=intake.get("domains") or []
        if domains:
            gf=intake.get("facts",[]) or []
            gq=intake.get("legal_questions",[]) or []
            domain_specific={d:{"facts":list(gf),"legal_questions":list(gq)} for d in domains}
        else:
            raise ValueError("No domains found in intake for retrieval.")
    vectorstore=load_vectorstore(country)
    results={}
    for domain,data in domain_specific.items():
        query=build_domain_query(data)
        if not query:
            results[domain]=[]
            continue
        docs=vectorstore.similarity_search(query,k=top_k)
        results[domain]=[{
            "section":doc.metadata.get("section",""),
            "act":doc.metadata.get("act",""),
            "jurisdiction":doc.metadata.get("jurisdiction",""),
            "title":doc.metadata.get("title",""),
            "content":doc.page_content.strip()
        } for doc in docs]
    return results
