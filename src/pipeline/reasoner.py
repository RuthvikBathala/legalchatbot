import os
from typing import Dict,Any,List
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_reasoner_prompt(domain:str,domain_data:Dict[str,Any],retrieved_docs:List[Any],country:str)->str:
    facts="\n".join(domain_data.get("facts",[]))
    legal_questions="\n".join(domain_data.get("legal_questions",[]))
    entities="\n".join(domain_data.get("entities",[]))
    timeline="\n".join(domain_data.get("timeline",[]))
    location=domain_data.get("location","Not specified")
    injuries="\n".join(domain_data.get("injuries",[]))
    damages="\n".join(domain_data.get("damages",[]))
    law_citations=[]
    for doc in retrieved_docs:
        if hasattr(doc,"page_content"):
            snippet=(doc.page_content or "").strip()
            meta=getattr(doc,"metadata",{}) or {}
        elif isinstance(doc,dict):
            snippet=(doc.get("page_content") or doc.get("content") or doc.get("text") or "").strip()
            meta=(doc.get("metadata") or doc) or {}
        else:
            continue
        section=(meta.get("section") or "").strip()
        act=(meta.get("act") or "").strip()
        jurisdiction=(meta.get("jurisdiction") or country.upper()).strip()
        title=(meta.get("title") or "").strip()
        if section and act:
            law_citations.append(f"As per Section {section} of the {act} ({jurisdiction}) — {snippet}")
        elif title:
            law_citations.append(f"{title} ({jurisdiction}) — {snippet}")
        else:
            law_citations.append(f"({jurisdiction}) {snippet}" if snippet else "")
    law_citations=[c for c in law_citations if c]
    laws_text="\n".join(law_citations) if law_citations else "No relevant laws found."
    return f"""
You are a senior legal associate specializing in {domain}.
Facts of the case:
{facts}
Legal questions:
{legal_questions}
Entities:
{entities}
Timeline:
{timeline}
Location:
{location}
Injuries/Damages:
{injuries if injuries else damages}
Relevant laws and citations:
{laws_text}
Write a detailed, human-readable legal opinion that includes clearly labeled sections:
1. Summary of the case
2. Relevant legal provisions with proper citations
3. Analysis applying facts to laws
4. Documents the client must gather
5. Recommended next steps
6. Risks and limitation periods
7. Disclaimer
Use formal but clear legal language. Do not output JSON. Use headings and bullet points where helpful.
"""

def reason_on_case(intake:Dict[str,Any],retrieved_laws:Dict[str,List[Any]])->Dict[str,str]:
    all_results={}
    country=intake.get("country","")
    for domain,domain_data in intake.get("domain_specific",{}).items():
        prompt=format_reasoner_prompt(domain,domain_data,retrieved_laws.get(domain,[]),country)
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a legal reasoning assistant and advisor. Provide a detailed, clear, formal opinion based on facts and laws."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2
        )
        try:
            advice=response.choices[0].message.content.strip()
            all_results[domain]=advice
        except Exception as e:
            all_results[domain]=f"[Error processing domain {domain}]:{str(e)}"
    return all_results
