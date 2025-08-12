import os
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_reasoner_prompt(domain:str, domain_data:Dict[str,Any],retrieved_docs:List[str])->str:
    facts="\n".join(domain_data.get("facts",[]))
    legal_questions="\n".join(domain_data.get("legal_questions",[]))
    entities="\n".join(domain_data.get("entities",[]))
    timeline="\n".join(domain_data.get("timeline",[]))
    location=domain_data.get("location","Not specified")
    injuries="\n".join(domain_data.get("injuries",[]))
    damages="\n".join(domain_data.get("damages",[]))

    laws_text="\n\n".join(retrieved_docs) if retrieved_docs else "No relevant laws found."

    return f"""
You are a senior legal associate specializing in {domain}.
The user's case details are as follows:

Facts:
{facts}

User's Legal Questions:
{legal_questions}

Entities Involved:
{entities}

Timeline of Events:
{timeline}

Location:
{location}

Injuries/Damages:
{injuries if injuries else damages}

Relevant Laws (retrieved from legal database):
{laws_text}

TASK:
1. Answer the user's legal questions directly using the facts and laws provided.
2. Provide a clear explanation of how each fact maps to the relevant law.
3. Suggest the next legal steps the user should take.
4. Highlight any documents the user will need to proceed.
5. Point out any risks, deadlines, or limitation periods.
6. If laws are missing or insufficient, clearly state that more information is needed.

Your response should be structured in this JSON format:
{{
  "answers_to_questions": [ ... ],
  "next_steps": [ ... ],
  "documents_needed": [ ... ],
  "risks": [ ... ],
  "limitation_periods": [ ... ],
  "disclaimer": "This is not a substitute for professional legal advice."
}}
    """


def reason_on_case(intake:Dict[str,Any],retrieved_laws:Dict[str,List[Any]])->Dict[str,Any]:
    all_results={}
    for domain,domain_data in intake.get("domain_specific",{}).items():
        docs_content=[doc.page_content for doc in retrieved_laws.get(domain,[])]
        prompt=format_reasoner_prompt(domain,domain_data,docs_content)
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a legal reasoning assistant and advisor. Your role is to analyze the user's case, apply the relevant laws, and provide clear, structured legal guidance"},
                {"role":"user","content":prompt}
            ],
            temperature=0.2
        )
        try:
            structured_output=response.choices[0].message.content.strip()
            all_results[domain]=structured_output
        except Exception as e:
            all_results[domain]=f"[Error processing domain {domain}]:{str(e)}"
    return all_results
