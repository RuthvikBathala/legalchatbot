import os
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

LEGAL_DOMAINS = [
    "criminal_law",
    "civil_law",
    "family_law",
    "employment_law",
    "accident_law",
    "sexual_offense",
    "juvenile_law",
    "property_law",
    "consumer_law",
    "contract_law",
    "cyber_law",
    "public_services"
]

def classify_domains(user_query: str) -> List[str]:
    system_prompt = f"""
You are an expert legal assistant. Your job is to identify all relevant legal domains that apply to the user's legal situation.

Available legal domains:
{', '.join(LEGAL_DOMAINS)}

Instructions:
- Analyze the user's query.
- Return a JSON list of domain identifiers (from above) that are relevant.
- If none match or it's unclear, return ["general"].
- Do NOT explain anything else. Output only the JSON list.

Example:
["criminal_law", "sexual_offense"]
"""
    try:
        response=openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.3,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user","content":user_query}
            ]
        )
        output=response['choices'][0]['message']['content'].strip()
        domains=eval(output) if output.startswith("[") else ["general"]
        return domains if isinstance(domains, list) else ["general"]
    except Exception as e:
        print("Error in domain classification:", e)
        return ["general"]

 