import os
import json
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
def call_llm_to_format_json(prompt:str,domain_outputs:Dict[str,str])->str:
    combined_input=""
    for domain, output in domain_outputs.items():
        combined_input+=f"\n\n--- {domain} ---\n{output.strip()}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":combined_input}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def format_and_merge_intake(domain_outputs:Dict[str,str])->Dict:
    prompt_path=Path("prompt_temp")/"formatter.txt"
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompt=f.read()
    formatted_str=call_llm_to_format_json(prompt,domain_outputs)
    try:
        return json.loads(formatted_str)
    except json.JSONDecodeError:
        print("ERROR! LLM returned invalid JSON.")
        print(formatted_str)
        return {}
