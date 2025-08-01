import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt(domain:str)->str:
    prompt_path=Path("prompt_temp")/f"{domain}.txt"
    with open(prompt_path,"r",encoding="utf-8") as f:
        return f.read()

def call_llm_with_prompt(prompt:str, user_input:str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system", "content":prompt},
            {"role":"user", "content":user_input}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def run_domain_intake(user_input:str, domains:List[str])->Dict[str,str]:
    outputs={}
    for domain in domains:
        try:
            prompt=load_prompt(domain)
            raw_output=call_llm_with_prompt(prompt,user_input)
            outputs[domain]=raw_output
        except Exception as e:
            print(f"ERROR! Failed for {domain}: {e}")
    return outputs
