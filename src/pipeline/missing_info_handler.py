import os
import json
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _llm_followups(intake:Dict[str,Any])->List[str]:
    try:
        prompt="You are a lawyer conducting an intake. Given the JSON below, return ONLY a JSON array of concise follow-up questions that would help you give initial legal guidance. Do not include explanations."
        r=_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content":prompt},
                {"role":"user","content":json.dumps(intake,ensure_ascii=False)}
            ]
        )
        txt=r.choices[0].message.content.strip()
        if txt.startswith("```"):
            if txt.lower().startswith("```json"):
                txt=txt[7:].lstrip()
            else:
                txt=txt[3:].lstrip()
            if txt.endswith("```"): txt=txt[:-3].rstrip()
        data=json.loads(txt)
        if isinstance(data,dict) and "questions" in data: data=data["questions"]
        if isinstance(data,list):
            return [q for q in data if isinstance(q,str) and q.strip()]
    except Exception:
        return []
    return []

def is_intake_complete(intake:Dict[str,Any])->Tuple[bool,List[str]]:
    REQUIRED_KEYS=["country","facts","legal_questions"]
    missing=[]
    for key in REQUIRED_KEYS:
        if not intake.get(key):
            missing.append(key)
    return (len(missing)==0), missing

def extract_follow_up_questions(intake:Dict[str,Any])->List[str]:
    raw_questions=intake.get("missing_info",[])
    if not isinstance(raw_questions,list):
        raw_questions=[]
    questions=[q for q in raw_questions if isinstance(q,str) and q.strip()]
    extra=_llm_followups(intake)
    if extra:
        questions.extend(extra)
    seen=set();out=[]
    for q in questions:
        if q not in seen:
            out.append(q);seen.add(q)
    return out

def summarize_missing_info(intake: dict)->dict:
    CRITICAL_KEYS=["country", "facts","legal_questions","domains"]
    OPTIONAL_KEYS=["entities","timeline","location", "injuries", "damages"]

    missing_critical=[]
    missing_optional=[]

    for key in CRITICAL_KEYS:
        if not intake.get(key):
            missing_critical.append(key)

    for key in OPTIONAL_KEYS:
        if not intake.get(key):
            missing_optional.append(key)

    complete=len(missing_critical)==0
    follow_up=intake.get("missing_info",[]) or []

    return {
        "is_complete": complete,
        "missing_keys": missing_critical+missing_optional,
        "missing_critical": missing_critical,
        "missing_optional": missing_optional,
        "follow_up_questions": follow_up
    }
