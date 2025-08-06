import json
from typing import Dict, List, Tuple, Any


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
        return []
    return [q for q in raw_questions if isinstance(q,str) and q.strip()]

def summarize_missing_info(intake:Dict[str,Any])->Dict[str,Any]:
    complete,missing_keys=is_intake_complete(intake)
    follow_up=extract_follow_up_questions(intake)
    return {
        "is_complete":complete,
        "missing_keys":missing_keys,
        "follow_up_questions":follow_up
    }

