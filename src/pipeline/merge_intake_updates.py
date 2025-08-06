from typing import Dict,Any

def merge_user_responses(intake:Dict[str,Any],answers:Dict[str,Any])->Dict[str,Any]:
    if "country" in answers and answers["country"]:
        intake["country"]=answers["country"]

    if "facts" in answers and answers["facts"]:
        existing_facts=intake.get("facts",[])
        intake["facts"]=list(set(existing_facts+answers["facts"]))

    if "legal_questions" in answers and answers["legal_questions"]:
        existing_qs=intake.get("legal_questions",[])
        intake["legal_questions"]=list(set(existing_qs+answers["legal_questions"]))

    domain_specific=intake.get("domain_specific",{})
    for domain,data in domain_specific.items():
        if domain in answers and isinstance(answers[domain],dict):
            if "facts" in answers[domain] and answers[domain]["facts"]:
                domain_facts=data.get("facts",[])
                data["facts"]=list(set(domain_facts+answers[domain]["facts"]))
            if "legal_questions" in answers[domain] and answers[domain]["legal_questions"]:
                domain_qs=data.get("legal_questions",[])
                data["legal_questions"]=list(set(domain_qs+answers[domain]["legal_questions"]))
            domain_specific[domain]=data
    intake["domain_specific"]=domain_specific
    return intake
