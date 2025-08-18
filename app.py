import json
import re
import streamlit as st
from src.pipeline import domain_classifier,intake_parser,intake_formatter,retriever,reasoner
from src.pipeline.missing_info_handler import summarize_missing_info
from src.pipeline.merge_intake_updates import merge_user_responses

st.set_page_config(page_title="Legal Chatbot",layout="wide")

COUNTRIES={"USA (United States)":"usa","United Kingdom":"uk","Canada":"canada","Australia":"australia","India":"india","European Union":"eu"}

if "chat" not in st.session_state: st.session_state.chat=[]
if "phase" not in st.session_state: st.session_state.phase="greeting"
if "intake" not in st.session_state: st.session_state.intake=None
if "predicted_domains" not in st.session_state: st.session_state.predicted_domains=[]
if "country_code" not in st.session_state: st.session_state.country_code=list(COUNTRIES.values())[0]
if "answered_followups" not in st.session_state: st.session_state.answered_followups=set()

with st.sidebar:
    st.header("Settings")
    country_label=st.selectbox("Select the country whose laws apply",list(COUNTRIES.keys()),index=list(COUNTRIES.values()).index(st.session_state.country_code) if st.session_state.country_code in COUNTRIES.values() else 0)
    st.session_state.country_code=COUNTRIES[country_label]
    st.caption("Your selected country affects which embeddings index is used for retrieval.")

st.title("🧑‍⚖️ AI Legal Assistant (Chat)")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def say(role,text):
    st.session_state.chat.append({"role":role,"content":text})
    with st.chat_message(role):
        st.markdown(text)

def normalize_question(q:str)->str:
    return re.sub(r'[^\w\s]','',q.strip().lower())

def ask_followups(intake):
    check=summarize_missing_info(intake)
    fq_all=check.get("follow_up_questions") or []
    unique_norm=[]
    for q in fq_all:
        nq=normalize_question(q)
        if nq and nq not in st.session_state.answered_followups:
            unique_norm.append((nq,q))
    if check.get("is_complete"): return []
    if (not check.get("missing_keys",[])) or (not check.get("missing_critical",[])):
        if not unique_norm:
            final_q="I have most of the important details. Anything else you'd like to add before I advise?"
            nq=normalize_question(final_q)
            if nq not in st.session_state.answered_followups:
                unique_norm=[(nq,final_q)]
    if not unique_norm and not check.get("is_complete"):
        fallback_q="Please add any missing facts, dates, parties, and what outcome you want."
        nq=normalize_question(fallback_q)
        if nq not in st.session_state.answered_followups:
            unique_norm=[(nq,fallback_q)]
    bullets="\n".join([f"- {orig}" for _,orig in unique_norm])
    say("assistant",f"To advise properly, I need a few details:\n{bullets}\n\nReply in one message. You can answer in short phrases.")
    return unique_norm

if st.session_state.phase=="greeting" and not st.session_state.chat:
    say("assistant","Hi! Describe your legal situation.")

user_msg=st.chat_input("Type your message")

if user_msg:
    say("user",user_msg)

    if st.session_state.phase in ("greeting","intake") and st.session_state.intake is None:
        with st.spinner("Analyzing your case..."):
            predicted_domains=domain_classifier.classify_domains(user_msg)
            st.session_state.predicted_domains=predicted_domains
            raw_outputs=intake_parser.run_domain_intake(user_msg,predicted_domains)
            intake=intake_formatter.format_and_merge_intake(raw_outputs)
            intake["country"]=st.session_state.country_code
            if not intake.get("domain_specific") and intake.get("domains"):
                ds={}
                gf=intake.get("facts",[]) or []
                gq=intake.get("legal_questions",[]) or []
                for d in intake["domains"]:
                    ds[d]={"facts":list(gf),"legal_questions":list(gq)}
                intake["domain_specific"]=ds
            st.session_state.intake=intake
            st.session_state.phase="followups"
        needed=ask_followups(st.session_state.intake)
        if not needed: st.session_state.phase="reason"

    elif st.session_state.phase=="followups":
        answers={"facts":[]}
        parts=[p.strip() for p in user_msg.replace("\n",";").replace(".",";").split(";") if p.strip()]
        answers["facts"].extend(parts)
        st.session_state.intake=merge_user_responses(st.session_state.intake,answers)
        user_input_augmented=(user_msg if len(user_msg)<1500 else user_msg[:1500])
        raw_outputs=intake_parser.run_domain_intake(user_input_augmented,st.session_state.predicted_domains)
        intake2=intake_formatter.format_and_merge_intake(raw_outputs)
        intake2["country"]=st.session_state.country_code
        if not intake2.get("domain_specific") and intake2.get("domains"):
            ds={}
            gf=intake2.get("facts",[]) or []
            gq=intake2.get("legal_questions",[]) or []
            for d in intake2["domains"]:
                ds[d]={"facts":list(gf),"legal_questions":list(gq)}
            intake2["domain_specific"]=ds
        st.session_state.intake=merge_user_responses(st.session_state.intake,intake2)
        asked_now=ask_followups(st.session_state.intake)
        for nq,_ in asked_now: st.session_state.answered_followups.add(nq)
        if not asked_now: st.session_state.phase="reason"

    if st.session_state.phase=="reason":
        with st.spinner("Retrieving laws and generating guidance..."):
            retrieved_laws=retriever.retrieve_relevant_laws(st.session_state.intake)
            final_answer=reasoner.reason_on_case(st.session_state.intake,retrieved_laws)
        for domain,advice in final_answer.items():
            say("assistant",f"### {domain.replace('_',' ').title()}\n\n{advice}")
        st.session_state.phase="done"
        say("assistant","If you’d like, share more details or ask follow-up questions.")
