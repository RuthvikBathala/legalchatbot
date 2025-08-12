import streamlit as st
from src.pipeline import (
    domain_classifier,
    intake_parser,
    intake_formatter,
    retriever,
    reasoner,
)
from src.pipeline.missing_info_handler import summarize_missing_info
from src.pipeline.merge_user_responses import merge_user_responses

st.set_page_config(page_title="Legal Chatbot", layout="wide")
st.title("🧑‍⚖️ AI Legal Assistant")
st.write("Describe your legal issue below:")

# --- session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "final_output" not in st.session_state:
    st.session_state.final_output = {}
if "formatted_intake" not in st.session_state:
    st.session_state.formatted_intake = None
if "predicted_domains" not in st.session_state:
    st.session_state.predicted_domains = []

# --- input ---
user_input = st.text_area("Your Situation", height=200)

# --- submit ---
if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a legal situation to begin.")
        st.stop()

    with st.spinner("Analyzing your case..."):
        # Step 1: domains
        predicted_domains = domain_classifier.predict_domains(user_input)
        st.session_state.predicted_domains = predicted_domains
        st.session_state.history.append(("🤖 Predicted Domain(s)", str(predicted_domains)))

        # Step 2a: domain prompts → raw outputs
        raw_outputs = intake_parser.run_domain_intake(user_input, predicted_domains)

        # Step 2b: format & merge JSONs (make sure your intake_formatter exposes this name)
        formatted_intake = intake_formatter.format_intake_json(raw_outputs)
        st.session_state.formatted_intake = formatted_intake
        st.session_state.history.append(("📋 Intake Extracted", formatted_intake))

# --- if we have an intake, run the missing-info loop UI ---
if st.session_state.formatted_intake:
    formatted_intake = st.session_state.formatted_intake

    # Loop: ask for missing info until complete
    while True:
        check = summarize_missing_info(formatted_intake)

        if check["is_complete"]:
            st.success("✅ Intake complete. Retrieving relevant laws...")
            break

        st.warning("We need a bit more information before we can continue.")

        # Build a small form to collect answers (global + per-domain)
        with st.form(key="followups"):
            st.markdown("### 📌 Follow-up Questions")
            # Show LLM-proposed follow-ups (free-form answers become new facts)
            answers = {"facts": []}
            for q in check["follow_up_questions"]:
                resp = st.text_input(q, key=q)
                if resp.strip():
                    answers["facts"].append(resp.strip())

            # If country was missing, let user supply it explicitly
            country_missing = "country" in check["missing_keys"]
            if country_missing:
                st.markdown("### 🌍 Country")
                c = st.text_input("Which country's laws apply?")
                if c.strip():
                    answers["country"] = c.strip()

            # Optional: per-domain refinements (facts / legal_questions)
            st.markdown("### 🧭 Domain-specific details (optional)")
            for domain in formatted_intake.get("domain_specific", {}).keys():
                with st.expander(f"Add info for {domain}"):
                    add_fact = st.text_area(f"{domain}: Add any facts (one per line)", key=f"{domain}_facts")
                    add_qs = st.text_area(f"{domain}: Add any legal questions (one per line)", key=f"{domain}_qs")
                    dom_payload = {}
                    if add_fact.strip():
                        dom_payload["facts"] = [ln.strip() for ln in add_fact.splitlines() if ln.strip()]
                    if add_qs.strip():
                        dom_payload["legal_questions"] = [ln.strip() for ln in add_qs.splitlines() if ln.strip()]
                    if dom_payload:
                        answers[domain] = dom_payload

            submitted = st.form_submit_button("Submit additional details")

        if not submitted:
            st.stop()

        # Merge user answers into intake (supports country, general facts/qs, and per-domain updates)
        formatted_intake = merge_user_responses(formatted_intake, answers)
        st.session_state.formatted_intake = formatted_intake  # persist
        st.session_state.history.append(("🔁 Merged Follow-ups", formatted_intake))

        # Optional (advanced): re-run your 2-step intake to let LLM re-structure with the new info.
        # If you want that, uncomment below:
        # user_input_augmented = user_input + "\n\nAdditional details: " + " ".join(answers.get("facts", []))
        # raw_outputs = intake_parser.run_domain_intake(user_input_augmented, st.session_state.predicted_domains)
        # formatted_intake = intake_formatter.format_intake_json(raw_outputs)
        # st.session_state.formatted_intake = formatted_intake
        # st.session_state.history.append(("🧮 Re-parsed Intake", formatted_intake))

    # Step 4: Retrieve (country-based, domain-aware queries)
    retrieved_laws = retriever.retrieve_relevant_laws(formatted_intake)
    st.session_state.history.append(("📚 Laws Retrieved", f"{ {k: len(v) for k,v in retrieved_laws.items()} }"))

    # Step 5: Reasoner (per-domain JSON answers)
    final_answer = reasoner.reason_on_case(formatted_intake, retrieved_laws)
    st.session_state.history.append(("🧠 Legal Guidance", final_answer))

    st.success("✅ Legal response generated!")
    st.session_state.final_output = final_answer

# --- final display ---
if st.session_state.final_output:
    st.subheader("📑 Final Legal Recommendation")
    for domain, output in st.session_state.final_output.items():
        st.markdown(f"### 🔹 {domain.title()}")
        st.code(output, language="json")

# --- debug/history ---
with st.expander("🕓 Show Pipeline History / Debug"):
    for title, content in st.session_state.history:
        st.markdown(f"**{title}**")
        st.write(content)
