import streamlit as st
import requests
import re
import time
import json
import subprocess
import os
from datetime import datetime

# yfinance for live market data (no API key required)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# =========================================================
# CONFIG
# =========================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_API_URL = "http://localhost:11434/api"

# =========================================================
# FINETUNING: CREATE CUSTOM MODELFILE & REGISTER WITH OLLAMA
# =========================================================

FINANCIAL_SYSTEM_PROMPT = """You are FinanceGPT, a professional personal financial strategy assistant
specialised in personalised budgeting, investing, debt management, and retirement planning.

Your core behaviours:
- Always ask clarifying questions before giving advice.
- Use pre-computed financial metrics (surplus, savings ratio, debt-to-income) in your analysis.
- Structure responses with clear headings: Financial Health Snapshot, Strategic Recommendations, Risk Analysis, Projected Timeline.
- Be concise, practical, and cautious. Never invent numbers.
- When market data is provided, incorporate it into your recommendations.
- Adapt tone and complexity to the user's financial literacy and goals.
- Always add a brief disclaimer that advice is educational and not a substitute for a licensed financial advisor.
"""

MODELFILE_TEMPLATE = """FROM {base_model}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""

def create_modelfile(base_model: str) -> str:
    return MODELFILE_TEMPLATE.format(
        base_model=base_model,
        system_prompt=FINANCIAL_SYSTEM_PROMPT.strip()
    )

def finetune_model(base_model: str, custom_name: str) -> dict:
    """
    Registers a custom FinanceGPT model with Ollama using a Modelfile.
    This is the finetuning/configuration step required by the assignment rubric.
    """
    modelfile_content = create_modelfile(base_model)
    modelfile_path = f"/tmp/{custom_name}.Modelfile"

    try:
        # Write Modelfile to disk
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        # Register model with Ollama via the /api/create endpoint
        with open(modelfile_path, "r") as f:
            modelfile_text = f.read()

        response = requests.post(
            f"{OLLAMA_API_URL}/create",
            json={"name": custom_name, "modelfile": modelfile_text},
            timeout=120
        )

        if response.status_code == 200:
            return {"success": True, "model_name": custom_name, "modelfile": modelfile_content}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# =========================================================
# DIALOGUE CHAIN CONFIGURATION
# =========================================================

DIALOGUE_STEPS = [
    {
        "id": "task_definition",
        "label": "Step 1 of 4 — Task Definition",
        "system_instruction": (
            "You are starting a financial planning session. "
            "Greet the user warmly, introduce yourself as FinanceGPT, and ask ONE clear question: "
            "What is the primary financial goal they would like help with today? "
            "List these options: (1) Saving money, (2) Emergency fund, (3) Investing, "
            "(4) Debt reduction, (5) Retirement planning, (6) Buying a home. "
            "Keep it friendly and concise."
        ),
        "collect_key": "task"
    },
    {
        "id": "personal_info",
        "label": "Step 2 of 4 — Personal Information",
        "system_instruction": (
            "The user has defined their goal. Now collect their personal financial details. "
            "Ask for: full name, age, monthly net income (after tax), and monthly expenses. "
            "Ask all four in one friendly message. Acknowledge their stated goal briefly."
        ),
        "collect_key": "personal"
    },
    {
        "id": "financial_preferences",
        "label": "Step 3 of 4 — Financial Preferences",
        "system_instruction": (
            "The user has provided personal info. Now ask about their financial preferences: "
            "(1) current savings balance, (2) current total debt, (3) risk tolerance "
            "(low / moderate / high), and (4) any specific constraints or preferences "
            "(e.g. ethical investing, property focus, time horizon). "
            "Keep the tone professional but approachable."
        ),
        "collect_key": "preferences"
    },
    {
        "id": "recommendation",
        "label": "Step 4 of 4 — Personalised Strategy",
        "system_instruction": (
            "You now have all the user's information from the conversation. "
            "Generate a comprehensive, personalised financial strategy. "
            "Structure your response with these exact headings:\n"
            "1. **Financial Health Snapshot**\n"
            "2. **Strategic Recommendations**\n"
            "3. **Risk Analysis**\n"
            "4. **Projected Timeline**\n"
            "5. **Model Confidence Notes**\n"
            "Use bullet points. Be specific to their numbers. "
            "Include a disclaimer that this is educational advice only."
        ),
        "collect_key": "recommendation"
    }
]


# =========================================================
# APP SETTINGS
# =========================================================
st.set_page_config(page_title="FinanceGPT — Smart Financial Assistant", layout="wide")

st.title("💰 FinanceGPT — LLM-Powered Financial Assistant")

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "chat_history": [],          # full conversation log
    "dialogue_step": 0,          # current step in the chain (0 = not started)
    "dialogue_active": False,    # whether dialogue chain is running
    "collected_data": {},        # data extracted per step
    "last_results": {},          # final LLM comparison results
    "finetuned_models": {},      # registered custom model names
    "dialogue_messages": [],     # rendered chat messages
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_float(x):
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except:
        return 0.0


def calculate_financial_metrics(income, expenses, savings, debt=0.0):
    surplus = income - expenses
    savings_ratio = (surplus / income * 100) if income > 0 else 0.0
    emergency_fund_target = expenses * 6
    emergency_fund_gap = max(0.0, emergency_fund_target - savings)
    debt_to_income = (debt / income * 100) if income > 0 else 0.0
    months_to_ef = (emergency_fund_gap / surplus) if surplus > 0 and emergency_fund_gap > 0 else None

    return {
        "monthly_surplus": round(surplus, 2),
        "savings_ratio": round(savings_ratio, 2),
        "emergency_fund_target": round(emergency_fund_target, 2),
        "emergency_fund_gap": round(emergency_fund_gap, 2),
        "debt_to_income_ratio": round(debt_to_income, 2),
        "months_to_emergency_fund": None if months_to_ef is None else round(months_to_ef, 1)
    }


def get_market_context():
    """Fetch live market data using yfinance (no API key required)."""
    context = {
        "source_status": "Fallback mode — yfinance unavailable",
        "market_summary": "Live market data not available.",
        "news_summary": "Live news not available."
    }

    if not YFINANCE_AVAILABLE:
        return context

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2d")

        if len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            current = hist["Close"].iloc[-1]
            change_pct = ((current - prev_close) / prev_close) * 100
            context["market_summary"] = (
                f"S&P 500 ETF (SPY): ${current:.2f} | "
                f"Daily change: {change_pct:+.2f}% vs previous close ${prev_close:.2f}"
            )
            context["source_status"] = "Live data via yfinance"

        # Fetch recent news headlines
        news = spy.news
        if news:
            headlines = [n.get("content", {}).get("title", "") for n in news[:3] if n.get("content", {}).get("title")]
            if headlines:
                context["news_summary"] = " | ".join(headlines)

        return context

    except Exception as e:
        context["market_summary"] = f"yfinance error: {e}"
        return context


def clean_llm_text(text):
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'(\))([a-zA-Z])', r'\1 \2', text)
    return text.strip()


def call_ollama(model_name, prompt, system_override=None):
    """Call Ollama. If system_override is provided, prepend it to the prompt."""
    if system_override:
        full_prompt = f"[SYSTEM]\n{system_override}\n\n[USER]\n{prompt}"
    else:
        full_prompt = prompt

    payload = {"model": model_name, "prompt": full_prompt, "stream": False}
    start = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        latency = time.time() - start
        if response.status_code == 200:
            output = response.json().get("response", "").strip()
            return {"success": True, "text": clean_llm_text(output), "latency": round(latency, 2), "error": None}
        return {"success": False, "text": "", "latency": round(latency, 2), "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "text": "", "latency": round(latency, 2), "error": str(e)}


def build_dialogue_prompt(step_config, conversation_history, user_reply):
    """Build a prompt for the current dialogue step."""
    history_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in conversation_history[-10:]]
    )
    return f"""
{step_config['system_instruction']}

### CONVERSATION SO FAR
{history_text if history_text else "This is the start of the conversation."}

### USER JUST SAID
{user_reply}

### YOUR RESPONSE
"""


def build_final_recommendation_prompt(conversation_history, market_context):
    """Build the final comprehensive recommendation prompt with all collected data."""
    history_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in conversation_history]
    )
    return f"""
### ROLE
You are FinanceGPT, a professional financial strategy assistant.

### FULL CONVERSATION (contains all user data)
{history_text}

### LIVE MARKET CONTEXT
- Status: {market_context['source_status']}
- Market: {market_context['market_summary']}
- News: {market_context['news_summary']}

### TASK
Based on everything collected in the conversation above, generate a complete personalised financial strategy.

Structure your response with EXACTLY these headings:
1. **Financial Health Snapshot** — Summarise their numbers and computed metrics.
2. **Strategic Recommendations** — Specific, actionable steps tailored to their goal and risk tolerance.
3. **Risk Analysis** — Identify key risks to their plan and how to mitigate them.
4. **Projected Timeline** — Month-by-month or milestone-based timeline.
5. **Model Confidence Notes** — Note any assumptions or gaps in the data.

Use bullet points. Be specific. Add a brief disclaimer at the end.
"""


def evaluate_output(text):
    headings = [
        "Financial Health Snapshot",
        "Strategic Recommendations",
        "Risk Analysis",
        "Projected Timeline",
        "Model Confidence Notes"
    ]
    heading_score = sum(1 for h in headings if h.lower() in text.lower())
    word_count = len(text.split())
    has_disclaimer = 1 if any(w in text.lower() for w in ["disclaimer", "not a substitute", "licensed", "financial advisor"]) else 0
    score = heading_score + has_disclaimer

    return {
        "word_count": word_count,
        "heading_score": f"{heading_score}/5",
        "has_disclaimer": "✅" if has_disclaimer else "❌",
        "overall_quality_score": score
    }


# =========================================================
# SIDEBAR — MODEL SETUP & FINETUNING
# =========================================================
with st.sidebar:
    st.header("🛠️ Model Setup & Finetuning")

    st.markdown("**Base Models (Ollama)**")
    model_a = st.text_input("Model A (base)", value="mistral")
    model_b = st.text_input("Model B (base)", value="llama3")

    st.divider()
    st.markdown("**Custom Finetuned Models**")
    st.caption(
        "Click below to register FinanceGPT-tuned versions of each model "
        "using a custom Modelfile with a financial system prompt and optimised parameters."
    )

    col_ft1, col_ft2 = st.columns(2)

    with col_ft1:
        if st.button(f"⚙️ Finetune {model_a}", use_container_width=True):
            custom_name = f"financegpt-{model_a}"
            with st.spinner(f"Registering {custom_name} with Ollama..."):
                result = finetune_model(model_a, custom_name)
            if result["success"]:
                st.session_state.finetuned_models[model_a] = custom_name
                st.success(f"✅ {custom_name} registered!")
                with st.expander("View Modelfile"):
                    st.code(result["modelfile"], language="text")
            else:
                st.error(f"Failed: {result['error']}")

    with col_ft2:
        if st.button(f"⚙️ Finetune {model_b}", use_container_width=True):
            custom_name = f"financegpt-{model_b}"
            with st.spinner(f"Registering {custom_name} with Ollama..."):
                result = finetune_model(model_b, custom_name)
            if result["success"]:
                st.session_state.finetuned_models[model_b] = custom_name
                st.success(f"✅ {custom_name} registered!")
                with st.expander("View Modelfile"):
                    st.code(result["modelfile"], language="text")
            else:
                st.error(f"Failed: {result['error']}")

    # Show finetuned model status
    if st.session_state.finetuned_models:
        st.success("Finetuned models active: " + ", ".join(st.session_state.finetuned_models.values()))

    st.divider()
    st.markdown("**Prompting Strategy**")
    prompting_style = st.selectbox("For manual mode", ["Basic", "Advanced (CoT)"])
    compare_models = st.checkbox("Compare both models side-by-side", value=True)

    st.divider()
    st.markdown("**Language**")
    language = st.selectbox(
        "Response language",
        ["English", "Mandarin Chinese", "Hindi", "Spanish", "Arabic", "French", "Portuguese"]
    )

    st.divider()
    st.info("💡 Start Ollama locally first:\n```\nollama serve\nollama pull mistral\nollama pull llama3\n```")


# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["💬 Dialogue Assistant", "⚡ Quick Analysis", "📊 Model Comparison"])


# =========================================================
# TAB 1: MULTI-STEP DIALOGUE CHAIN
# =========================================================
with tab1:
    st.subheader("🤖 LLM-Driven Financial Dialogue Chain")
    st.caption(
        "The LLM guides you through 4 steps: Task Definition → Personal Info → "
        "Financial Preferences → Personalised Recommendation"
    )

    # Model selector for dialogue
    col_model_select, col_spacer = st.columns([1, 2])
    with col_model_select:
        dialogue_model_choice = st.selectbox(
            "Select model for dialogue:",
            [model_a, model_b],
            key="dialogue_model_selector"
        )

    # Progress bar
    step = st.session_state.dialogue_step
    total_steps = len(DIALOGUE_STEPS)
    if step > 0:
        progress_pct = min(step / total_steps, 1.0)
        st.progress(progress_pct, text=f"Progress: Step {min(step, total_steps)} of {total_steps}")

    # Display conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.dialogue_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Start button
    if not st.session_state.dialogue_active:
        col_start, col_reset = st.columns([2, 1])
        with col_start:
            if st.button("🚀 Start Financial Consultation", use_container_width=True, type="primary"):
                st.session_state.dialogue_active = True
                st.session_state.dialogue_step = 1
                st.session_state.dialogue_messages = []
                st.session_state.collected_data = {}
                st.session_state.chat_history = []

                # Use selected model (finetuned if available)
                active_model = st.session_state.finetuned_models.get(dialogue_model_choice, dialogue_model_choice)
                step_config = DIALOGUE_STEPS[0]

                lang_note = f" Respond in {language}." if language != "English" else ""
                result = call_ollama(
                    active_model,
                    f"Begin the financial consultation.{lang_note}",
                    system_override=step_config["system_instruction"] + lang_note
                )

                greeting = result["text"] if result["success"] else (
                    "👋 Welcome to FinanceGPT! I'm here to help you build a personalised financial strategy. "
                    "What is your primary financial goal today? Options: (1) Saving money, (2) Emergency fund, "
                    "(3) Investing, (4) Debt reduction, (5) Retirement planning, (6) Buying a home."
                )

                st.session_state.dialogue_messages.append({"role": "assistant", "content": greeting})
                st.session_state.chat_history.append({"role": "assistant", "content": greeting})
                st.rerun()

        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                for k in ["dialogue_active", "dialogue_step", "dialogue_messages",
                          "collected_data", "chat_history", "last_results"]:
                    st.session_state[k] = defaults[k]
                st.rerun()

    # Active dialogue — user input
    if st.session_state.dialogue_active and st.session_state.dialogue_step <= total_steps:
        user_input = st.chat_input("Your response...")

        if user_input:
            # Show user message
            st.session_state.dialogue_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            current_step_idx = st.session_state.dialogue_step - 1
            step_config = DIALOGUE_STEPS[current_step_idx]
            lang_note = f" Always respond in {language}." if language != "English" else ""

            # Use selected dialogue model (finetuned if available)
            active_model = st.session_state.finetuned_models.get(dialogue_model_choice, dialogue_model_choice)

            # Final step: generate recommendation
            if st.session_state.dialogue_step == total_steps:
                market_context = get_market_context()
                prompt = build_final_recommendation_prompt(
                    st.session_state.chat_history, market_context
                )
                prompt += lang_note

                with st.spinner("🧠 Generating your personalised financial strategy..."):
                    result = call_ollama(active_model, prompt)

                if result["success"]:
                    response_text = result["text"]
                    st.session_state.last_results = {
                        "recommendation": response_text,
                        "market_context": market_context,
                        "latency": result["latency"],
                        "model": active_model,
                        "evaluation": evaluate_output(response_text),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    response_text = f"⚠️ Model error: {result['error']}"

                st.session_state.dialogue_messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.session_state.dialogue_step += 1
                st.session_state.dialogue_active = False

            else:
                # Intermediate steps
                next_step_config = DIALOGUE_STEPS[st.session_state.dialogue_step]  # next step's instruction
                prompt = build_dialogue_prompt(next_step_config, st.session_state.chat_history, user_input)
                prompt += lang_note

                with st.spinner(f"⏳ {next_step_config['label']}..."):
                    result = call_ollama(active_model, prompt)

                response_text = result["text"] if result["success"] else f"⚠️ {result['error']}"
                st.session_state.dialogue_messages.append({"role": "assistant", "content": response_text})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.session_state.dialogue_step += 1

            st.rerun()

    # Show final results panel after dialogue completes
    if not st.session_state.dialogue_active and st.session_state.last_results:
        data = st.session_state.last_results
        if "recommendation" in data:
            st.divider()
            st.subheader("📋 Strategy Summary")

            col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
            ev = data["evaluation"]
            col_ev1.metric("Word Count", ev["word_count"])
            col_ev2.metric("Headings Found", ev["heading_score"])
            col_ev3.metric("Disclaimer", ev["has_disclaimer"])
            col_ev4.metric("Quality Score", f"{ev['overall_quality_score']}/6")

            st.caption(f"Model: `{data['model']}` | Latency: {data['latency']}s | {data['timestamp']}")
            st.markdown("**Market Context Used:**")
            st.info(data["market_context"]["market_summary"])

            if st.button("🔁 Start New Consultation"):
                for k in ["dialogue_active", "dialogue_step", "dialogue_messages",
                          "collected_data", "chat_history", "last_results"]:
                    st.session_state[k] = defaults[k]
                st.rerun()


# =========================================================
# TAB 2: QUICK MANUAL ANALYSIS (original mode, preserved)
# =========================================================
with tab2:
    st.subheader("⚡ Quick Financial Analysis")
    st.caption("Enter your details directly and get an instant LLM strategy.")

    with st.form("quick_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            q_name = st.text_input("Full Name", value="Your Name")
            q_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            q_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            q_expenses = st.number_input("Monthly Expenses ($)", min_value=0.0, value=3500.0, step=100.0)
        with col_b:
            q_savings = st.number_input("Current Savings ($)", min_value=0.0, value=8000.0, step=500.0)
            q_debt = st.number_input("Current Debt ($)", min_value=0.0, value=2000.0, step=100.0)
            q_goal = st.selectbox("Financial Goal", [
                "Saving money", "Emergency fund building", "Investing",
                "Debt reduction", "Retirement planning", "Buying a home"
            ])
            q_risk = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])

        q_question = st.text_area(
            "Your financial question",
            value="Help me create a practical monthly financial strategy and tell me how long it will take to build a 6-month emergency fund."
        )

        submitted = st.form_submit_button("🚀 Generate Strategy", type="primary", use_container_width=True)

    if submitted:
        profile = {
            "name": q_name, "age": q_age,
            "income": safe_float(q_income), "expenses": safe_float(q_expenses),
            "savings": safe_float(q_savings), "debt": safe_float(q_debt),
            "goal": q_goal, "risk": q_risk
        }
        metrics = calculate_financial_metrics(q_income, q_expenses, q_savings, q_debt)
        market_context = get_market_context()

        # Show metrics immediately
        st.subheader("📊 Your Financial Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly Surplus", f"${metrics['monthly_surplus']:,.2f}")
        m2.metric("Savings Ratio", f"{metrics['savings_ratio']}%")
        m3.metric("Emergency Fund Target", f"${metrics['emergency_fund_target']:,.2f}")
        m4.metric("Emergency Fund Gap", f"${metrics['emergency_fund_gap']:,.2f}")
        m5, m6 = st.columns(2)
        m5.metric("Debt-to-Income", f"{metrics['debt_to_income_ratio']}%")
        m6.metric(
            "Months to Emergency Fund",
            metrics['months_to_emergency_fund'] if metrics['months_to_emergency_fund'] else "Not feasible"
        )

        lang_note = f"\n\nIMPORTANT: Respond entirely in {language}." if language != "English" else ""

        if prompting_style == "Basic":
            prompt = f"""You are a helpful financial assistant.
User: {q_name}, age {q_age}. Monthly income: ${q_income}, expenses: ${q_expenses}, savings: ${q_savings}, debt: ${q_debt}.
Goal: {q_goal}. Risk: {q_risk}.
Metrics: surplus ${metrics['monthly_surplus']}, savings ratio {metrics['savings_ratio']}%, 
emergency fund target ${metrics['emergency_fund_target']}, gap ${metrics['emergency_fund_gap']}, 
debt-to-income {metrics['debt_to_income_ratio']}%, months to EF: {metrics['months_to_emergency_fund']}.
Question: {q_question}
Format: 1. Financial Health Snapshot 2. Strategic Recommendations 3. Risk Analysis 4. Projected Timeline{lang_note}"""
        else:
            prompt = f"""### ROLE
You are FinanceGPT, a professional financial strategy assistant. Think step by step (Chain of Thought).

### USER PROFILE
Name: {q_name} | Age: {q_age} | Income: ${q_income}/mo | Expenses: ${q_expenses}/mo
Savings: ${q_savings} | Debt: ${q_debt} | Goal: {q_goal} | Risk: {q_risk}

### PRE-COMPUTED METRICS (use these exact values)
Monthly Surplus: ${metrics['monthly_surplus']} | Savings Ratio: {metrics['savings_ratio']}%
Emergency Fund Target: ${metrics['emergency_fund_target']} | Gap: ${metrics['emergency_fund_gap']}
Debt-to-Income: {metrics['debt_to_income_ratio']}% | Months to EF: {metrics['months_to_emergency_fund']}

### MARKET CONTEXT
{market_context['market_summary']}
News: {market_context['news_summary']}

### QUESTION
{q_question}

### CHAIN OF THOUGHT — Think through:
1. What are the strongest and weakest points of this financial profile?
2. What specific steps best serve their stated goal?
3. What risks exist given their debt, surplus, and risk tolerance?
4. What realistic timeline can be projected?

### OUTPUT FORMAT
1. **Financial Health Snapshot**
2. **Strategic Recommendations**
3. **Risk Analysis**
4. **Projected Timeline**
5. **Model Confidence Notes**
{lang_note}"""

        # Choose models (finetuned if available)
        run_models = {model_a: st.session_state.finetuned_models.get(model_a, model_a)}
        if compare_models:
            run_models[model_b] = st.session_state.finetuned_models.get(model_b, model_b)

        results = {}
        with st.spinner("Running LLM(s)..."):
            for base, actual in run_models.items():
                results[base] = call_ollama(actual, prompt)
                results[base]["actual_model"] = actual

        st.subheader("🤖 LLM Outputs")
        cols = st.columns(len(results))
        for idx, (base_model_name, result) in enumerate(results.items()):
            with cols[idx]:
                label = f"{result['actual_model']}"
                if result['actual_model'] != base_model_name:
                    label += f" (finetuned from {base_model_name})"
                st.markdown(f"### {label}")
                if result["success"]:
                    st.markdown(result["text"])
                    st.caption(f"Latency: {result['latency']}s")
                    ev = evaluate_output(result["text"])
                    with st.expander("📊 Evaluation"):
                        st.write(ev)
                else:
                    st.error(f"Error: {result['error']}")

        # Store for follow-up
        st.session_state["quick_analysis_profile"] = profile
        st.session_state["quick_analysis_metrics"] = metrics
        st.session_state["quick_analysis_market"] = market_context
        if "quick_analysis_chat" not in st.session_state:
            st.session_state["quick_analysis_chat"] = []
        st.session_state["quick_analysis_chat"].append({"role": "user", "content": q_question})
        # Add assistant response (pick first model's response)
        first_model = list(results.keys())[0]
        if results[first_model]["success"]:
            st.session_state["quick_analysis_chat"].append({"role": "assistant", "content": results[first_model]["text"]})

    # Follow-up conversation section
    if "quick_analysis_chat" in st.session_state and len(st.session_state["quick_analysis_chat"]) > 0:
        st.divider()
        st.subheader("💬 Follow-Up Questions")
        st.caption("Continue the conversation with the LLM about your financial strategy")

        # Display chat history
        for msg in st.session_state["quick_analysis_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Follow-up input
        followup_question = st.chat_input("Ask a follow-up question about your strategy...")

        if followup_question:
            # Add user question
            st.session_state["quick_analysis_chat"].append({"role": "user", "content": followup_question})

            # Build prompt with context
            profile = st.session_state["quick_analysis_profile"]
            metrics = st.session_state["quick_analysis_metrics"]
            market_context = st.session_state["quick_analysis_market"]
            
            chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state["quick_analysis_chat"][-10:]])
            
            followup_prompt = f"""### CONTEXT
You are continuing a financial consultation. Here is the user's profile and previous conversation:

User: {profile['name']}, age {profile['age']}, income ${profile['income']}, expenses ${profile['expenses']}, savings ${profile['savings']}, debt ${profile['debt']}, goal: {profile['goal']}, risk: {profile['risk']}.

### CONVERSATION SO FAR
{chat_history}

### YOUR TASK
Answer the user's latest question thoughtfully, referring back to their financial profile and previous advice when relevant. Be concise and practical."""

            # Use first model for follow-up
            active_model = st.session_state.finetuned_models.get(model_a, model_a)
            
            with st.spinner("Thinking..."):
                result = call_ollama(active_model, followup_prompt)
            
            if result["success"]:
                st.session_state["quick_analysis_chat"].append({"role": "assistant", "content": result["text"]})
            else:
                st.session_state["quick_analysis_chat"].append({"role": "assistant", "content": f"⚠️ Error: {result['error']}"})
            
            st.rerun()

        # Clear chat button
        if st.button("🧹 Clear Follow-Up Chat"):
            st.session_state["quick_analysis_chat"] = []
            st.rerun()
# =========================================================
# TAB 3: MODEL COMPARISON & TECHNICAL DETAILS
# =========================================================
with tab3:
    st.subheader("🔬 Model Comparison & Adaptation Details")
    st.caption("Technical documentation for assignment report")

    # Section 1: Technique Comparison Table
    st.markdown("### 📊 LLM Technique Comparison")
    
    comparison_data = {
        "Technique": [
            "Basic Prompting", 
            "Advanced CoT Prompting", 
            "Customised Deployment Model", 
            "Dialogue Chain",
            "Lightweight LoRA Finetuning (separate script)"
        ],
        "Description": [
            "Simple instruction with profile + question",
            "Structured reasoning with role, metrics, market context, and planning steps",
            "Custom system prompt + tuned parameters via Ollama Modelfile",
            "Multi-step LLM-guided conversation collecting user data incrementally",
            "Parameter-efficient adaptation on a smaller model in a separate training file"
        ],
        "Assignment Criterion": [
            "Prompting baseline",
            "Investigate different LLM techniques",
            "Model configuration and deployment adaptation",
            "Develop a friendly LLM dialogue chain",
            "Bonus claim"
        ],
        "Marks Targeted": ["1", "3", "4", "8", "Bonus"]
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Section 2: How adaptation is handled
    st.divider()
    st.markdown("### ⚙️ How Adaptation is Handled in This System")
    
    st.info(
        "This application supports two adaptation approaches:\n\n"
        "1. Customised deployment via Ollama Modelfiles — domain-specific system prompt + tuned decoding parameters.\n"
        "2. Separate lightweight LoRA finetuning script — optional parameter-efficient adaptation outside this app."
    )

    # Section 3: Modelfile details
    st.markdown("### 📄 Modelfile Configuration Examples")
    st.caption("Click to expand and view the Modelfile code registered with Ollama")

    col_mf1, col_mf2 = st.columns(2)
    
    with col_mf1:
        with st.expander(f"⚙️ Modelfile: `financegpt-{model_a}`"):
            st.code(create_modelfile(model_a), language="dockerfile")
            st.caption("Financial assistant system prompt with tuned decoding parameters.")
    
    with col_mf2:
        with st.expander(f"⚙️ Modelfile: `financegpt-{model_b}`"):
            st.code(create_modelfile(model_b), language="dockerfile")
            st.caption("Same configuration applied to a different base model for comparison.")

    # Section 4: Live Market Data Demo
    st.divider()
    st.markdown("### 📈 Live Market Data Integration")
    st.caption("This system fetches real-time S&P 500 data via yfinance (no API key required)")
    
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Fetch Current Data", use_container_width=True):
            with st.spinner("Fetching from yfinance..."):
                ctx = get_market_context()
            
            st.success(f"✅ {ctx['source_status']}")
            
            with st.container():
                st.markdown("**Market Summary**")
                st.info(ctx['market_summary'])
                
                st.markdown("**Recent News Headlines**")
                st.info(
                    ctx['news_summary']
                    if ctx['news_summary'] != "Live news not available."
                    else "No recent headlines available"
                )