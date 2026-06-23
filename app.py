import os
import uuid
import streamlit as st
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from state.state import TravelState
from nodes.flight_node import flight_agent
from nodes.hotel_agent_node import hotel_agent
from nodes.itenary_node import itinerary_agent
from nodes.final_agent_node import final_agent

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ AI Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e0e0e0 !important;
    }

    /* ── Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero-banner h1 {
        color: #fff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Agent Pipeline Cards ── */
    .pipeline-container {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .pipeline-step {
        flex: 1;
        min-width: 140px;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        position: relative;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .pipeline-step .step-icon {
        font-size: 1.5rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .step-waiting {
        background: rgba(255,255,255,0.05);
        color: rgba(255,255,255,0.4);
    }
    .step-active {
        background: rgba(102, 126, 234, 0.25);
        color: #a78bfa;
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102,126,234,0.3);
        animation: pulse 1.5s infinite;
    }
    .step-done {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border-color: #34d399;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102,126,234,0.3); }
        50% { box-shadow: 0 0 30px rgba(102,126,234,0.5); }
    }

    /* ── Chat Messages ── */
    .chat-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        max-height: 55vh;
        overflow-y: auto;
    }
    .msg-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 0.92rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.25);
    }
    .msg-ai {
        background: rgba(255,255,255,0.07);
        color: #e0e0e0;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 0;
        max-width: 85%;
        font-size: 0.92rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Input Area ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #fff !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 15px rgba(102,126,234,0.3) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.35) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
    }

    /* ── Info Cards in Sidebar ── */
    .info-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-card h4 {
        color: #a78bfa;
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
    }
    .info-card p {
        color: rgba(255,255,255,0.6);
        margin: 0;
        font-size: 0.8rem;
    }

    /* ── Expander (results sections) ── */
    .stExpander {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }

    /* ── DB Status Badge ── */
    .db-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .db-connected {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52,211,153,0.3);
    }
    .db-disconnected {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
        border: 1px solid rgba(248,113,113,0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_resource
def get_db_connection():
    """Create a persistent PostgreSQL connection."""
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL:
        return None, None
    try:
        conn = psycopg.connect(DB_URL, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return conn, checkpointer
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None, None


def build_graph(checkpointer=None):
    """Build the LangGraph travel agent pipeline."""
    graph_builder = StateGraph(TravelState)

    graph_builder.add_node("flight_node", flight_agent)
    graph_builder.add_node("hotel_node", hotel_agent)
    graph_builder.add_node("itenary_node", itinerary_agent)
    graph_builder.add_node("final_node", final_agent)

    graph_builder.add_edge(START, "flight_node")
    graph_builder.add_edge("flight_node", "hotel_node")
    graph_builder.add_edge("hotel_node", "itenary_node")
    graph_builder.add_edge("itenary_node", "final_node")
    graph_builder.add_edge("final_node", END)

    if checkpointer:
        return graph_builder.compile(checkpointer=checkpointer)
    return graph_builder.compile()


def render_pipeline(current_step):
    """Render the agent pipeline progress bar."""
    steps = [
        ("✈️", "Flight Agent", "flight_node"),
        ("🏨", "Hotel Agent", "hotel_node"),
        ("📋", "Itinerary Agent", "itenary_node"),
        ("🎯", "Final Agent", "final_node"),
    ]
    step_names = [s[2] for s in steps]

    if current_step is None:
        idx = -1
    elif current_step == "done":
        idx = len(steps)
    else:
        idx = step_names.index(current_step) if current_step in step_names else -1

    html = '<div class="pipeline-container">'
    for i, (icon, label, _) in enumerate(steps):
        if i < idx or current_step == "done":
            cls = "step-done"
            status = "✓"
        elif i == idx:
            cls = "step-active"
            status = "⟳"
        else:
            cls = "step-waiting"
            status = "○"
        html += f"""
        <div class="pipeline-step {cls}">
            <span class="step-icon">{icon}</span>
            {label}<br>
            <small>{status}</small>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_{uuid.uuid4().hex[:8]}"
if "pipeline_step" not in st.session_state:
    st.session_state.pipeline_step = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Travel Planner")
    st.markdown("---")

    # DB connection status
    conn, checkpointer = get_db_connection()
    if conn:
        st.markdown(
            '<span class="db-badge db-connected">● PostgreSQL Connected</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="db-badge db-disconnected">● PostgreSQL Disconnected</span>',
            unsafe_allow_html=True
        )
        st.caption("Running without memory persistence.")

    st.markdown("---")

    # Thread management
    st.markdown("### 🧵 Session")
    st.markdown(f"""
    <div class="info-card">
        <h4>Thread ID</h4>
        <p>{st.session_state.thread_id}</p>
    </div>
    """, unsafe_allow_html=True)

    custom_thread = st.text_input(
        "Switch Thread ID",
        placeholder="e.g. user_huzaifa",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Switch", use_container_width=True):
            if custom_thread.strip():
                st.session_state.thread_id = custom_thread.strip()
                
                # 1. Temporarily build the app to fetch the database state
                app = build_graph(checkpointer)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                state_snapshot = app.get_state(config)
                
                # 2. Clear current UI state
                st.session_state.messages = []
                st.session_state.last_result = None
                st.session_state.pipeline_step = None
                
                # 3. If this thread exists in the DB, load its messages back into the UI!
                if state_snapshot and state_snapshot.values:
                    saved_messages = state_snapshot.values.get("messages", [])
                    for m in saved_messages:
                        # Map LangChain message types to your UI roles
                        if isinstance(m, HumanMessage):
                            st.session_state.messages.append({"role": "user", "content": m.content})
                        elif isinstance(m, AIMessage):
                            st.session_state.messages.append({"role": "assistant", "content": m.content})
                    
                    # Restore the last detailed results so the expanders work
                    st.session_state.last_result = state_snapshot.values
                    st.session_state.pipeline_step = "done"

                st.rerun()
    with col2:
        if st.button("🆕 New", use_container_width=True):
            st.session_state.thread_id = f"user_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.pipeline_step = None
            st.rerun()

    st.markdown("---")

    # Agent info
    st.markdown("### 🤖 Agent Pipeline")
    agents_info = [
        ("✈️ Flight Agent", "Searches live flights via AviationStack API"),
        ("🏨 Hotel Agent", "Finds hotels using Tavily web search"),
        ("📋 Itinerary Agent", "Creates a structured travel plan via LLM"),
        ("🎯 Final Agent", "Generates the polished final response"),
    ]
    for name, desc in agents_info:
        st.markdown(f"""
        <div class="info-card">
            <h4>{name}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.pipeline_step = None
        st.rerun()


# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>✈️ AI Travel Planner</h1>
    <p>Powered by multi-agent LangGraph pipeline &bull; Flights &bull; Hotels &bull; Itineraries</p>
</div>
""", unsafe_allow_html=True)

# Pipeline status
render_pipeline(st.session_state.pipeline_step)

# Chat history
if st.session_state.messages:
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-user">🧑 {msg["content"]}</div>'
        else:
            # Escape HTML-like content and preserve newlines
            content = msg["content"].replace("\n", "<br>")
            chat_html += f'<div class="msg-ai">🤖 {content}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: rgba(255,255,255,0.3);">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌏</div>
        <p style="font-size: 1.1rem;">Enter your travel request below to get started!</p>
        <p style="font-size: 0.85rem;">Try: "Plan a 5-day trip from New York to Tokyo"</p>
    </div>
    """, unsafe_allow_html=True)

# Show detailed results in expandable sections
if st.session_state.last_result:
    result = st.session_state.last_result
    st.markdown("### 📊 Detailed Agent Results")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("✈️ Flight Results", expanded=False):
            st.markdown(result.get("flight_results", "No data"))
    with col2:
        with st.expander("🏨 Hotel Results", expanded=False):
            st.markdown(result.get("hotel_results", "No data"))
    with st.expander("📋 Full Itinerary", expanded=False):
        st.markdown(result.get("itinerary", "No data"))


# ── Input + Invoke ───────────────────────────────────────────────────────────
user_input = st.chat_input("Where do you want to travel? ✈️")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build graph
    app = build_graph(checkpointer)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "flight_results": "",
        "hotel_results": "",
        "itinerary": "",
        "llm_calls": 0,
    }

    # Run with step-by-step status updates
    with st.spinner("🔄 Agents are working on your travel plan..."):
        step_order = ["flight_node", "hotel_node", "itenary_node", "final_node"]
        status_placeholder = st.empty()

        try:
            if checkpointer:
                # Stream node-by-node for live pipeline updates
                final_result = None
                for event in app.stream(initial_state, config=config):
                    for node_name in event:
                        if node_name in step_order:
                            st.session_state.pipeline_step = node_name
                        final_result = event[node_name]

                # Merge all partial results into a full state view
                full_state = app.get_state(config).values
                st.session_state.last_result = full_state
            else:
                # Without checkpointer — use invoke
                result = app.invoke(initial_state)
                st.session_state.last_result = result

            st.session_state.pipeline_step = "done"

            # Extract the final AI response
            state = st.session_state.last_result
            final_messages = state.get("messages", [])
            # Get the last AI message (from final_agent)
            ai_response = ""
            for m in reversed(final_messages):
                if hasattr(m, "content") and isinstance(m, AIMessage):
                    ai_response = m.content
                    break

            if not ai_response:
                ai_response = state.get("itinerary", "Travel plan generated!")

            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )

        except Exception as e:
            st.session_state.pipeline_step = None
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            )

    st.rerun()
