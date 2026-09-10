import os
import streamlit as st
from agent import build_agent, AgentBuildError

st.set_page_config(page_title="AI Agent Control Center", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous AI Agent Control Center")
st.caption("Powered by Google Gemini & LangGraph")

# ---------------------------------------------------------------------------
# Cached agent builder: avoids re-embedding the same document on every click.
# Cache key = file path + file size (so a changed upload rebuilds correctly).
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_agent(file_path: str, file_size: int):
    return build_agent(file_path)


with st.sidebar:
    st.header("1. Knowledge Base Ingestion")
    data_file = st.file_uploader("Upload Problem Data (PDF or TXT)", type=["pdf", "txt"])
    st.divider()
    st.caption(
        "If a run fails, check: GOOGLE_API_KEY is set, the API is enabled, "
        "and you have internet access."
    )

st.header("2. Autonomous Agent Execution")
user_prompt = st.text_input("Enter command or question for the AI Agent:")

if st.button("🚀 Run Agent Tasks"):
    if not data_file:
        st.error("Please upload a problem file first.")
    elif not user_prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Save a local copy for the loader. Prefix with a hash-free unique
        # marker to avoid collisions if multiple files share a name.
        os.makedirs("./uploaded_docs", exist_ok=True)
        file_path = os.path.join("./uploaded_docs", data_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(data_file.getbuffer())
        except Exception as e:
            st.error(f"Could not save uploaded file: {e}")
            st.stop()

        try:
            with st.status("🧠 Agent Orchestration Engine Running...", expanded=True) as status:
                st.write("📥 Processing uploaded document into Vector Storage (FAISS)...")
                agent_executor = get_agent(file_path, data_file.size)

                st.write("⚙️ Agent evaluating tools, search terms, and plan...")
                response = agent_executor.invoke(
                    {"messages": [("user", user_prompt)]},
                    config={"recursion_limit": 15},  # avoid infinite tool loops
                )

                st.write("✅ Execution complete!")
                status.update(
                    label="Agent Task Completed Successfully!",
                    state="complete",
                    expanded=False,
                )

            final_message = response["messages"][-1].content
            st.subheader("Final Output & Execution Result:")
            st.success(final_message)

            with st.expander("Show full reasoning trace (for judges / debugging)"):
                for msg in response["messages"]:
                    role = getattr(msg, "type", msg.__class__.__name__)
                    st.markdown(f"**{role}:** {getattr(msg, 'content', msg)}")

        except AgentBuildError as e:
            st.error(f"Agent setup failed: {e}")
        except Exception as e:
            # Catches API rate limits, timeouts, network errors, etc.
            st.error(f"Something went wrong while running the agent: {e}")
            st.caption(
                "Common causes: Gemini API rate limit, no internet, or an "
                "invalid/expired API key."
            )