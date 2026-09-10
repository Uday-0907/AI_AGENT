import os
from dotenv import load_dotenv

# Document Handling & Vector Store
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Google AI Models (Cloud-based, lightweight, zero PyTorch/DLL requirements)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Core Agent Tools & Orchestration
from langchain_core.tools import tool, create_retriever_tool
from langgraph.prebuilt import create_react_agent

load_dotenv()


class AgentBuildError(Exception):
    """Raised when the agent pipeline fails to build (bad file, bad key, etc.)."""
    pass


def _check_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise AgentBuildError(
            "GOOGLE_API_KEY is missing. Check your .env file is in the project "
            "root and contains GOOGLE_API_KEY=your_key_here."
        )
    return key


def _load_document(data_file_path: str):
    if not os.path.exists(data_file_path):
        raise AgentBuildError(f"File not found: {data_file_path}")

    try:
        if data_file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(data_file_path)
        else:
            loader = TextLoader(data_file_path)
        docs = loader.load()
    except Exception as e:
        raise AgentBuildError(f"Could not read document: {e}")

    if not docs or not any(d.page_content.strip() for d in docs):
        raise AgentBuildError(
            "Document loaded but appears empty (e.g. a scanned/image-only PDF "
            "with no extractable text)."
        )
    return docs


# ---------------------------------------------------------------------------
# SWAPPABLE DOMAIN TOOLS
# On competition day, replace / extend these with tools that match the actual
# Round 2 problem statement. Three generic patterns are provided below so you
# only need to edit the body of the function, not the wiring.
# ---------------------------------------------------------------------------

@tool
def calculator_tool(expression: str) -> str:
    """Evaluates a basic arithmetic expression, e.g. '12 * (4 + 3)'.
    Use for any math the agent needs to do (totals, averages, differences)."""
    try:
        # Restricted eval: only digits, operators, parentheses, decimal points.
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: expression contains disallowed characters."
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def data_lookup_tool(query: str) -> str:
    """Looks up a value from a small in-memory dataset by key.
    SWAP THIS on competition day: replace the `mock_db` dict below with
    whatever structured data the problem statement gives you (a CSV loaded
    into a dict, a JSON config, product/policy records, etc.)."""
    mock_db = {
        "example_key": "example_value",
    }
    key = query.strip().lower()
    if key in mock_db:
        return f"Found: {mock_db[key]}"
    return f"No entry found for '{query}'. Available keys: {list(mock_db.keys())}"


@tool
def execute_custom_action(data_input: str) -> str:
    """Executes a domain-specific operation, calculation, or data check.
    SWAP THIS on competition day: this is the placeholder for whatever the
    Round 2 problem statement actually asks the agent to *do* (call an API,
    write to a database, run a validation rule, trigger a workflow, etc.)."""
    return f"[Tool Action Output]: Successfully processed '{data_input}'"


def build_agent(data_file_path: str, extra_tools=None):
    """Builds and returns a compiled LangGraph ReAct agent.

    Raises AgentBuildError with a human-readable message on any failure,
    so the Streamlit UI can show something useful instead of a raw traceback.
    """
    api_key = _check_api_key()
    docs = _load_document(data_file_path)

    try:
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        ).split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )

        vectorstore = FAISS.from_documents(splits, embeddings)

        rag_tool = create_retriever_tool(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            name="search_documents",
            description="Searches through domain context, policies, or challenge datasets.",
        )
    except Exception as e:
        raise AgentBuildError(f"Failed to build vector store / retriever: {e}")

    tools = [rag_tool, calculator_tool, data_lookup_tool, execute_custom_action]
    if extra_tools:
        tools.extend(extra_tools)

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
        return create_react_agent(llm, tools)
    except Exception as e:
        raise AgentBuildError(f"Failed to initialize the LLM / agent: {e}")