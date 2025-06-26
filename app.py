# import streamlit as st # REMOVED
import os
import json
import re
import uuid
# import extra_streamlit_components as esc # REMOVED
from typing import List, Dict, Any, Optional, Callable, Tuple # Added Optional, Callable, Tuple
from functools import lru_cache # For caching similar to st.cache_data / st.cache_resource
import pandas as pd
from PyPDF2 import PdfReader # For PDF processing
# from docx import Document # Imported lower for get_discussion_docx, and locally in process_uploaded_file_taipy
from io import BytesIO
import time # Added for simulated streaming and DOCX export timestamp

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
import ui # ADDED
from taipy.gui import State, Gui, notify # ADDED
from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool

# Import necessary libraries for Hugging Face integration
from huggingface_hub import HfFileSystem

# Initialize HfFileSystem globally
fs = HfFileSystem()
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

_TAIPY_AGENT_INSTANCE: Optional[Any] = None # Global store for the agent
_TAIPY_STATE_DEPENDANT_TOOLS_INITIALIZED: bool = False # Flag for tool init

DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

from tools import UI_ACCESSIBLE_WORKSPACE
from config import HF_USER_MEMORIES_DATASET_ID

MAX_CHAT_HISTORY_MESSAGES = 15

_APP_CONTEXT: Dict[str, Any] = {
    "user_id": None,
    "chat_metadata": {},
    "all_chat_messages": {},
    "uploaded_documents_content": {},
    "uploaded_dataframes_content": {},
    "long_term_memory_enabled_pref": True,
    "session_control_flags_initialized": False,
    "initial_greeting_shown_for_session": False,
    "_last_memory_state_was_enabled": True,
    "llm_settings": {
        "temperature": 0.7,
        "verbosity": 3,
        "search_results_count": 5,
    }
}

def setup_global_llm_settings() -> tuple[bool, str | None]:
    print("Initializing LLM settings...")
    try:
        initialize_agent_settings()
        if Settings.llm and hasattr(Settings.llm, 'temperature'):
            _APP_CONTEXT["llm_settings"]["temperature"] = Settings.llm.temperature
        print("LLM settings initialized successfully.")
        return True, None
    except Exception as e:
        error_message = f"Fatal Error: Could not initialize LLM settings. {e}"
        print(error_message)
        return False, error_message

@lru_cache(maxsize=1)
def _get_initial_greeting_text() -> str:
    return generate_llm_greeting()

@lru_cache(maxsize=32)
def _cached_generate_suggested_prompts(chat_history_tuple: Tuple[Tuple[str, str], ...]) -> List[str]:
    print("Generating suggested prompts (Taipy)...")
    chat_history = [{"role": item[0], "content": item[1]} for item in chat_history_tuple]
    return generate_suggested_prompts(chat_history)

def read_uploaded_document_tool_fn(filename: str) -> str:
    if filename not in _APP_CONTEXT.get("uploaded_documents_content", {}):
        return f"Error: Document '{filename}' not found. Available: {list(_APP_CONTEXT.get('uploaded_documents_content', {}).keys())}"
    return _APP_CONTEXT["uploaded_documents_content"][filename]

def analyze_dataframe_tool_fn(filename: str, head_rows: int = 5) -> str:
    if filename not in _APP_CONTEXT.get("uploaded_dataframes_content", {}):
        return f"Error: DataFrame '{filename}' not found. Available: {list(_APP_CONTEXT.get('uploaded_dataframes_content', {}).keys())}"
    df = _APP_CONTEXT["uploaded_dataframes_content"][filename]
    info_str = f"DataFrame: {filename}\nShape: {df.shape}\nColumns: {', '.join(df.columns)}\nData Types:\n{df.dtypes.to_string()}\n"
    head_rows = max(0, min(head_rows, len(df)))
    if head_rows > 0:
        info_str += f"First {head_rows} rows:\n{df.head(head_rows).to_string()}\n"
    else:
        info_str += "No head rows requested or available.\n"
    info_str += f"Summary Statistics:\n{df.describe().to_string()}\n"
    return info_str

def setup_agent_instance() -> None: # Removed max_search_results from signature, will use _APP_CONTEXT
    global _TAIPY_AGENT_INSTANCE, _TAIPY_STATE_DEPENDANT_TOOLS_INITIALIZED
    if _TAIPY_AGENT_INSTANCE is not None and _TAIPY_STATE_DEPENDANT_TOOLS_INITIALIZED:
        print("AI agent already initialized.")
        return

    print("Initializing AI agent (Taipy)...")
    try:
        uploaded_doc_reader_tool = FunctionTool.from_defaults(fn=read_uploaded_document_tool_fn, name="read_uploaded_document", description="Reads the full text content of a document previously uploaded by the user.")
        dataframe_analyzer_tool = FunctionTool.from_defaults(fn=analyze_dataframe_tool_fn, name="analyze_uploaded_dataframe", description="Provides summary information about a pandas DataFrame previously uploaded by the user.")

        agent_instance = create_orchestrator_agent(
            dynamic_tools=[uploaded_doc_reader_tool, dataframe_analyzer_tool],
            max_search_results=_APP_CONTEXT.get("llm_settings", {}).get("search_results_count", 5)
        )
        _TAIPY_AGENT_INSTANCE = agent_instance
        _TAIPY_STATE_DEPENDANT_TOOLS_INITIALIZED = True
        print("AI agent initialized successfully.")
    except Exception as e:
        error_message = f"Failed to initialize the AI agent. Please check configurations. Error: {e}"
        print(error_message)
        raise RuntimeError(error_message) from e

def get_user_id_and_cookie_pref_from_state(state: State) -> Tuple[str, bool]:
    ltm_pref = state.long_term_memory_enabled_ui
    user_id = getattr(state, 'user_id', None)

    if ltm_pref:
        if not user_id:
            user_id = str(uuid.uuid4())
            state.user_id = user_id
            if hasattr(state, 'eval_js'): # Check if eval_js is available (it should be on a Taipy State object)
                state.eval_js(f"document.cookie = 'user_id={user_id};path=/;max-age=31536000';")
                print(f"Generated new user_id for LTM: {user_id} and attempted to set cookie.")
            else:
                print(f"Generated new user_id for LTM: {user_id} (eval_js not available on state to set cookie).")
        return user_id, ltm_pref
    else:
        temp_user_id = getattr(state, 'temp_user_id', None)
        if not temp_user_id:
            temp_user_id = str(uuid.uuid4())
            state.temp_user_id = temp_user_id
        if hasattr(state, 'eval_js'):
            state.eval_js("document.cookie = 'user_id=;path=/;expires=Thu, 01 Jan 1970 00:00:00 GMT';")
            print(f"Using temporary user_id: {temp_user_id} (attempted to clear user_id cookie).")
        else:
            print(f"Using temporary user_id: {temp_user_id} (eval_js not available on state to clear cookie).")
        return temp_user_id, ltm_pref

def initialize_user_session_data_taipy(state: State) -> None:
    print("Initializing user session data (Taipy)...")
    current_user_id, ltm_enabled = get_user_id_and_cookie_pref_from_state(state)
    _APP_CONTEXT["user_id"] = current_user_id
    _APP_CONTEXT["long_term_memory_enabled_pref"] = ltm_enabled

    if ltm_enabled:
        print(f"Loading user data for user {current_user_id} from Hugging Face...")
        user_data = _load_user_data_from_hf(current_user_id)
        _APP_CONTEXT["chat_metadata"] = user_data["metadata"]
        _APP_CONTEXT["all_chat_messages"] = user_data["messages"]
        state.chat_metadata_ui = user_data["metadata"]
        print(f"Loaded {len(user_data['metadata'])} chats for user {current_user_id}.")
    else:
        print(f"Long-term memory disabled. No historical data loaded for temporary user_id {current_user_id}.")
        _APP_CONTEXT["chat_metadata"] = {}
        _APP_CONTEXT["all_chat_messages"] = {}
        state.chat_metadata_ui = {}
    state.chat_history_lov = list(state.chat_metadata_ui.items())

def _load_user_data_from_hf(user_id: str) -> Dict[str, Any]:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set. Cannot load user data from Hugging Face.")
        return {"metadata": {}, "messages": {}}
    # ... (rest of the HF loading logic remains the same)
    all_chat_metadata = {}
    all_chat_messages = {}
    try:
        metadata_filename_in_repo = f"user_memories/{user_id}_metadata.json"
        messages_filename_in_repo = f"user_memories/{user_id}_messages.json"
        metadata_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{metadata_filename_in_repo}"
        messages_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{messages_filename_in_repo}"
        try:
            metadata_content = fs.read_text(metadata_hf_path, token=hf_token)
            all_chat_metadata = json.loads(metadata_content)
        except Exception: # Broad exception for file not found or parsing error
            all_chat_metadata = {}
        try:
            messages_content = fs.read_text(messages_hf_path, token=hf_token)
            all_chat_messages = json.loads(messages_content)
        except Exception:
            all_chat_messages = {}
        return {"metadata": all_chat_metadata, "messages": all_chat_messages}
    except Exception as e:
        print(f"Error loading user data from Hugging Face for user {user_id}: {e}")
        return {"metadata": {}, "messages": {}}


def save_chat_history(state: State, user_id: str, chat_id: str, messages: List[Dict[str, Any]]):
    if not _APP_CONTEXT.get("long_term_memory_enabled_pref", False): return
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: return
    try:
        messages_filename_in_repo = f"user_memories/{user_id}_messages.json"
        messages_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{messages_filename_in_repo}"
        existing_messages = {}
        try:
            existing_messages_content = fs.read_text(messages_hf_path, token=hf_token)
            existing_messages = json.loads(existing_messages_content)
        except Exception: pass # Ignore if file not found or empty
        existing_messages[chat_id] = messages
        with fs.open(messages_hf_path, "w", token=hf_token) as f:
            f.write(json.dumps(existing_messages, indent=2))
        print(f"Chat history for chat {chat_id} saved to HF.")
    except Exception as e:
        print(f"Error saving chat history to HF for chat {chat_id}: {e}")
        notify(state, "error", f"Error saving chat history: {e}")

def save_chat_metadata(state: State, user_id: str, chat_metadata: Dict[str, str]):
    if not _APP_CONTEXT.get("long_term_memory_enabled_pref", False): return
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token: return
    try:
        metadata_filename_in_repo = f"user_memories/{user_id}_metadata.json"
        metadata_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{metadata_filename_in_repo}"
        with fs.open(metadata_hf_path, "w", token=hf_token) as f:
            f.write(json.dumps(chat_metadata, indent=2))
        print(f"Chat metadata for user {user_id} saved to HF.")
    except Exception as e:
        print(f"Error saving chat metadata to HF for user {user_id}: {e}")
        notify(state, "error", f"Error saving chat metadata: {e}")

def format_chat_history(ui_messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    truncated_messages = ui_messages[-MAX_CHAT_HISTORY_MESSAGES:]
    history = []
    for msg in truncated_messages:
        role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
        history.append(ChatMessage(role=role, content=msg["content"]))
    return history

def get_agent_response(query: str, chat_history: List[ChatMessage]) -> Generator[str, None, None]:
    global _TAIPY_AGENT_INSTANCE
    if not _TAIPY_AGENT_INSTANCE:
        yield "Error: Agent not initialized."
        return
    agent = _TAIPY_AGENT_INSTANCE
    try:
        current_temperature = _APP_CONTEXT.get("llm_settings", {}).get("temperature", 0.7)
        current_verbosity = _APP_CONTEXT.get("llm_settings", {}).get("verbosity", 3)
        if Settings.llm and hasattr(Settings.llm, 'temperature'):
            Settings.llm.temperature = current_temperature
        modified_query = f"Verbosity Level: {current_verbosity}. {query}"
        response = agent.chat(modified_query, chat_history=chat_history)
        response_text = response.response if hasattr(response, 'response') else str(response)
        words = response_text.split(" ")
        for word in words:
            yield word + " "
            time.sleep(0.02) # Adjusted for potentially faster UI updates
    except Exception as e:
        error_message = f"I apologize, but I encountered an error: {str(e)}"
        print(f"Error getting agent response: {type(e).__name__} - {e}")
        yield error_message

def create_new_chat_session_taipy(state: State):
    new_chat_id = str(uuid.uuid4())
    ltm_enabled = state.long_term_memory_enabled_ui
    new_chat_name = "Current Session"
    if ltm_enabled:
        existing_idea_nums = [int(m.group(1)) for name in state.chat_metadata_ui.values() if (m := re.match(r"Idea (\d+)", name))]
        new_chat_name = f"Idea {max(existing_idea_nums) + 1 if existing_idea_nums else 1}"

    current_chat_metadata = dict(state.chat_metadata_ui)
    current_chat_metadata[new_chat_id] = new_chat_name
    state.chat_metadata_ui = current_chat_metadata
    state.chat_history_lov = list(current_chat_metadata.items())
    
    initial_messages = [{"role": "assistant", "content": _get_initial_greeting_text()}]
    state.messages_history = initial_messages
    state.current_chat_id_ui = new_chat_id
    
    messages_tuple = tuple((msg["role"], msg["content"]) for msg in initial_messages) # Create tuple for caching
    state.suggested_prompts_ui = _cached_generate_suggested_prompts(messages_tuple)

    _APP_CONTEXT["chat_metadata"] = dict(state.chat_metadata_ui)
    _APP_CONTEXT["all_chat_messages"][new_chat_id] = list(initial_messages)
    _APP_CONTEXT["current_chat_id"] = new_chat_id

    user_id = _APP_CONTEXT.get("user_id")
    if ltm_enabled and user_id:
        save_chat_metadata(state, user_id, state.chat_metadata_ui)

    print(f"Created new Taipy chat session: '{new_chat_name}' (ID: {new_chat_id})")
    notify(state, "info", f"New chat '{new_chat_name}' started.")

def switch_chat_taipy(state: State, chat_id: str):
    ltm_enabled = state.long_term_memory_enabled_ui
    if not ltm_enabled:
        print("LTM disabled. Cannot switch to historical chats.")
        create_new_chat_session_taipy(state)
        return

    if chat_id not in state.chat_metadata_ui:
        print(f"Error: Chat ID '{chat_id}' not found.")
        notify(state, "error", f"Chat ID '{chat_id}' not found.")
        return

    if _APP_CONTEXT["all_chat_messages"].get(chat_id) is None:
        _APP_CONTEXT["all_chat_messages"][chat_id] = []
            
    state.messages_history = _APP_CONTEXT["all_chat_messages"].get(chat_id, [])
    state.current_chat_id_ui = chat_id
    
    messages_tuple = tuple((msg["role"], msg["content"]) for msg in state.messages_history)
    state.suggested_prompts_ui = _cached_generate_suggested_prompts(messages_tuple)
    print(f"Switched to chat: '{state.chat_metadata_ui.get(chat_id, 'Unknown')}' (ID: {chat_id})")

def delete_chat_session_taipy(state: State, chat_id: str):
    ltm_enabled = state.long_term_memory_enabled_ui
    if not ltm_enabled:
        if chat_id == state.current_chat_id_ui: create_new_chat_session_taipy(state)
        return

    user_id = _APP_CONTEXT.get("user_id")
    if not user_id: notify(state, "error", "User ID not found."); return
        
    try:
        if chat_id in _APP_CONTEXT["all_chat_messages"]: del _APP_CONTEXT["all_chat_messages"][chat_id]
        if chat_id in state.chat_metadata_ui: del state.chat_metadata_ui[chat_id]
        _APP_CONTEXT["chat_metadata"] = dict(state.chat_metadata_ui)
        save_chat_metadata(state, user_id, state.chat_metadata_ui)
        
        # ... (Hugging Face file deletion logic remains similar, ensure it uses user_id from _APP_CONTEXT) ...
        messages_filename_in_repo = f"user_memories/{user_id}_messages.json"
        messages_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/{messages_filename_in_repo}"
        existing_messages = {}
        try:
            existing_messages_content = fs.read_text(messages_hf_path, token=os.getenv("HF_TOKEN"))
            existing_messages = json.loads(existing_messages_content)
        except Exception: pass
        if chat_id in existing_messages:
            del existing_messages[chat_id]
            with fs.open(messages_hf_path, "w", token=os.getenv("HF_TOKEN")) as f:
                f.write(json.dumps(existing_messages, indent=2))

        notify(state, "success", f"Chat '{chat_id}' deleted.")
        if chat_id == state.current_chat_id_ui:
            if state.chat_metadata_ui:
                switch_chat_taipy(state, next(iter(state.chat_metadata_ui)))
            else:
                create_new_chat_session_taipy(state)
        else:
            state.chat_history_lov = list(state.chat_metadata_ui.items())
    except Exception as e:
        notify(state, "error", f"Error deleting chat: {e}")

def rename_chat_taipy(state: State, chat_id: str, new_name: str):
    ltm_enabled = state.long_term_memory_enabled_ui
    if not ltm_enabled: notify(state, "warning", "LTM disabled."); return
    user_id = _APP_CONTEXT.get("user_id")
    if not user_id: notify(state, "error", "User ID not found."); return

    if chat_id and new_name and new_name != state.chat_metadata_ui.get(chat_id):
        state.chat_metadata_ui[chat_id] = new_name
        _APP_CONTEXT["chat_metadata"] = dict(state.chat_metadata_ui)
        save_chat_metadata(state, user_id, state.chat_metadata_ui)
        notify(state, "success", f"Chat renamed to '{new_name}'.")
        state.chat_history_lov = list(state.chat_metadata_ui.items())
    else:
        notify(state, "info", "No change in chat name.")

def get_discussion_markdown(chat_id: str) -> str:
    messages = _APP_CONTEXT.get("all_chat_messages", {}).get(chat_id, [])
    # ... (formatting logic remains the same) ...
    markdown_content = []
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        markdown_content.append(f"**{role}:**\n{content}\n\n---")
    return "\n".join(markdown_content)


def get_discussion_docx(chat_id: str) -> bytes:
    from docx import Document # Import here if not global
    messages = _APP_CONTEXT.get("all_chat_messages", {}).get(chat_id, [])
    chat_name = _APP_CONTEXT.get("chat_metadata", {}).get(chat_id, 'Untitled Chat')
    document = Document()
    document.add_heading(f"Chat Discussion: {chat_name}", level=1)
    document.add_paragraph(f"Exported on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        document.add_heading(f"{role}:", level=3)
        document.add_paragraph(content)
        document.add_paragraph("---")
    byte_stream = BytesIO()
    document.save(byte_stream)
    byte_stream.seek(0)
    return byte_stream.getvalue()

def handle_user_input_taipy(state: State, user_input: str | None):
    prompt_to_process = user_input
    if prompt_to_process:
        user_id = _APP_CONTEXT.get("user_id")
        ltm_enabled = state.long_term_memory_enabled_ui

        if not state.current_chat_id_ui or (len(state.messages_history) == 1 and state.messages_history[0]["role"] == "assistant"):
            if not state.current_chat_id_ui:
                 create_new_chat_session_taipy(state)
            if ltm_enabled and user_id and state.current_chat_id_ui:
                save_chat_metadata(state, user_id, state.chat_metadata_ui)
            print(f"Activated/processing first message for chat '{state.chat_metadata_ui.get(state.current_chat_id_ui)}'.")

        current_messages = list(state.messages_history)
        current_messages.append({"role": "user", "content": prompt_to_process})
        state.messages_history = current_messages

        history_for_agent = format_chat_history(state.messages_history)
        response_generator = get_agent_response(prompt_to_process, chat_history=history_for_agent)

        assistant_response_content = ""
        current_messages = list(state.messages_history) # Ensure it's a list
        current_messages.append({"role": "assistant", "content": "Thinking..."}) # Add placeholder
        state.messages_history = current_messages # Assign back to trigger update

        for chunk in response_generator:
            assistant_response_content += chunk
            # To update a list item in Taipy and ensure reactivity:
            temp_messages = list(state.messages_history)
            if temp_messages: # Ensure list is not empty
                temp_messages[-1]["content"] = assistant_response_content
                state.messages_history = temp_messages # Assign the modified list back
            time.sleep(0.02)

        if state.current_chat_id_ui:
            _APP_CONTEXT["all_chat_messages"][state.current_chat_id_ui] = list(state.messages_history)
        if ltm_enabled and user_id and state.current_chat_id_ui:
            save_chat_history(state, user_id, state.current_chat_id_ui, state.messages_history)

        messages_tuple = tuple((msg["role"], msg["content"]) for msg in state.messages_history)
        state.suggested_prompts_ui = _cached_generate_suggested_prompts(messages_tuple)

def reset_chat_taipy(state: State):
    print("Resetting chat by creating a new session...")
    create_new_chat_session_taipy(state)

def handle_regeneration_request_taipy(state: State):
    if not state.messages_history or state.messages_history[-1]['role'] != 'assistant':
        notify(state, "warning", "Nothing to regenerate.")
        return

    if len(state.messages_history) == 1:
        response_generator = get_agent_response("Regenerate initial greeting", [])
        new_greeting = "".join(list(response_generator)) # Consume generator
        state.messages_history = [{"role": "assistant", "content": new_greeting}]
        if state.long_term_memory_enabled_ui and _APP_CONTEXT.get("user_id") and state.current_chat_id_ui:
            save_chat_history(state, _APP_CONTEXT["user_id"], state.current_chat_id_ui, state.messages_history)
        messages_tuple = tuple((msg["role"], msg["content"]) for msg in state.messages_history)
        state.suggested_prompts_ui = _cached_generate_suggested_prompts(messages_tuple)
        return

    current_messages = list(state.messages_history)
    current_messages.pop()
    if not current_messages or current_messages[-1]['role'] != 'user':
        current_messages.append({"role": "assistant", "content": "Could not regenerate."})
        state.messages_history = current_messages
        return

    prompt_to_regenerate = current_messages[-1]['content']
    history_for_regen = format_chat_history(current_messages[:-1])
    response_generator = get_agent_response(prompt_to_regenerate, chat_history=history_for_regen)

    assistant_response_content = ""
    current_messages.append({"role": "assistant", "content": "Regenerating..."})
    state.messages_history = current_messages

    for chunk in response_generator:
        assistant_response_content += chunk
        temp_messages = list(state.messages_history)
        if temp_messages:
            temp_messages[-1]["content"] = assistant_response_content
            state.messages_history = temp_messages
        time.sleep(0.02)

    if state.long_term_memory_enabled_ui and _APP_CONTEXT.get("user_id") and state.current_chat_id_ui:
        save_chat_history(state, _APP_CONTEXT["user_id"], state.current_chat_id_ui, state.messages_history)
    messages_tuple = tuple((msg["role"], msg["content"]) for msg in state.messages_history)
    state.suggested_prompts_ui = _cached_generate_suggested_prompts(messages_tuple)

def forget_me_and_reset_taipy(state: State):
    user_id_to_delete = _APP_CONTEXT.get("user_id")
    hf_token = os.getenv("HF_TOKEN")
    if user_id_to_delete and hf_token:
        try:
            # ... (HF file deletion logic remains similar) ...
            metadata_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{user_id_to_delete}_metadata.json"
            messages_hf_path = f"datasets/{HF_USER_MEMORIES_DATASET_ID}/user_memories/{user_id_to_delete}_messages.json"
            try: fs.rm(metadata_hf_path, token=hf_token)
            except Exception: pass
            try: fs.rm(messages_hf_path, token=hf_token)
            except Exception: pass
        except Exception as e:
            notify(state, "error", f"Failed to delete cloud data: {e}")

    if hasattr(state, 'eval_js'):
        state.eval_js("document.cookie = 'user_id=;path=/;expires=Thu, 01 Jan 1970 00:00:00 GMT';")
        state.eval_js("document.cookie = 'long_term_memory_pref=;path=/;expires=Thu, 01 Jan 1970 00:00:00 GMT';")
        state.eval_js("window.location.reload();")

    # Reset _APP_CONTEXT for a clean state if page doesn't fully reload or JS fails
    _APP_CONTEXT.update({
        "user_id": None, "chat_metadata": {}, "all_chat_messages": {},
        "uploaded_documents_content": {}, "uploaded_dataframes_content": {},
        "session_control_flags_initialized": False,
         "initial_greeting_shown_for_session": False,
    })
    # Also reset relevant Taipy state variables to their defaults
    # This is important if the JS reload doesn't happen or isn't immediate
    ui_defaults = ui.state_vars.copy()
    for key in ["messages_history", "current_chat_id_ui", "chat_metadata_ui", "chat_history_lov", "uploaded_files_display", "file_upload_status"]:
        setattr(state, key, ui_defaults[key])
    notify(state, "info", "All data and cookies have been requested for deletion. The page will reload.")


def set_long_term_memory_preference_taipy(state: State, new_ltm_value: bool):
    _APP_CONTEXT["long_term_memory_enabled_pref"] = new_ltm_value
    _APP_CONTEXT["_last_memory_state_was_enabled"] = not new_ltm_value

    try:
        if hasattr(state, 'eval_js'):
            state.eval_js(f"document.cookie = 'long_term_memory_pref={str(new_ltm_value).lower()};path=/;max-age=31536000';")
            print(f"LTM preference saved to cookie (via JS): {new_ltm_value}")
            notify(state, "info", f"LTM preference set to {new_ltm_value}. Page will refresh to apply.")
            _APP_CONTEXT["session_control_flags_initialized"] = False
            state.eval_js("window.location.reload();")
        else:
            notify(state, "warning", "Could not save LTM preference to cookie (eval_js not available).")
    except Exception as e:
        print(f"ERROR: Failed to save LTM preference to cookie: {e}")
        notify(state, "error", f"Failed to save preference: {e}")

def get_ui_accessible_workspace_path_for_ui() -> str:
    return UI_ACCESSIBLE_WORKSPACE

def get_project_root_for_ui() -> str:
    return PROJECT_ROOT

def on_taipy_init(state: State):
    """
    Called when a new Taipy client session starts.
    Attempt to load user ID and LTM preference from cookies (simulated via JS call if possible).
    Then initialize user session data.
    """
    print("Taipy on_init called. Initializing session...")
    # This is where we'd ideally get cookie values via JS and then call initialize_user_session_data_taipy
    # For now, initialize_user_session_data_taipy uses defaults or state-persisted user_id.
    # A more robust way would be needed for true cross-session LTM via cookies.
    
    # Let's assume LTM preference from ui.py's initial state is the default/current.
    # And user_id is also from ui.py's initial state (None, then generated).
    initialize_user_session_data_taipy(state)

    # Populate initial chat if needed (e.g. if no chats and LTM on, or default greeting)
    if not state.messages_history: # If message history is empty after init
        if not _APP_CONTEXT.get("initial_greeting_shown_for_session"):
            create_new_chat_session_taipy(state) # This will set initial greeting
            _APP_CONTEXT["initial_greeting_shown_for_session"] = True
    
    # Ensure agent tools have correct search result count from settings
    if _TAIPY_AGENT_INSTANCE and hasattr(_TAIPY_AGENT_INSTANCE, 'llm') and hasattr(_TAIPY_AGENT_INSTANCE.llm, 'context_window'): # rough check
         # Re-init agent if search result count might have changed and needs to be passed to tools
         # This is a bit heavy-handed. Better if tools could dynamically get this.
         # For now, we assume setup_agent_instance reads from _APP_CONTEXT which is updated by set_llm_settings_taipy
         pass


def main_taipy():
    success, error_message = setup_global_llm_settings()
    if not success:
        print(f"FATAL ERROR: {error_message}")
        return

    setup_agent_instance()

    initial_taipy_state = ui.state_vars.copy()
    initial_ltm_enabled = _APP_CONTEXT.get("long_term_memory_enabled_pref", True)
    initial_taipy_state["long_term_memory_enabled_ui"] = initial_ltm_enabled

    # This part is a bit tricky - user_id and chat data loading depends on LTM pref
    # which might itself come from a cookie (async via JS).
    # For initial load, we use defaults then on_init can refine.
    _APP_CONTEXT["user_id"] = str(uuid.uuid4()) # Placeholder, on_init will set properly
    if initial_ltm_enabled:
        user_data = _load_user_data_from_hf(_APP_CONTEXT["user_id"]) # This uses placeholder user_id first time
        initial_taipy_state["chat_metadata_ui"] = user_data["metadata"]
        initial_taipy_state["chat_history_lov"] = list(user_data["metadata"].items())
        current_chat_id_to_load = next(iter(user_data["metadata"])) if user_data["metadata"] else None
        if current_chat_id_to_load:
            initial_taipy_state["current_chat_id_ui"] = current_chat_id_to_load
            initial_taipy_state["messages_history"] = user_data["messages"].get(current_chat_id_to_load, [])
        else:
            initial_taipy_state["messages_history"] = [{"role": "assistant", "content": _get_initial_greeting_text()}]
    else:
        initial_taipy_state["messages_history"] = [{"role": "assistant", "content": _get_initial_greeting_text()}]

    messages_tuple = tuple((msg["role"], msg["content"]) for msg in initial_taipy_state["messages_history"])
    initial_taipy_state["suggested_prompts_ui"] = _cached_generate_suggested_prompts(messages_tuple if messages_tuple else tuple())
    initial_taipy_state["uploaded_files_display"] = []
    initial_taipy_state["file_upload_status"] = ""
    initial_taipy_state["user_id"] = _APP_CONTEXT["user_id"] # Pass the generated/loaded user_id

    app_callbacks_for_ui = {
        "new_chat_callback": create_new_chat_session_taipy,
        "switch_chat_callback": switch_chat_taipy,
        "delete_chat_callback": delete_chat_session_taipy,
        "rename_chat_callback": rename_chat_taipy,
        "get_discussion_markdown_callback": get_discussion_markdown,
        "get_discussion_docx_callback": get_discussion_docx,
        "process_uploaded_file_callback": process_uploaded_file_taipy,
        "remove_uploaded_file_callback": remove_uploaded_file_taipy,
        "set_llm_settings_callback": set_llm_settings_taipy,
        "set_long_term_memory_callback": set_long_term_memory_preference_taipy,
        "forget_me_callback": forget_me_and_reset_taipy,
        "handle_user_input_callback": handle_user_input_taipy,
        "regenerate_callback": handle_regeneration_request_taipy,
        "get_ui_accessible_workspace_path": get_ui_accessible_workspace_path_for_ui,
        "get_project_root": get_project_root_for_ui,
    }

    gui_instance = ui.init_ui(app_callbacks_for_ui, initial_taipy_state)
    gui_instance.on_init = on_taipy_init # Register on_init function

    os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)

    gui_instance.run(title="ESI Scholarly Instructor (Taipy)",
                     dark_mode=False,
                     use_reloader=True,
                     port=5005,
                     host="0.0.0.0")


def process_uploaded_file_taipy(state: State, file_name: str, file_content_bytes: bytes):
    print(f"Processing uploaded file (Taipy): {file_name}")
    state.file_upload_status = f"Processing '{file_name}'..."
    assistant_message = None
    file_extension = os.path.splitext(file_name)[1].lower()

    try:
        os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)
        file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
        with open(file_path_in_workspace, "wb") as f:
            f.write(file_content_bytes)
        notify(state, "success", f"File '{file_name}' saved to workspace.")
    except Exception as e:
        notify(state, "error", f"Error saving file '{file_name}' to workspace: {e}")
        state.file_upload_status = f"Error saving '{file_name}'."
        return

    processed_type = None
    if file_extension in [".pdf", ".docx", ".md", ".txt"]:
        text_content = ""
        try:
            if file_extension == ".pdf":
                reader = PdfReader(BytesIO(file_content_bytes))
                for page_num, page in enumerate(reader.pages):
                    extracted_page_text = page.extract_text()
                    if extracted_page_text: text_content += extracted_page_text + "\n"
                    else: print(f"Warning: Could not extract text from page {page_num + 1} of PDF '{file_name}'.")
            elif file_extension == ".docx":
                from docx import Document # Local import for this specific use
                document = Document(BytesIO(file_content_bytes))
                for para in document.paragraphs: text_content += para.text + "\n"
            elif file_extension in [".md", ".txt"]:
                text_content = file_content_bytes.decode("utf-8", errors='replace') # Added error handling for decode

            _APP_CONTEXT["uploaded_documents_content"][file_name] = text_content
            processed_type = "document"
            notify(state, "success", f"Document '{file_name}' processed for agent access.")
            assistant_message = f"I've received your document: `{file_name}`. You can now ask me to `read_uploaded_document('{file_name}')`."
        except Exception as e:
            notify(state, "error", f"Error extracting content from '{file_name}': {e}")
            state.file_upload_status = f"Error extracting content from '{file_name}'."
            if os.path.exists(file_path_in_workspace):
                try: os.remove(file_path_in_workspace)
                except Exception as del_e: print(f"Error deleting partially processed file {file_path_in_workspace}: {del_e}")
            return
    elif file_extension in [".csv", ".xlsx", ".sav"]:
        try:
            df = None
            if file_extension == ".csv":
                df = pd.read_csv(BytesIO(file_content_bytes))
            elif file_extension == ".xlsx":
                df = pd.read_excel(BytesIO(file_content_bytes))
            elif file_extension == ".sav":
                try:
                    import pyreadstat
                    df, _ = pyreadstat.read_sav(file_path_in_workspace)
                except ImportError:
                    notify(state, "error", "`pyreadstat` needed for .sav files. Please install it.")
                    state.file_upload_status = "Failed to read .sav: pyreadstat missing."
                    return
                except Exception as e_sav:
                    notify(state, "error", f"Error reading .sav file '{file_name}': {e_sav}")
                    state.file_upload_status = f"Error reading .sav file '{file_name}'."
                    return
            if df is not None:
                _APP_CONTEXT["uploaded_dataframes_content"][file_name] = df
                processed_type = "dataframe"
                notify(state, "success", f"Dataset '{file_name}' processed.")
                assistant_message = f"I've received dataset: `{file_name}`. Ask me to `analyze_uploaded_dataframe('{file_name}')`."
            else:
                notify(state, "warning", f"Could not load dataframe from '{file_name}'.")
                state.file_upload_status = f"Could not load dataframe from '{file_name}'."
                return
        except Exception as e:
            notify(state, "error", f"Error processing dataset '{file_name}': {e}")
            state.file_upload_status = f"Error processing '{file_name}'."
            return
    else:
        notify(state, "warning", f"Unsupported file type: {file_extension}. File saved, but not processed for agent.")
        state.file_upload_status = f"Unsupported file type: {file_extension} for '{file_name}'."
        return

    if processed_type: # Only update UI list if successfully processed for agent
        new_file_entry = {
            "id": file_name, "name": file_name,
            "type": "doc" if processed_type == "document" else "df",
            "icon": "📄" if processed_type == "document" else "📊",
            "actions": f"<|button|on_action=on_uploaded_file_table_action|label=Delete|class_name=taipy-error|file_name={file_name}|file_type={processed_type}|>"
        }
        current_uploaded_files = list(state.uploaded_files_display) # Make a copy for Taipy
        current_uploaded_files.append(new_file_entry)
        state.uploaded_files_display = current_uploaded_files
    
    if assistant_message:
        current_messages = list(state.messages_history)
        current_messages.append({"role": "assistant", "content": assistant_message})
        state.messages_history = current_messages
        if state.current_chat_id_ui:
            _APP_CONTEXT["all_chat_messages"][state.current_chat_id_ui] = list(state.messages_history)
    state.file_upload_status = f"'{file_name}' processed."


def remove_uploaded_file_taipy(state: State, file_name: str, file_type: str):
    current_uploaded_files = [f for f in state.uploaded_files_display if f["name"] != file_name]
    state.uploaded_files_display = current_uploaded_files

    if file_type == "doc" or file_type == "document":
        if file_name in _APP_CONTEXT["uploaded_documents_content"]:
            del _APP_CONTEXT["uploaded_documents_content"][file_name]
    elif file_type == "df" or file_type == "dataframe":
        if file_name in _APP_CONTEXT["uploaded_dataframes_content"]:
            del _APP_CONTEXT["uploaded_dataframes_content"][file_name]

    file_path_in_workspace = os.path.join(UI_ACCESSIBLE_WORKSPACE, file_name)
    if os.path.exists(file_path_in_workspace):
        try:
            os.remove(file_path_in_workspace)
        except Exception as e:
            print(f"Error deleting physical file '{file_path_in_workspace}': {e}")
            notify(state, "error", f"Error deleting physical file '{file_name}': {e}")
    notify(state, "info", f"File '{file_name}' removed.")


def set_llm_settings_taipy(state: State, settings: Dict[str, Any]):
    _APP_CONTEXT["llm_settings"].update(settings) # Update specific settings
    if Settings.llm and hasattr(Settings.llm, 'temperature'):
        Settings.llm.temperature = _APP_CONTEXT["llm_settings"].get("temperature", 0.7)

    # Update agent's max_search_results if it changed - this might require agent re-initialization or a setter.
    # For now, new agent instances created via setup_agent_instance will pick up the new search_results_count.
    # If the agent is already live, this setting change won't affect it unless explicitly handled.
    # Consider re-calling setup_agent_instance if search_results_count changes,
    # or make agent tools read this dynamically from _APP_CONTEXT.
    if "search_results_count" in settings:
         print(f"Search results count changed to: {settings['search_results_count']}. Agent may need re-init for this to take full effect on some tools.")
         # Potentially re-initialize agent if this setting is critical for live agent
         # setup_agent_instance() # This would re-init with current settings
         pass


    print(f"LLM settings updated in app.py: {_APP_CONTEXT['llm_settings']}")
    notify(state, "info", "LLM settings updated.")


# Make sure PROJECT_ROOT is defined before this line if __name__ == "__main__": is used for execution
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")
    main_taipy()
