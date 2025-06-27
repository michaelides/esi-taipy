import os
import json
import re
import sys
import uuid
from typing import List, Dict, Any, Optional, Callable, Generator, Tuple
from functools import lru_cache
# import pandas as pd # Commented out for minimal version
# from PyPDF2 import PdfReader # Commented out for minimal version
from io import BytesIO
import time
import asyncio

# from llama_index.core.llms import ChatMessage, MessageRole # Commented out
# from llama_index.core import Settings # Commented out
import ui
from taipy.gui import State, Gui, notify
# from agent import create_orchestrator_agent, generate_suggested_prompts, SUGGESTED_PROMPT_COUNT, DEFAULT_PROMPTS, initialize_settings as initialize_agent_settings, generate_llm_greeting # Commented out
from dotenv import load_dotenv
# from llama_index.core.tools import FunctionTool # Commented out

# from huggingface_hub import HfFileSystem # Commented out

# fs = HfFileSystem() # Commented out
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, "workspace_ui") # Defined later if needed

# _TAIPY_AGENT_INSTANCE: Optional[Any] = None
# _TAIPY_STATE_DEPENDANT_TOOLS_INITIALIZED: bool = False

# DOWNLOAD_MARKER = "---DOWNLOAD_FILE---"
# RAG_SOURCE_MARKER_PREFIX = "---RAG_SOURCE---"

# from tools import UI_ACCESSIBLE_WORKSPACE # Defined later if needed
# from config import HF_USER_MEMORIES_DATASET_ID # Commented out

# MAX_CHAT_HISTORY_MESSAGES = 15 # Commented out

_APP_CONTEXT: Dict[str, Any] = {
    "long_term_memory_enabled_pref": True, # Keep for on_taipy_init structure if needed by ui.py
    # Most other app_context keys removed for minimal version
}

# --- All agent, LLM, data processing, chat logic functions commented out for minimal version ---
# setup_global_llm_settings, _get_initial_greeting_text, _cached_generate_suggested_prompts,
# read_uploaded_document_tool_fn, analyze_dataframe_tool_fn, setup_agent_instance,
# get_user_id_and_cookie_pref_from_state, initialize_user_session_data_taipy (partially kept),
# _load_user_data_from_hf, save_chat_history, save_chat_metadata, format_chat_history,
# get_agent_response, create_new_chat_session_taipy, switch_chat_taipy,
# delete_chat_session_taipy, rename_chat_taipy, get_discussion_markdown, get_discussion_docx,
# handle_user_input_taipy, reset_chat_taipy, handle_regeneration_request_taipy,
# forget_me_and_reset_taipy, set_long_term_memory_preference_taipy,
# process_uploaded_file_taipy, remove_uploaded_file_taipy, set_llm_settings_taipy
# get_ui_accessible_workspace_path_for_ui, get_project_root_for_ui.

# Simplified initialize_user_session_data_taipy for minimal test
def initialize_user_session_data_taipy(state: State) -> None:
    print("Minimal initialize_user_session_data_taipy called.")
    # We expect 'simple_message' to be set via globals in gui.run()
    # If 'long_term_memory_enabled_ui' is needed by ui.py's structure (even if not used), ensure it exists.
    if not hasattr(state, 'long_term_memory_enabled_ui'): # Should be provided by initial_taipy_state
        print("DEBUG: minimal on_init: long_term_memory_enabled_ui not on state, setting default.")
        # This direct assignment might fail if not properly declared, but globals should handle it.
        # Forcing it here for the sake of allowing ui.py's current structure to pass if it expects it.
        # state.long_term_memory_enabled_ui = _APP_CONTEXT.get("long_term_memory_enabled_pref", True)
        pass # Let globals handle it. Accessing it in on_taipy_init might be the issue.

def on_taipy_init(state: State):
    print("Minimal Taipy on_init called. Session initializing. (No state access in this version)")
    # initialize_user_session_data_taipy(state) # Further simplify: comment out for now
    # Check if simple_message is accessible - REMOVED for this test
    # try:
    #     print(f"Minimal on_init: state.simple_message = {state.simple_message}")
    # except Exception as e:
    #     print(f"Minimal on_init: Error accessing state.simple_message: {e}")

    # If ui.py still has long_term_memory_enabled_ui in its state_vars for some reason,
    # ensure it's on the state to avoid errors if Taipy tries to bind it.
    # This should be set via the `globals` argument in `gui.run`.
    # if not hasattr(state, 'long_term_memory_enabled_ui'): # REMOVED for this test
    #     print("Minimal on_init: 'long_term_memory_enabled_ui' still not found. This is unexpected if initial_taipy_state included it for globals.")



def main_taipy():
    print("Minimal main_taipy started.")
    # success, error_message = setup_global_llm_settings() # Commented out
    # if not success:
    #     print(f"FATAL ERROR: {error_message}")
    #     return
    # setup_agent_instance() # Commented out

    # Minimal state, must match what minimal ui.py expects in its state_vars and page
    initial_taipy_state = {
        "simple_message": "Hello from Minimal Taipy APP!",
        # Ensure all keys defined in ui.state_vars are present here if ui.init_ui expects them
        # even if ui.py is now minimal. The ui.state_vars is the source for ui.init_ui's expectation.
        "long_term_memory_enabled_ui": _APP_CONTEXT.get("long_term_memory_enabled_pref", True) # from original ui.state_vars
        # Add other keys from original ui.state_vars with default/dummy values if ui.init_ui crashes without them
        # For a truly minimal test, ui.py's state_vars should also be minimal.
        # Current minimal ui.py only has "simple_message".
        # However, if ui.init_ui uses the full ui.state_vars to populate initial_state_from_app for add_shared_variables,
        # then all those keys must be present in this initial_taipy_state.
        # Let's assume the current minimal ui.py's state_vars has only "simple_message".
        # The ui.py I wrote has: state_vars = { "simple_message": "..." }
        # So, this should be fine.
    }
    
    # Minimal callbacks
    app_callbacks_for_ui = {}

    gui_instance = ui.init_ui(app_callbacks_for_ui, initial_taipy_state)
    gui_instance.on_init = on_taipy_init

    # UI_ACCESSIBLE_WORKSPACE definition might be needed if ui.py tries to use it, even if commented out parts
    # For minimal, assume it's not immediately needed.
    # global UI_ACCESSIBLE_WORKSPACE
    # UI_ACCESSIBLE_WORKSPACE = os.path.join(PROJECT_ROOT, "workspace_ui")
    # os.makedirs(UI_ACCESSIBLE_WORKSPACE, exist_ok=True)

    print("Minimal setup complete, running GUI...")
    gui_instance.run(
        title="ESI Minimal (Taipy)",
        dark_mode=False,
        use_reloader=True, # Keep reloader to see if it's part of the issue
        port=5005,
        host="0.0.0.0",
        globals=initial_taipy_state # Provide initial state to Taipy
    )

if __name__ == "__main__":
    # if not os.getenv("GOOGLE_API_KEY"): # Commented out
    #     print("⚠️ GOOGLE_API_KEY environment variable not set. The agent may not work properly.")
    main_taipy()
