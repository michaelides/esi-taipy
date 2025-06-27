import os
import re
import json
from typing import List, Dict, Any, Optional, Callable
import html
import pandas as pd
# from PyPDF2 import PdfReader # Will be handled by app.py or agent
# from docx import Document # Will be handled by app.py or agent
import io
import inspect # For get_module_name_from_frame if needed, or for explicit callable registration
import sys

from taipy.gui import Gui, State, Markdown, Page, get_module_name_from_state, notify
import taipy.gui.builder as tgb

# Assuming PROJECT_ROOT and UI_ACCESSIBLE_WORKSPACE might be needed for file paths
# These would ideally be passed from app.py or configured globally
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) # This will be ui.py's dir
# UI_ACCESSIBLE_WORKSPACE should be managed by app.py and paths passed to Taipy as needed for file downloads/uploads.

# --- Global object to hold app_callbacks from app.py ---
# This is a simple way to make them accessible to Taipy callbacks defined in this module.
_app_callbacks: Dict[str, Callable] = {}

# --- Taipy State Variables ---
# These mirror the Streamlit session_state variables relevant to the UI.
# Their initial values will be set in `create_gui` and then managed by Taipy.
state_vars = {
    "messages_history": [],  # List of {"role": "user/assistant", "content": "..."}
    "current_chat_id_ui": None,
    "chat_metadata_ui": {}, # Dict of {chat_id: chat_name}
    "chat_history_lov": [], # List of (chat_id, chat_name) for menu
    "editing_chat_id_ui": None, # For renaming chats
    "renaming_chat_input_ui": "",

    "uploaded_files_display": [], # List of {"id": file_name, "name": file_name, "type": "doc/df", "icon": "📄/📊"}
    "file_upload_status": "",

    "llm_temperature_ui": 0.7,
    "llm_verbosity_ui": 3,
    "search_results_count_ui": 5,
    "long_term_memory_enabled_ui": False,

    "suggested_prompts_ui": ["Suggest a research question", "Help me structure my literature review", "Explain qualitative coding"],
    "chat_input_value_ui": "",

    "show_forget_me_dialog": False,
    "show_rename_dialog_ui": False,
    "chat_id_to_rename_ui": "",
    "new_chat_name_ui": ""
}


# --- UI Page Definition (Markdown) ---
# Using Taipy Markdown to define the layout.
# Parts can be used for collapsible sections like Streamlit's expander.
# The `chat` control is central for message display.
# Other controls like `menu`, `file_selector`, `slider`, `input` map to Streamlit equivalents.

main_page_md = """
<|toggle|theme|>
<|layout|columns=300px 1fr|gap=10px|
    <|part|class_name=sidebar|
        <|expandable|title=Chat History|expanded=False|
            <|button|on_action=new_chat_action|>+ New Chat</|button|>
            <|menu|lov={chat_history_lov}|adapter=chat_history_adapter|on_action=on_chat_menu_action|value={current_chat_id_ui}|width=100%|>
        |>
        <|expandable|title=Upload Files|expanded=False|
            <|file_selector|on_action=handle_file_upload_action|extensions=.pdf,.docx,.md,.txt,.csv,.xlsx,.sav,.rdata,.rds|label=Upload Document/Dataset|multiple=False|drop_message=Drop file here|>
            <br/>
            <|text|{file_upload_status}|raw=True|>
            **Uploaded Files:**
            <|table|data={uploaded_files_display}|columns=name:Name;type:Type;actions:Actions|editable=False|on_action=on_uploaded_file_table_action|rebuild|>
        |>
        <|expandable|title=LLM Settings|expanded=False|
            LLM Temperature: <|text|{llm_temperature_ui:.1f}|>
            <|slider|value={llm_temperature_ui}|min=0.0|max=2.0|step=0.1|on_change=on_llm_setting_change|propagate=False|>
            LLM Verbosity: <|text|{llm_verbosity_ui}|>
            <|slider|value={llm_verbosity_ui}|min=1|max=5|step=1|on_change=on_llm_setting_change|propagate=False|>
            Search Results: <|text|{search_results_count_ui}|>
            <|slider|value={search_results_count_ui}|min=3|max=15|step=1|on_change=on_llm_setting_change|propagate=False|>
            <|toggle|value={long_term_memory_enabled_ui}|label=Long-term Memory|on_change=on_llm_setting_change|propagate=False|>
            <|button|on_action=open_forget_me_dialog|class_name=taipy-error|>Forget Me (Delete All Data)</|button|>
        |>
        <|expandable|title=About ESI|expanded=False|
            <|text|ESI: ESI Scholarly Instructor - Your AI partner for dissertation research.|mode=md|>
            <|text|⚠️ Always consult your supervisor for final guidance and decisions.|mode=md|>
            <|text|Made for NBS7091A and NBS7095x|mode=md|>
        |>
    |>

    <|part|class_name=chat_area|
        <|navbar|>
        # 🎓 ESI: ESI Scholarly Instructor
        <|text|Your AI partner for brainstorming and structuring your dissertation research|mode=md|>

        <div class="chat-container" style="height: 60vh; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom:10px;">
        <|part|render={len(messages_history) > 0}|
            <|repeater|data={messages_history}|
                <|layout|columns=auto 1fr|class_name={class_name_for_message(item)}|
                    <|text|{get_avatar_for_message(item)}|class_name=avatar|>
                    <|text|{get_content_for_message(item)}|mode=md|class_name=message_content|>
                |>
                <|part|render={item.role == 'assistant' and has_download_or_image(item)}|
                    <|layout|columns=auto 1fr|
                        <|html|{get_download_or_image_html(state, item)}|>
                    |>
                |>
                 <|part|render={item.role == 'assistant' and has_rag_sources(item)}|
                    <|text|**Sources:**|mode=md|>
                    <|html|{get_rag_sources_html(item)}|>
                |>
                 <|part|render={item.role == 'assistant' and is_last_assistant_message(state, item, index)}|
                     <|button|icon=content_copy|on_action=copy_message_action|label=Copy|class_name=small_button|index={index}|> <|button|icon=refresh|on_action=regenerate_response_action|label=Regenerate|class_name=small_button|index={index}|>
                |>
            |>
        |>
        </div>

        <|layout|columns=1fr auto|gap=5px|
            <|input|value={chat_input_value_ui}|label=Ask me anything...|on_action=on_chat_input_action|class_name=chat_input|change_delay=-1|width=100%|>
            <|button|on_action=on_chat_input_action|label=Send|icon=send|>
        |>
        <br/>
        **Suggested Prompts:**
        <|layout|columns=1 1 1|gap=10px|class_name=suggested_prompts_layout|
            <|button|on_action=on_suggested_prompt_action|label={suggested_prompts_ui[0] if len(suggested_prompts_ui) > 0 else "Prompt 1"}|width=100%|class_name=prompt_button|>
            <|button|on_action=on_suggested_prompt_action|label={suggested_prompts_ui[1] if len(suggested_prompts_ui) > 1 else "Prompt 2"}|width=100%|class_name=prompt_button|>
            <|button|on_action=on_suggested_prompt_action|label={suggested_prompts_ui[2] if len(suggested_prompts_ui) > 2 else "Prompt 3"}|width=100%|class_name=prompt_button|>
        |>
    |>
|>

<|dialog|open={show_forget_me_dialog}|title=Confirm Deletion|on_action=dialog_action_forget_me|width=400px|
    <p>This will permanently delete ALL your saved chat histories and remove your user ID cookie from this browser. This action cannot be undone.</p>
    <p><strong>Are you sure you want to proceed?</strong></p>
    <|layout|columns=1 1|
        <|button|on_action=dialog_action_forget_me|label=Yes, Delete All Data|class_name=taipy-error|>
        <|button|on_action=dialog_action_forget_me|label=No, Cancel|>
    |>
|>

<|dialog|open={show_rename_dialog_ui}|title=Rename Chat|on_action=dialog_action_rename_chat|width=300px|
    <|input|value={new_chat_name_ui}|label=New chat name:|>
    <|layout|columns=1 1|
        <|button|on_action=dialog_action_rename_chat|label=Rename|>
        <|button|on_action=dialog_action_rename_chat|label=Cancel|>
    |>
|>
"""

# --- Helper functions for dynamic UI content ---
def class_name_for_message(item: Dict[str, str]) -> str:
    return "user_message_layout" if item.get("role") == "user" else "assistant_message_layout"

def get_avatar_for_message(item: Dict[str, str]) -> str:
    return "👤" if item.get("role") == "user" else "🎓" # Simple text avatars

def get_content_for_message(item: Dict[str, str]) -> str:
    # Basic Markdown rendering for now. Taipy's <|chat|> might handle this better if used directly.
    # Need to escape HTML if content is directly from user/LLM to prevent injection.
    # For now, assuming content is safe or will be sanitized before adding to messages_history.
    content = item.get("content", "")
    # Basic handling for code blocks, Taipy Markdown should render them.
    return content

def has_download_or_image(item: Dict[str, str]) -> bool:
    content = item.get("content", "")
    return "---DOWNLOAD_FILE---" in content or ".png" in content or ".jpg" in content # Simplified check

def get_download_or_image_html(state: State, item: Dict[str, str]) -> str:
    # This needs to parse the content and generate appropriate Taipy controls or HTML.
    # For downloads, a <|file_download|> control. For images, <|image|>.
    # This will require access to the file paths, potentially via UI_ACCESSIBLE_WORKSPACE.
    # This function will generate Taipy Markdown/HTML for downloads or images.
    # It needs access to UI_ACCESSIBLE_WORKSPACE, which should be configured in app.py
    # and potentially passed to Taipy's state or context if files are served by Taipy.

    # For now, assume UI_ACCESSIBLE_WORKSPACE is available globally or via state.
    # This might need adjustment based on how app.py serves files.
    # ui_accessible_workspace = state.ui_accessible_workspace if hasattr(state, "ui_accessible_workspace") else "workspace_ui"
    ui_accessible_workspace = _app_callbacks.get("get_ui_accessible_workspace_path", lambda: "workspace_ui")()


    html_content = ""
    text_to_display = item.get("content", "")

    # --- Extract Code Interpreter download marker ---
    # Regex for "---DOWNLOAD_FILE---filename.ext"
    code_marker_pattern = re.compile(r"---DOWNLOAD_FILE---([^\n]+)", re.IGNORECASE)

    for match in code_marker_pattern.finditer(text_to_display):
        extracted_filename = match.group(1).strip()
        # Construct the relative path for Taipy's file serving if needed, or absolute if served by external means.
        # Assuming files are in UI_ACCESSIBLE_WORKSPACE relative to where Taipy serves static files from.
        # If Taipy serves from project root, this path needs to be relative to that.
        # Or, app.py can expose a download endpoint.

        # Simplification: Assume app.py will make these files available via a route like /files/filename.ext
        # and Taipy's <|file_download|> can point to this URL, or serve directly if path is local.

        file_path_for_taipy = os.path.join(ui_accessible_workspace, extracted_filename) # Relative path to workspace

        # Check if the file exists (important for robustness)
        # This check should ideally use an absolute path based on PROJECT_ROOT
        # project_root_path = _app_callbacks.get("get_project_root", lambda: ".")()
        # absolute_file_path = os.path.join(project_root_path, file_path_for_taipy)
        # For now, we assume app.py has placed it correctly and it's findable by Taipy.

        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        if os.path.splitext(extracted_filename)[1].lower() in image_extensions:
            # Use <|image|> control. Content can be a path or URL.
            # If Taipy serves static files from a dir, this path should be relative to it.
            # Or, it can be a URL if app.py serves it via a route.
            # Assuming path relative to a configured static folder for Taipy.
            # For local dev, Taipy might serve from project root by default for relative paths.
            # Let's assume `file_path_for_taipy` can be resolved by Taipy's image control.
            # If files are in `workspace_ui/image.png`, and Taipy serves from project root,
            # then `workspace_ui/image.png` should work.
            html_content += f"<|image|content={file_path_for_taipy}|label={extracted_filename}|width=300px|>\n"
        else:
            # Use <|file_download|> control. Content can be path or URL.
            # Label is what user sees. Name is the downloaded file name.
            html_content += f"<|file_download|content={file_path_for_taipy}|label=Download {extracted_filename}|name={extracted_filename}|>\n"

    return html_content


def has_rag_sources(item: Dict[str, str]) -> bool:
    content = item.get("content", "")
    return "---RAG_SOURCE---" in content

def get_rag_sources_html(item: Dict[str, str]) -> str:
    content = item.get("content", "")
    rag_source_pattern = re.compile(r"---RAG_SOURCE---({.*?})", re.DOTALL)
    sources_html = ""

    for match in rag_source_pattern.finditer(content):
        json_str = match.group(1)
        try:
            rag_data = json.loads(json_str)
            source_type = rag_data.get("type")

            if source_type == "pdf":
                pdf_name = rag_data.get("name", "source.pdf")
                pdf_source_path = rag_data.get("path") # http:// or file://
                citation_num = rag_data.get('citation_number')
                citation_prefix = f"[{citation_num}] " if citation_num else ""

                if pdf_source_path and pdf_source_path.startswith("http"):
                    sources_html += f'<div>{citation_prefix}<a href="{pdf_source_path}" target="_blank" title="{pdf_source_path}">📄 {pdf_name} (Web PDF)</a></div>'
                elif pdf_source_path and pdf_source_path.startswith("file://"):
                    local_file_path_from_rag = pdf_source_path[len("file://"):]
                    # This local_file_path_from_rag is an *absolute* path on the server where the agent ran.
                    # To make it downloadable via Taipy, it needs to be:
                    # 1. Copied to a Taipy-accessible workspace (e.g., UI_ACCESSIBLE_WORKSPACE)
                    # 2. Or app.py needs to provide a download endpoint for arbitrary server files (security risk).

                    # Assuming app.py copies these RAG PDFs to UI_ACCESSIBLE_WORKSPACE when they are generated by the agent.
                    # The name of the file in the workspace should be predictable, e.g., based on `pdf_name` or a hash.
                    # For simplicity, let's assume the `pdf_name` is the filename in the workspace.
                    ui_accessible_workspace = _app_callbacks.get("get_ui_accessible_workspace_path", lambda: "workspace_ui")()
                    file_path_for_taipy = os.path.join(ui_accessible_workspace, pdf_name)

                    # Here, Taipy's file_download expects a path relative to its serving context, or an absolute path
                    # if it has access, or a URL.
                    # We'll use the relative path within the workspace.
                    sources_html += f'<div>{citation_prefix}<|file_download|content={file_path_for_taipy}|label=📄 {pdf_name} (Local PDF)|name={pdf_name}|></div>'
                else:
                    sources_html += f'<div>{citation_prefix}📄 {pdf_name} (Path format not supported: {pdf_source_path})</div>'

            elif source_type == "web":
                url = rag_data.get("url")
                title = rag_data.get("title", url)
                if url:
                    sources_html += f'<div><a href="{url}" target="_blank" title="{url}">🌐 {title}</a></div>'

        except json.JSONDecodeError:
            sources_html += "<div>Error parsing RAG source JSON.</div>"
        except Exception as e:
            sources_html += f"<div>Error processing RAG source: {e}</div>"

    return sources_html if sources_html else ""


def is_last_assistant_message(state: State, item: Dict[str, str], index: int)-> bool:
    # Check if it's the last message in the messages_history list
    # and if it's an assistant message.
    # The `index` is provided by Taipy's repeater.
    return item.get("role") == "assistant" and index == len(state.messages_history) - 1

# --- Adapter for Chat History Menu ---
def chat_history_adapter(item: Any) -> List[Any]: # type: ignore
    """Adapts chat metadata for the Taipy menu control.
    item is expected to be a tuple (chat_id, chat_name).
    The menu expects a list of (id, label, icon, {optional_popup_menu_items}).
    """
    if not isinstance(item, tuple) or len(item) != 2:
        return ["", "Invalid item", ""] # id, label, icon

    chat_id, chat_name = item

    # Define actions for the popup menu (three dots)
    # Each action is (label, callback_function_name_as_string, icon_name)
    # The callback function (e.g., 'handle_chat_rename_request') will receive chat_id via its arguments.
    popup_actions = [
        ("Rename", "request_rename_chat_action", "Edit"), # Icon names might need adjustment for Taipy
        ("Delete", "request_delete_chat_action", "Delete"),
        ("Download MD", "download_chat_md_action", "Download"),
        ("Download DOCX", "download_chat_docx_action", "Download")
    ]
    return [chat_id, chat_name, "chat", popup_actions]


# --- Callback Functions ---
# These will call functions passed from app.py via `_app_callbacks`.

def on_chat_menu_action(state: State, id: str, action: str, payload: Dict):
    """Handles actions from the chat history menu (select, or popup actions)."""
    action_details = payload.get("action") # This is how menu sends popup action details

    if action_details: # A popup menu item was clicked
        chat_id_affected = id # The menu item's id is the chat_id
        action_label = action_details # e.g., "Rename", "Delete"

        if action_label == "Rename":
            request_rename_chat_action(state, chat_id_affected)
        elif action_label == "Delete":
            request_delete_chat_action(state, chat_id_affected)
        elif action_label == "Download MD":
            download_chat_md_action(state, chat_id_affected)
        elif action_label == "Download DOCX":
            download_chat_docx_action(state, chat_id_affected)
        else:
            print(f"Unknown chat menu popup action: {action_label} for chat {chat_id_affected}")
    else: # A direct selection of a chat from the menu
        selected_chat_id = id # The id of the menu item is the chat_id
        if selected_chat_id != state.current_chat_id_ui:
            print(f"Switching to chat: {selected_chat_id}")
            if 'switch_chat_callback' in _app_callbacks:
                _app_callbacks['switch_chat_callback'](state, selected_chat_id)
            # App.py callback should update state.messages_history, state.current_chat_id_ui etc.

def request_rename_chat_action(state: State, chat_id: str):
    """Opens a dialog to rename a chat."""
    state.chat_id_to_rename_ui = chat_id
    state.new_chat_name_ui = state.chat_metadata_ui.get(chat_id, "")
    state.show_rename_dialog_ui = True

def request_delete_chat_action(state: State, chat_id: str):
    """Handles request to delete a chat (could show confirmation dialog first)."""
    print(f"Request to delete chat: {chat_id}")
    if 'delete_chat_callback' in _app_callbacks:
        # Maybe show a Taipy dialog here for confirmation before calling app.py
        _app_callbacks['delete_chat_callback'](state, chat_id)
    # App.py callback should update chat_history_lov, chat_metadata_ui, and potentially current chat.

def download_chat_md_action(state: State, chat_id: str):
    print(f"Request to download MD for chat: {chat_id}")
    if 'get_discussion_markdown_callback' in _app_callbacks:
        md_content = _app_callbacks['get_discussion_markdown_callback'](chat_id)
        # Use Taipy's file download mechanism
        # Need to make sure the file name is dynamic based on chat_name
        chat_name = state.chat_metadata_ui.get(chat_id, "chat")
        file_name = f"{chat_name.replace(' ', '_')}.md"
        # The `content` for file_download can be bytes or a path to a file.
        # If it's bytes, Taipy handles serving it.
        state.download(content=md_content.encode(), name=file_name)
        notify(state, "info", f"Downloading {file_name}...")

def download_chat_docx_action(state: State, chat_id: str):
    print(f"Request to download DOCX for chat: {chat_id}")
    if 'get_discussion_docx_callback' in _app_callbacks:
        docx_bytes = _app_callbacks['get_discussion_docx_callback'](chat_id)
        chat_name = state.chat_metadata_ui.get(chat_id, "chat")
        file_name = f"{chat_name.replace(' ', '_')}.docx"
        state.download(content=docx_bytes, name=file_name)
        notify(state, "info", f"Downloading {file_name}...")


def new_chat_action(state: State):
    print("New chat action triggered")
    if 'new_chat_callback' in _app_callbacks:
        _app_callbacks['new_chat_callback'](state)
    # App.py callback should update state variables (messages_history, current_chat_id_ui, etc.)

def handle_file_upload_action(state: State, id: str, action: str, payload: Dict):
    # Taipy's file_selector on_action payload contains file content in payload['args'][0]
    # This is different from Streamlit's UploadedFile object.
    # We need to adapt how app.py's process_uploaded_file is called.
    print(f"File upload action: {id}, {action}")
    uploaded_file_content = payload.get('args', [None])[0] # Content as bytes
    original_file_name = getattr(uploaded_file_content, "name", "uploaded_file") # Taipy might provide name via properties or payload.

    if uploaded_file_content:
        state.file_upload_status = f"Processing '{original_file_name}'..."
        if 'process_uploaded_file_callback' in _app_callbacks:
            # We need to pass content and name. App.py's callback might need adjustment
            # if it expects a Streamlit UploadedFile object.
            # For now, assume it can handle bytes and a name.
            _app_callbacks['process_uploaded_file_callback'](state, original_file_name, uploaded_file_content)
            # App.py callback should update state.uploaded_files_display and state.file_upload_status
        else:
            state.file_upload_status = "Error: File processing callback not configured."
    else:
        state.file_upload_status = "File upload failed or no file selected."


def on_uploaded_file_table_action(state: State, id: str, action_id: str, payload: Dict):
    """Handles actions on the uploaded files table (e.g., delete)."""
    row_index = payload.get("index") # Index of the row in uploaded_files_display
    action_name = payload.get("action") # Name of the action column clicked (e.g., "delete")

    if row_index is not None and action_name == "delete": # Assuming 'delete' is the action for the delete button
        if 0 <= row_index < len(state.uploaded_files_display):
            file_to_remove = state.uploaded_files_display[row_index]
            file_name = file_to_remove.get("name")
            file_type = file_to_remove.get("type") # "doc" or "df"
            print(f"Request to remove uploaded file: {file_name}")
            if 'remove_uploaded_file_callback' in _app_callbacks:
                _app_callbacks['remove_uploaded_file_callback'](state, file_name, file_type)
            # App.py callback should update state.uploaded_files_display
    else:
        print(f"Unhandled table action: id={id}, action_id={action_id}, payload={payload}")


def on_llm_setting_change(state: State, var_name: str, value: Any):
    """Handles changes in LLM settings sliders/toggle."""
    # var_name will be like 'llm_temperature_ui', 'llm_verbosity_ui', etc.
    # value is the new value from the control.
    print(f"LLM Setting changed: {var_name} = {value}")

    # Update the specific state variable. Taipy might do this automatically if `propagate=True` (default).
    # If `propagate=False` is used on sliders, we might need to set `state.var_name = value` here,
    # but usually Taipy handles updating the bound variable.
    # The primary purpose here is to notify app.py if needed, or trigger other logic.

    if 'set_llm_settings_callback' in _app_callbacks:
        # Send all settings to app.py, or app.py can read them from state when needed.
        settings = {
            "temperature": state.llm_temperature_ui,
            "verbosity": state.llm_verbosity_ui,
            "search_results_count": state.search_results_count_ui,
            "long_term_memory_enabled": state.long_term_memory_enabled_ui
        }
        # Distinguish which specific setting changed if callback needs that info
        if var_name == "llm_temperature_ui": settings["temperature"] = value
        elif var_name == "llm_verbosity_ui": settings["verbosity"] = value
        elif var_name == "search_results_count_ui": settings["search_results_count"] = value
        elif var_name == "long_term_memory_enabled_ui": settings["long_term_memory_enabled"] = value

        _app_callbacks['set_llm_settings_callback'](state, settings)

    # If long_term_memory_enabled_ui changed, app.py's callback should handle UI refresh/data loading.
    if var_name == "long_term_memory_enabled_ui":
        if 'set_long_term_memory_callback' in _app_callbacks: # Specific callback from stui
             _app_callbacks['set_long_term_memory_callback'](state, value)


def open_forget_me_dialog(state: State):
    state.show_forget_me_dialog = True

def dialog_action_forget_me(state: State, id: str, action: str, payload: Dict):
    """Handles actions from the 'Forget Me' confirmation dialog."""
    button_label = payload.get("args", [{}])[0].get("label", "") # Taipy dialog action sends button label
    if button_label == "Yes, Delete All Data":
        print("Forget Me confirmed.")
        if 'forget_me_callback' in _app_callbacks:
            _app_callbacks['forget_me_callback'](state)
            notify(state, "success", "All your data has been requested for deletion.")
        # App.py's callback should handle cookie deletion and data reset, then potentially trigger a page reload or state reset.
    else:
        print("Forget Me cancelled.")
        notify(state, "info", "Deletion cancelled.")
    state.show_forget_me_dialog = False


def dialog_action_rename_chat(state: State, id: str, action: str, payload: Dict):
    button_label = payload.get("args", [{}])[0].get("label", "")
    if button_label == "Rename":
        chat_id = state.chat_id_to_rename_ui
        new_name = state.new_chat_name_ui
        if chat_id and new_name:
            if 'rename_chat_callback' in _app_callbacks:
                _app_callbacks['rename_chat_callback'](state, chat_id, new_name)
                notify(state, "success", f"Chat renamed to '{new_name}'.")
            else:
                notify(state, "error", "Rename callback not configured.")
        else:
            notify(state, "warning", "Cannot rename: chat ID or new name is missing.")
    else: # Cancel
        notify(state, "info", "Rename cancelled.")
    state.show_rename_dialog_ui = False
    state.chat_id_to_rename_ui = ""
    state.new_chat_name_ui = ""


def on_suggested_prompt_action(state: State, id: str, action: str, payload: Dict):
    # The 'id' of the button might be its label if not specified otherwise.
    # Or, we can pass the prompt as an argument in the Markdown if Taipy supports it easily.
    # For now, assume `id` is the prompt text itself (from button's label).
    prompt_text = id # This depends on how <button on_action=...> passes its identifier.
                     # If `label` is used as `id`, this is fine.
    if payload and 'args' in payload and payload['args']: # More robust way if args are passed
        prompt_text = payload['args'][0].get('label', id)

    print(f"Suggested prompt clicked: {prompt_text}")
    state.chat_input_value_ui = prompt_text # Put suggestion in input box
    # Optionally, could directly trigger chat submission:
    # on_chat_input_action(state, None, None, {"args": [prompt_text]})


def on_chat_input_action(state: State, id: str, action: str, payload: Dict = None):
    user_input = state.chat_input_value_ui
    if not user_input or user_input.strip() == "":
        notify(state, "warning", "Please enter a message.")
        return

    print(f"Chat input submitted: {user_input}")
    if 'handle_user_input_callback' in _app_callbacks:
        # Add user message to history immediately for responsiveness
        # state.messages_history += [{"role": "user", "content": user_input}] # This might cause issues if not careful with Taipy's state updates

        _app_callbacks['handle_user_input_callback'](state, user_input) # app.py will handle adding to history and getting AI response
        state.chat_input_value_ui = "" # Clear input field
    else:
        # Fallback if no proper callback: just echo
        state.messages_history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": f"Echo (dev): {user_input}"}]
        state.chat_input_value_ui = ""
        notify(state, "error", "Chat handling not configured.")

def copy_message_action(state: State, id: str, action: str, payload: Dict): # type: ignore
    """Copies the content of the message (specified by index in payload) to clipboard."""
    message_idx = payload.get("index")
    if message_idx is not None and 0 <= message_idx < len(state.messages_history):
        content_to_copy = state.messages_history[message_idx]['content']

        # Clean up content: remove download/RAG markers for cleaner copy
        content_to_copy = re.sub(r"---DOWNLOAD_FILE---[^\n]+", "", content_to_copy)
        content_to_copy = re.sub(r"---RAG_SOURCE---{.*?}", "", content_to_copy, flags=re.DOTALL)
        content_to_copy = content_to_copy.strip()

        # Use Taipy's set_clipboard. If not available or not working as expected,
        # gui.eval_js would be the alternative.
        try:
            state.set_clipboard(content_to_copy) # Requires Taipy 2.0+
            notify(state, "success", "Message content copied to clipboard!")
        except AttributeError:
            # Fallback for older Taipy or if set_clipboard is not available on state
            # This would require Gui instance, which might be tricky here if not passed.
            # For now, notify about the issue.
            gui_instance = get_gui_instance() # Get existing or new instance
            if gui_instance:
                 # Escape for JS: quotes, backslashes, newlines
                js_string = json.dumps(content_to_copy)
                js_script = f"navigator.clipboard.writeText({js_string}).then(() => console.log('Copied via JS'), () => console.error('Failed to copy via JS'));"
                gui_instance.eval_js(js_script) # gui.eval_js might be the way if state.set_clipboard isn't there
                notify(state, "success", "Message content copied (attempted via JS)!") # User won't see console logs easily
            else:
                notify(state, "error", "Clipboard copy failed (UI instance not found).")
        except Exception as e:
            notify(state, "error", f"Failed to copy: {e}")
    else:
        notify(state, "error", "Could not determine message to copy.")


def regenerate_response_action(state: State, id: str, action: str, payload: Dict): # type: ignore
    print("Regenerate response action triggered")
    # The 'index' from the repeater button payload can tell us which message context this is for.
    # For regeneration, it's typically the last assistant message.
    if 'regenerate_callback' in _app_callbacks:
        _app_callbacks['regenerate_callback'](state) # app.py's callback will handle identifying the correct messages
    else:
        notify(state, "error", "Regeneration callback not configured.")

# --- GUI Instance and Initialization ---
_gui_instance: Optional[Gui] = None

def get_gui_instance(force_new: bool = False) -> Gui:
    """Gets the Gui instance, creating if necessary."""
    global _gui_instance
    if not _gui_instance or force_new:
        # Pass GLOBAL_CSS string directly to css_file argument
        _gui_instance = Gui(css_file=GLOBAL_CSS)
    return _gui_instance

def create_gui_page() -> Page:
    """
    Creates the Taipy Page object for the main application UI.
    """
    # The page content is defined by the Markdown string.
    # Callbacks are functions defined in this module. Taipy discovers them by name.
    page = Page(Markdown(main_page_md))

    # Add functions that are not automatically discovered or need to be explicitly available
    # This is often for helper functions used within the Markdown (e.g., {get_avatar_for_message(item)})
    # page.add_callable("get_avatar_for_message", get_avatar_for_message)
    # page.add_callable("get_content_for_message", get_content_for_message)
    # page.add_callable("class_name_for_message", class_name_for_message)
    # page.add_callable("has_download_or_image", has_download_or_image)
    # page.add_callable("get_download_or_image_html", get_download_or_image_html)
    # page.add_callable("has_rag_sources", has_rag_sources)
    # page.add_callable("get_rag_sources_html", get_rag_sources_html)
    # page.add_callable("is_last_assistant_message", is_last_assistant_message)
    # page.add_callable("chat_history_adapter", chat_history_adapter) # Adapter needs to be found

    # It's generally better if Taipy can find these through its module scanning.
    # If ui.py is run as the main script for Gui().run(), functions in its scope are found.
    # If Gui is created here but run from app.py, we might need to ensure context.
    return page


def init_ui(app_callbacks_from_app: Dict[str, Callable], initial_state_from_app: Dict[str, Any]):
    """
    Initializes the UI with callbacks and initial state values from app.py.
    Returns the Gui instance.
    """
    global _app_callbacks
    _app_callbacks = app_callbacks_from_app

    gui = get_gui_instance(force_new=True) # Get a fresh Gui instance

    # Set initial state values for all variables Taipy will manage for the UI
    # This should be done before the Gui runs or serves its first page.
    # We can pass them to gui.run(globals=initial_state_from_app) or set them on the state object.
    # For multi-user apps, state is per-user. For single-user (default), it's global.

    # Define the main page using the Markdown string
    # The Gui object will manage pages.
    # If we want to use the Page class directly for more control:
    # root_page = create_gui_page()
    # gui = Gui(page=root_page)

    # Simpler: Let Gui manage the Markdown page directly.
    # Callbacks defined in this module (ui.py) should be automatically found by Taipy
    # if this module is correctly in context when Gui is run.

    # The actual state object is accessed via `state` in callbacks.
    # To set initial values, they are typically passed to `gui.run(globals=...)` or
    # bound when creating controls if not using Markdown variables directly.
    # For variables in Markdown like {var_name}, their initial values are taken from `globals`
    # passed to `gui.run()` or from explicitly set shared variables.

    # The `state_vars` dict defined above holds the default structure.
    # `initial_state_from_app` will provide the actual starting values.

    # Ensure all functions that are referenced in the Markdown (e.g. `on_action=...`, `{func_name(...)}`)
    # are discoverable by Taipy. Typically, functions in the same module as the `Gui` instance or the
    # page definition are found. Using `gui._set_frame(globals())` can help if Gui is run from here.
    # If run from app.py, app.py might need to explicitly provide the module context for ui.py.

    # Add all functions from this module to be available for Taipy's expression evaluation.
    # This is important if ui.py is not the main script where Gui.run() is called.
    

    

    

    gui.add_shared_variables(initial_state_from_app) # Make initial state accessible to Markdown bindings

    # Define the page within the gui instance context
    gui.add_page("main", Markdown(main_page_md)) # Define the page with a name

    return gui


# Global style (can be refined)
GLOBAL_CSS = """
.sidebar { background-color: #f0f2f6; padding: 1em; border-radius: 8px; }
.chat_area { padding: 1em; }
.chat_input input { border-radius: 15px; } /* Style Taipy input within chat_input class */
.chat_input button { border-radius: 15px; }
.taipy-error { background-color: #ff4d4d !important; color: white !important; }
.taipy-error:hover { background-color: #cc0000 !important; }
.user_message_layout { margin-bottom: 10px; padding: 8px; background-color: #e6f3ff; border-radius: 8px 8px 0 8px; float: right; clear: both; max-width: 70%; }
.assistant_message_layout { margin-bottom: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 8px 8px 8px 0; float: left; clear: both; max-width: 70%; }
.avatar { font-size: 1.5em; margin-right: 8px; }
.message_content p { margin: 0; } /* Remove default paragraph margins in messages */
.suggested_prompts_layout button { background-color: #e0e0e0; border: none; padding: 8px 12px; border-radius: 15px; }
.suggested_prompts_layout button:hover { background-color: #c0c0c0; }
.small_button { padding: 2px 5px !important; font-size: 0.8em !important; margin-left: 5px !important; }
/* Add more Taipy-specific styling as needed */
"""
# Note: Applying CSS might need Gui(css_file="path/to/style.css") or embedding in Markdown.
# For now, this string can be written to a temp file or passed if Taipy API allows.
# Taipy typically uses a css_file argument in Gui constructor.

if __name__ == '__main__':
    # This part is for testing ui.py independently.
    # In the actual app, Gui.run() will be called from app.py.
    import sys

    # Mock app_callbacks for standalone testing
    mock_app_callbacks = {}

    def mock_new_chat_callback_fn(s: State):
        s.notify("info", "Mock: New chat created!")
        setattr(s, 'messages_history', [{"role": "assistant", "content": "Hello from new mock chat!"}])
        setattr(s, 'current_chat_id_ui', "new_mock_id")
        setattr(s, 'chat_history_lov', [("new_mock_id", "New Mock Chat")])

    def mock_switch_chat_callback_fn(s: State, chat_id: str):
        s.notify("info", f"Mock: Switched to chat {chat_id}")
        setattr(s, 'current_chat_id_ui', chat_id)
        setattr(s, 'messages_history', [{"role": "assistant", "content": f"Content for {chat_id}"}])

    def mock_remove_uploaded_file_callback_fn(s: State, name: str, ftype: str):
        s.notify("info", f"Mock: File '{name}' removed.")
        setattr(s, 'uploaded_files_display', [f for f in s.uploaded_files_display if f['name'] != name])

    def mock_set_llm_settings_callback_fn(s: State, settings: Dict):
        s.notify("info", f"Mock: LLM settings updated: {settings}")

    def mock_set_long_term_memory_callback_fn(s: State, enabled: bool):
        s.notify("info", f"Mock: Long-term memory set to {enabled}.")

    def mock_forget_me_callback_fn(s: State):
        s.notify("success", "Mock: Forget me triggered!")

    def mock_handle_user_input_callback_fn(s: State, user_input: str):
        s.notify("info", f"Mock: User input '{user_input}' sent.")
        current_messages = list(s.messages_history)
        current_messages.append({"role": "user", "content": user_input})
        current_messages.append({"role": "assistant", "content": f"Mock response to: {user_input}"})
        setattr(s, 'messages_history', current_messages)

    def mock_regenerate_callback_fn(s: State):
        s.notify("info", "Mock: Regenerate response!")
        current_messages = [msg for msg in s.messages_history if msg['role'] == 'user']
        if s.messages_history and s.messages_history[-1]['role'] == 'assistant':
            current_messages = list(s.messages_history)[:-1]
        current_messages.append({"role": "assistant", "content": "Mock: This is a regenerated response."})
        setattr(s, 'messages_history', current_messages)

    def mock_get_discussion_markdown_callback_fn(chat_id: str) -> str:
        return f"# Mock MD for {chat_id}\n- Message 1\n- Message 2"

    def mock_get_discussion_docx_callback_fn(chat_id: str) -> bytes:
        return b"Mock DOCX content"

    mock_app_callbacks = {
        'new_chat_callback': mock_new_chat_callback_fn,
        'switch_chat_callback': mock_switch_chat_callback_fn,
        'process_uploaded_file_callback': None, # This one is defined later
        'remove_uploaded_file_callback': mock_remove_uploaded_file_callback_fn,
        'set_llm_settings_callback': mock_set_llm_settings_callback_fn,
        'set_long_term_memory_callback': mock_set_long_term_memory_callback_fn,
        'forget_me_callback': mock_forget_me_callback_fn,
        'rename_chat_callback': None,
        'handle_user_input_callback': mock_handle_user_input_callback_fn,
        'regenerate_callback': mock_regenerate_callback_fn,
        'get_discussion_markdown_callback': mock_get_discussion_markdown_callback_fn,
        'get_discussion_docx_callback': mock_get_discussion_docx_callback_fn,
    }

    # Initialize state with some defaults for testing
    initial_test_state = state_vars.copy()
    initial_test_state.update({
        "chat_history_lov": [("id1", "Chat 1"), ("id2", "Old Name Chat 2")],
        "current_chat_id_ui": "id1",
        "chat_metadata_ui": {"id1": "Chat 1", "id2": "Old Name Chat 2"},
        "messages_history": [
            {"role": "assistant", "content": "Welcome to the ESI Chatbot!"},
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "Hi! How can I help you today? ---RAG_SOURCE---{\"type\": \"web\", \"url\": \"https://example.com\", \"title\": \"Example Source\"}"}
        ],
        "uploaded_files_display": [
            {"id": "file1.pdf", "name": "file1.pdf", "type": "doc", "icon": "📄", "actions": "<|button|on_action=on_uploaded_file_table_action|label=Delete|class_name=taipy-error|action=delete|>"},
        ],
        "long_term_memory_enabled_ui": True,
    })

    gui_instance = init_ui(mock_app_callbacks, initial_test_state)

    # Write CSS to a temporary file for Gui constructor if not using inline styles
    # For simplicity in this environment, we are defining it as a string.
    # Taipy Gui can take a `css_file` argument. If it's a string, it's content.
    # If Gui is already created, this won't apply. CSS should be part of Gui creation.
    # The `init_ui` function now returns the Gui instance.

    # Add CSS file to Gui instance. This is a bit of a workaround if not set at Gui init.
    # A better way is to pass css_file argument to Gui() constructor.
    # Let's assume init_ui handles Gui creation with CSS.
    # For standalone, we might need to do it like this:
    # temp_css_file = "temp_style.css"
    # with open(temp_css_file, "w") as f:
    #     f.write(GLOBAL_CSS)
    # gui_instance._config.css_file = temp_css_file # This is internal, better to set via constructor

    # The Gui instance is now created and configured in init_ui.
    # For it to find callables in *this* module when run from here:
    gui_instance._set_frame(globals())


    # Define the mock callback as a full function
    def mock_process_uploaded_file_callback_fn(s: State, name: str, content: bytes):
        s.notify("success", f"Mock: File '{name}' processed.")
        # Add to uploaded_files_display for table
        # Ensure s.uploaded_files_display is treated as a list that Taipy can update reactively
        current_files = list(s.uploaded_files_display) # Make a copy
        current_files.append({"id": name, "name": name,
                              "type": "doc" if name.endswith((".pdf",".docx")) else "df",
                              "icon": "📄" if name.endswith((".pdf",".docx")) else "📊",
                              "actions": "<|button|on_action=on_uploaded_file_table_action|label=Delete|class_name=taipy-error|action=delete|file_id=" + name + "|>"})
        s.uploaded_files_display = current_files

    def mock_rename_chat_callback_fn(s: State, chat_id: str, new_name: str):
        s.notify("success", f"Mock: Chat {chat_id} renamed to {new_name}.")
        temp_metadata = s.chat_metadata_ui.copy()
        temp_metadata[chat_id] = new_name
        s.chat_metadata_ui = temp_metadata
        # Update chat_history_lov
        s.chat_history_lov = [(cid, s.chat_metadata_ui.get(cid, "Unknown")) for cid, _ in s.chat_history_lov if cid == chat_id] + \
                             [(cid, cname) for cid, cname in s.chat_history_lov if cid != chat_id] # Rebuild to reflect name change

    # Assign the actual function to the callback dictionary
    mock_app_callbacks['process_uploaded_file_callback'] = mock_process_uploaded_file_callback_fn
    mock_app_callbacks['rename_chat_callback'] = mock_rename_chat_callback_fn

    gui_instance.run(run_server=True, port=5001, title="ESI Taipy UI Test", dark_mode=False, use_reloader=True)
    # if os.path.exists(temp_css_file):
    #     os.remove(temp_css_file)

print("ui.py structure refined with more detailed state and callbacks.")
# Next steps:
# - Ensure all callbacks correctly update Taipy's state.
# - Implement HTML/Markdown generation for downloads, images, RAG sources.
# - Test interactions thoroughly.
# - Integrate with app.py.
