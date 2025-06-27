import os
from typing import List, Dict, Any, Optional, Callable
from taipy.gui import Gui, State, Markdown, Page

# Minimal state
state_vars = {
    "simple_message": "Hello from Minimal Taipy UI!"
}

# Minimal Markdown page
main_page_md = """
# ESI Minimal Test Page
<|text|{simple_message}|>
"""

_app_callbacks: Dict[str, Callable] = {}
_gui_instance: Optional[Gui] = None

GLOBAL_CSS = """
body { font-family: sans-serif; margin: 20px; }
"""

def get_gui_instance(force_new: bool = False) -> Gui:
    global _gui_instance
    if not _gui_instance or force_new:
        _gui_instance = Gui(css_file=GLOBAL_CSS) # Pass css string directly
    return _gui_instance

def init_ui(app_callbacks_from_app: Dict[str, Callable], initial_state_from_app: Dict[str, Any]):
    global _app_callbacks
    _app_callbacks = app_callbacks_from_app

    gui = get_gui_instance(force_new=True)
    
    # Add shared variables (even for minimal state)
    # This ensures Taipy knows about 'simple_message' from the initial state
    # gui.add_shared_variables(initial_state_from_app) # REMOVED - Relying on gui.run(globals=...)

    # Define the minimal page
    # No complex callables needed for this minimal version
    gui.add_page("main", Markdown(main_page_md))

    return gui

print("ui.py structure drastically simplified for debugging.")

# All other functions (helpers, complex callbacks, adapters) are effectively removed or would be commented out.
# For overwrite_file_with_block, I am providing only the essential functions.
# Callbacks like on_chat_menu_action, handle_file_upload_action, etc., are omitted.
# Helper functions like class_name_for_message, get_download_or_image_html, etc., are omitted.
# The full original list of commented-out functions is not included here for brevity,
# as they are not part of this minimal working example.
