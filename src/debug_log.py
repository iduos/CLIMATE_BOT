#
# By Ian Drumm, The Univesity of Salford, UK.
#
import os
import sys
from typing import Any, Optional, TextIO
from pathlib import Path
import threading
from datetime import datetime

class DebugLogger:
    """
    A debug logging utility that supports both console and file output.
    Thread-safe and configurable through environment variables or direct initialization.
    """
    
    def __init__(
        self, 
        enabled: Optional[bool] = None,
        debug_file: Optional[str] = None,
        console_output: TextIO = sys.stderr,
        include_timestamp: bool = True,
        include_thread_id: bool = False
    ):
        """
        Initialize the debug logger.
        
        Args:
            enabled: Enable debug output (defaults to DEBUG_PRINT env var)
            debug_file: File path for debug output (defaults to DEBUG_FILE env var)
            console_output: Stream for console output (default: stderr)
            include_timestamp: Include timestamps in file output
            include_thread_id: Include thread ID in output
        """
        # Use environment variables as defaults if not explicitly set
        if enabled is None:
            enabled = os.getenv("DEBUG_PRINT", "").strip() == "1"
        
        if debug_file is None:
            debug_file = os.getenv("DEBUG_FILE")
        
        self.enabled = enabled
        self.debug_file_path = debug_file
        self.console_output = console_output
        self.include_timestamp = include_timestamp
        self.include_thread_id = include_thread_id
        
        # Thread safety for file operations
        self._file_lock = threading.Lock()
        
        # Ensure debug file directory exists
        if self.debug_file_path:
            debug_path = Path(self.debug_file_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
    
    def print(self, *args, **kwargs) -> None:
        """
        Print debug message to console if debugging is enabled.
        Compatible with built-in print() function.
        """
        if not self.enabled:
            return
        
        # Prepare the message
        message_parts = []
        
        if self.include_timestamp:
            message_parts.append(f"[{datetime.now().strftime('%H:%M:%S')}]")
        
        if self.include_thread_id:
            message_parts.append(f"[T{threading.get_ident()}]")
        
        # Build the prefix
        prefix = " ".join(message_parts)
        if prefix:
            prefix += " "
        
        # Override file parameter to use our console output
        kwargs['file'] = self.console_output
        
        if prefix:
            print(f"{prefix}{args[0] if args else ''}", *(args[1:] if len(args) > 1 else ()), **kwargs)
        else:
            print(*args, **kwargs)
    
    def write_file(self, label: str, content: str) -> None:
        """
        Write debug content to file with a label header.
        Thread-safe and only writes if debug file is configured.
        
        Args:
            label: Section label for the content
            content: Content to write
        """
        if not self.debug_file_path:
            return
        
        with self._file_lock:
            try:
                with open(self.debug_file_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().isoformat() if self.include_timestamp else ""
                    thread_id = f" [T{threading.get_ident()}]" if self.include_thread_id else ""
                    
                    header = f"\n\n===== {label}"
                    if timestamp:
                        header += f" [{timestamp}]"
                    header += f"{thread_id} ====="
                    
                    f.write(f"{header}\n{content}\n")
            except Exception as e:
                # Fallback to console if file write fails
                if self.enabled:
                    print(f"DEBUG: Failed to write to file {self.debug_file_path}: {e}", file=self.console_output)
    
    def write_json(self, label: str, obj: Any) -> None:
        """
        Write a JSON object to debug file with proper formatting.
        
        Args:
            label: Section label for the JSON
            obj: Object to serialize to JSON
        """
        import json
        try:
            json_content = json.dumps(obj, ensure_ascii=False, indent=2)
            self.write_file(label, json_content)
        except Exception as e:
            self.write_file(f"{label}_ERROR", f"Failed to serialize JSON: {e}\nObject: {repr(obj)}")
    
    def write_error(self, error_msg: str, additional_context: Optional[str] = None) -> None:
        """
        Write error information to both console and file.
        
        Args:
            error_msg: The error message
            additional_context: Optional additional context (e.g., raw response)
        """
        self.print(error_msg)
        
        full_content = error_msg
        if additional_context:
            full_content += f"\n\nAdditional Context:\n{additional_context}"
        
        self.write_file("ERROR", full_content)
    
    def is_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.enabled
    
    def has_file_output(self) -> bool:
        """Check if file output is configured."""
        return self.debug_file_path is not None
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable debug output at runtime."""
        self.enabled = enabled
    
    def set_debug_file(self, file_path: Optional[str]) -> None:
        """Change the debug file path at runtime."""
        self.debug_file_path = file_path
        if file_path:
            debug_path = Path(file_path)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
    
    def clear_debug_file(self) -> None:
        """Clear the debug file if it exists."""
        if not self.debug_file_path:
            return
        
        with self._file_lock:
            try:
                with open(self.debug_file_path, "w", encoding="utf-8") as f:
                    f.write(f"Debug log cleared at {datetime.now().isoformat()}\n")
            except Exception as e:
                if self.enabled:
                    print(f"DEBUG: Failed to clear file {self.debug_file_path}: {e}", file=self.console_output)


# Global debug logger instance for backward compatibility
_global_debug_logger = DebugLogger()

# Backward compatibility functions
def dprint(*args, **kwargs):
    """Legacy debug print function - uses global logger instance."""
    _global_debug_logger.print(*args, **kwargs)

def dwrite_file(label: str, content: str):
    """Legacy debug file write function - uses global logger instance."""
    _global_debug_logger.write_file(label, content)
