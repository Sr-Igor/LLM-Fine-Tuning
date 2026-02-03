import select
import subprocess
import time
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .resource_monitor import ResourceMonitor


class TerminalPresenter:
    """Manages visual output in the terminal using Rich."""

    def __init__(self):
        """Initialize the TerminalPresenter with a Rich Console."""
        self.console = Console()

    def log(self, message: str, level: str = "info"):
        """Logs a stylized message."""
        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red bold",
        }
        style = styles.get(level, "white")
        prefix = {"info": "ℹ️ ", "success": "✅ ", "warning": "⚠️ ", "error": "❌ "}.get(level, "")

        self.console.print(f"[{style}]{prefix}{message}[/{style}]")

    def show_panel(self, title: str, content: dict, style: str = "blue"):
        """Displays an informative panel."""
        text = ""
        for k, v in content.items():
            text += f"[bold]{k}:[/bold] {v}\n"

        self.console.print(Panel(text.strip(), title=title, border_style=style))

    def run_command_with_spinner(self, cmd: List[str], message: str):
        """Executes subprocess command with spinner and monitoring."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(message, total=None)
            last_mem_update = 0

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                while True:
                    last_mem_update = self._update_resource_monitor(
                        progress, task, message, last_mem_update
                    )
                    self._process_stdout(process)

                    if process.poll() is not None:
                        break

                if process.returncode != 0:
                    stderr = process.stderr.read()
                    raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

            except Exception as e:
                self.log(f"Execution failed: {str(e)}", "error")
                raise

    def _update_resource_monitor(
        self, progress: Progress, task, message: str, last_mem_update: float
    ) -> float:
        """Updates the resource monitor if necessary."""
        current_time = time.time()
        if current_time - last_mem_update > 2.0:
            res_status = ResourceMonitor.get_resource_status()
            progress.update(task, description=f"{message} [dim]{res_status}[/dim]")
            return current_time
        return last_mem_update

    def _process_stdout(self, process: subprocess.Popen):
        """Reads and displays process output."""
        reads = [process.stdout.fileno()]
        ret = select.select(reads, [], [], 0.1)

        if ret[0]:
            line = process.stdout.readline()
            if line and line.strip():
                self.console.print(f"[dim]{line.strip()}[/dim]")
