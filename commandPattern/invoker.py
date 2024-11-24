from commandPattern.command import Command

class CommandInvoker:
    def __init__(self):
        self._commands = []
        self._history = []

    def add_command(self, command: Command) -> None:
        self._commands.append(command)

    def execute_commands(self) -> None:
        for command in self._commands:
            command.execute()
            self._history.append(command)
        self._commands.clear()

    def undo_last_command(self) -> None:
        if self._history:
            command = self._history.pop()
            command.undo()