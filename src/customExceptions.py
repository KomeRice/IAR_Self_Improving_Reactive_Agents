class InvalidMoveError(Exception):
	def __init__(self, agentId, agentX, agentY, moveAttempted):
		self.agentId = agentId
		self.agentX = agentX
		self.agentY = agentY
		self.moveAttempted = moveAttempted
		super().__init__(f'Agent {agentId} @ ({agentX}, {agentY}) attempted illegal move ({moveAttempted})')

	def __str__(self):
		return f'Agent {self.agentId} @ ({self.agentX}, {self.agentY}) attempted illegal move ({self.moveAttempted})'

class GridReaderError(Exception):
	def __init__(self, lineNumber, filePath, errorMessage):
		self.lineNumber = lineNumber
		self.filePath = filePath
		self.errorMessage = errorMessage
		super().__init__(f'Could not read file @ {filePath}: \n{errorMessage} @ Line {lineNumber} (Ignoring comments)')

	def __str__(self):
		return f'Could not read file @ {self.filePath}: \n{self.errorMessage} @ Line {self.lineNumber} (Ignoring comments)'
