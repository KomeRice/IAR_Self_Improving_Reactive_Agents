import random
import numpy as np
from customExceptions import InvalidMoveError

class Agent:
	newId = 0
	def __init__(self, gameInst, x = 0, y = 0, symbol = 'A'):
		self.x = x
		self.y = y
		self.id = Agent.newId
		self.symbol = symbol
		self.gameInst = gameInst
		Agent.newId += 1

	def moveUp(self):
		try:
			self.gameInst.doMove(self, 0, -1)
		except InvalidMoveError:
			print(InvalidMoveError)

	def moveDown(self):
		try:
			self.gameInst.doMove(self, 0, 1)
		except InvalidMoveError:
			print(InvalidMoveError)

	def moveLeft(self):
		try:
			self.gameInst.doMove(self, -1, 0)
		except InvalidMoveError:
			print(InvalidMoveError)

	def moveRight(self):
		try:
			self.gameInst.doMove(self, 1, 0)
		except InvalidMoveError:
			print(InvalidMoveError)

	def step(self):
		raise NotImplementedError

	def tilesInRadiusGen(self, radius, resolution = 1):
		for i in range(-radius, radius + 1):
			jRad = radius - np.abs(i)
			for j in range(-jRad, jRad + 1, resolution):
				yield self.x + i, self.y + j

	def __str__(self):
		return self.symbol

	def __eq__(self, other):
		return str(other) == self.symbol


class FoodAgent(Agent):
	def __init__(self, gameInst, x = 0, y = 0, symbol = '$', energyGain = 15):
		super().__init__(gameInst, x, y, symbol)
		self.energyGain = energyGain

	def step(self):
		return


class MainAgent(Agent):
	def __init__(self, gameInst, x = 0, y = 0, symbol = 'I', baseEnergy = 40):
		super().__init__(gameInst, x, y, symbol)
		self.energy = baseEnergy

	def step(self):
		choices = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
		random.choice(choices)()

		return

class EnemyAgent(Agent):
	def __init__(self, gameInst, x = 0, y = 0, symbol='E', moveChance = 0.8):
		super().__init__(gameInst, x, y, symbol)
		self.moveChance = moveChance

	def step(self):
		if random.random() > self.moveChance:
			return

		choices = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
		random.choice(choices)()

	# TODO: Probability distribution (See appendix A)
