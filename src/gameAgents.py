import random
import numpy as np
from customExceptions import InvalidMoveError
import numpy as np
import math

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

	def tilesInRadiusGen(self, radius, resolution = 1, minDist = 1):
		for i in range(-radius, radius + 1, resolution):
			jRad = radius - abs(i)
			for j in range(-jRad, jRad + 1, resolution):
				if abs(i) + abs(j) < minDist:
					continue
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

		sensorY = {
			'radius': 10,
			'resolution': 2,
			'minDist': 10
		}
		sensorO = {
			'radius': 6,
			'resolution': 2,
			'minDist': 3
		}
		sensorX = {
			'radius': 2,
			'resolution': 1,
			'minDist': 1
		}

		self.foodSensor = [sensorX, sensorO, sensorY]
		self.enemySensor = [sensorX, sensorO]
		self.obstacleSensor = [{
			'radius': 4,
			'resolution': 1,
			'minDist': 1
		}]

	def doFoodSensor(self):
		obsList = []
		for sensor in self.foodSensor:
			for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
				try:
					if self.gameInst.isAgentAt(x, y):
						agent = self.gameInst.getAgentAt(x, y)
						if isinstance(agent, FoodAgent):
							obsList.append(1)
							continue
				except IndexError:
					obsList.append(0)
					continue

				obsList.append(0)
		return np.array(obsList)

	def doEnemySensor(self):
		obsList = []
		for sensor in self.enemySensor:
			for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
				try:
					if self.gameInst.isAgentAt(x, y):
						agent = self.gameInst.getAgentAt(x, y)
						if isinstance(agent, EnemyAgent):
							obsList.append(1)
							continue
				except IndexError:
					obsList.append(0)
					continue

				obsList.append(0)
		return np.array(obsList)

	def doObstacleSensor(self):
		obsList = []
		for sensor in self.obstacleSensor:
			for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
				try:
					if self.gameInst.at(x, y) == 'O':
						obsList.append(1)
						continue
				except IndexError:
					obsList.append(0)
					continue

				obsList.append(0)
		return np.array(obsList)

	def step(self):
		action = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
		random.choice(action)()

		return

class EnemyAgent(Agent):
	def __init__(self, gameInst, x = 0, y = 0, symbol='E', moveChance = 0.8):
		super().__init__(gameInst, x, y, symbol)
		self.moveChance = moveChance

	def step(self):#appendix a
		if random.random() > self.moveChance:
			return
		old_pos = [self.x,self.y]
		actions = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
		act = [[self.x,self.y-1],[self.x,self.y+1],[self.x-1,self.y],[self.x+1,self.y]]
		prob = []
		for k in act:
			prob.append(math.exp(0.33*self.w_angle(k)*self.T_dist()))
		random.choices(actions,weights=prob)[0]()
		if [self.x,self.y] == old_pos:# on fait quoi si il reste sur place?
			print("Dead")
		

	
	def T_dist(self):
		a = np.array([self.gameInst.mainAgent.x,self.gameInst.mainAgent.y])
		b = np.array([self.x,self.y])
		dist=np.linalg.norm(a-b)
		if dist<=4:
			return 15-dist
		elif dist<=15:
			return 9-dist/2
		else:
			return 1
	
	def w_angle(self,pos_aftermove):
		a = np.array([self.x,self.y])
		b = np.array(pos_aftermove)
		inner = np.inner(a, b) 
		norms = np.linalg.norm(a) * np.linalg.norm(b)
		cos = inner / norms
		rad = np.arccos(np.clip(cos, -1.0, 1.0))
		angle = np.rad2deg(rad)
		return (180-abs(angle))/180
