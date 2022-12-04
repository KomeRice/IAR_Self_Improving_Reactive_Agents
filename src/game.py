import gameAgents as ag
from customExceptions import InvalidMoveError
import numpy as np

class GameInstance:
	def __init__(self, rows = 25, cols = 25, initialFood = 10):
		self.rows = rows
		self.cols = cols
		self.grid = [[' ' for _ in range(cols)] for _ in range(rows)]
		self.remainingFood = initialFood
		self.initialFood = initialFood
		self.agents = {}
		self.mainAgent = None

	def addMainAgent(self, x, y):
		newMain = ag.MainAgent(self, x, y)
		self.agents[newMain.id] = newMain
		self.grid[y][x] = newMain.id
		self.mainAgent = newMain

	def addEnemyAgent(self, x, y):
		newEnemy = ag.EnemyAgent(self, x, y)
		self.agents[newEnemy.id] = newEnemy
		self.grid[y][x] = newEnemy.id

	def addFoodAgent(self, x, y):
		newFood = ag.FoodAgent(self, x, y)
		self.agents[newFood.id] = newFood
		self.grid[y][x] = newFood.id

	def movePossible(self, agent, deltaX, deltaY):
		self.did_collide = 0 <= agent.x + deltaX < self.cols and 0 <= agent.y + deltaY < self.rows and self.at(agent.x + deltaX, agent.y + deltaY) != 'O'
		return self.did_collide

	def at(self, x, y):
		return self.grid[y][x]

	def isAgentAt(self, x, y):
		return self.at(x, y) not in ['O', ' ']

	def getAgentAt(self, x, y):
		return self.agents[self.at(x, y)]

	def doMove(self, agent, deltaX, deltaY):
		if not self.movePossible(agent, deltaX, deltaY):
			return False,0
		newX = agent.x + deltaX
		newY = agent.y + deltaY

		# Main Agent special behaviors
		if isinstance(agent, ag.MainAgent):
			# Check for food
			reward = 0
			if self.isAgentAt(newX, newY):
				agentAtLoc = self.getAgentAt(newX, newY)
				if agentAtLoc.symbol == '$':
					agent.energy += agentAtLoc.energyGain
					reward+=0.4
					self.remainingFood -= 1
					if self.remainingFood == 0:
						print('Game over: Won by agent')
						return True,reward
				# Check for enemy
				elif agentAtLoc.symbol == 'E':
					print('Game over: Died to enemy')
					return True,-1
			agent.energy -= 1
			# Check for energy
			if agent.energy == 0:
				print('Game over: Died to exhaustion')
				return True,reward

		self.grid[agent.y][agent.x] = ' '
		self.grid[newY][newX] = agent.id
		agent.x = newX
		agent.y = newY
		return False,reward

	def getGameStateString(self):
		startEnd = '+' + ''.join('-+' for _ in range(self.cols))
		lines = [startEnd]
		for y in range(self.rows):
			newLine = [(self.getAgentAt(x, y).symbol if self.isAgentAt(x, y) else self.at(x, y)) for x in range(self.cols)]
			lines.append('|' + '|'.join(newLine) + '|')
		lines.append(startEnd)
		return '\n'.join(lines)

	def printGameState(self):
		print(self.getGameStateString())

	def step(self,agent):
		done,reward = self.mainAgent.step()
		for k in self.agents:
			if isinstance(self.agents[k], ag.MainAgent):
				continue
			self.agents[k].step()
		obs = self.observation()
		return obs,reward,done

	def observation(self):
		food = self.mainAgent.doFoodSensor()
		ennemy = self.mainAgent.doEnemySensor()
		obstacle = self.mainAgent.doObstacleSensor()
		energy = self.mainAgent.energy
		max_energy = self.mainAgent.max_energy
		previous_action = self.mainAgent.previous_action

		obs = [*ennemy,*food,*obstacle]
		for i in range(16):
			if i<=round(16/max_energy*energy):
				obs.append(1)
			else:
				obs.append(0)
		obs = obs + previous_action
		if self.did_collide:
			obs.append(1)
		else:
			obs.append(0)
		return obs

