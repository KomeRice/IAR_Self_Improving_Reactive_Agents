import random
import gameAgents as ag
from customExceptions import InvalidMoveError
import numpy as np


class GameInstance:
    def __init__(self, rows=25, cols=25, initialFood=10):
        self.rows = rows
        self.cols = cols
        self.grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        self.remainingFood = initialFood
        self.initialFood = initialFood
        self.agents = {}
        self.mainAgent = None
        self.state_space = 145
        self.action_space = 4
        self.verbose = False
        self.initialMainAgentPosition = []
        self.initialEnemyPositions = []
        self.initialEmptyTiles = []
        self.initialWalls = []
        self.foodPositionsToRestore = {}

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

    def distributeFood(self):
        emptyTiles = self.initialEmptyTiles.copy()
        for _ in range(self.initialFood):
            foodPos = random.choice(emptyTiles)
            emptyTiles.remove(foodPos)
            self.addFoodAgent(foodPos[0], foodPos[1])

    def movePossible(self, agent, deltaX, deltaY):
        possible = 0 <= agent.x + deltaX < self.cols and 0 <= agent.y + deltaY < self.rows and self.at(
            agent.x + deltaX, agent.y + deltaY) != 'O'
        agent.did_collide = not possible
        return possible

    def at(self, x, y):
        if x < 0 or y < 0 or x >= self.cols or y >= self.rows:
            return 'O'
        return self.grid[y][x]

    def isAgentAt(self, x, y):
        return self.at(x, y) not in ['O', ' ']

    def getAgentAt(self, x, y):
        return self.agents[self.at(x, y)]

    def sensor(self, x, y, reach_matrix, type_to_detect):
        for offx, offy in reach_matrix:
            try:
                if self.isAgentAt(x + offx, y + offy):
                    if isinstance(self.getAgentAt(x + offx, y + offy), type_to_detect):
                        return 1
            except IndexError:
                continue
        return 0

    def doMove(self, agent, deltaX, deltaY):
        if not self.movePossible(agent, deltaX, deltaY):
            if isinstance(agent, ag.MainAgent):
                agent.energy -= 1
                if agent.energy == 0:
                    if self.verbose:
                        print('Game over: Died to exhaustion')
                    return True, -1
            return False, 0
        newX = agent.x + deltaX
        newY = agent.y + deltaY
        reward = 0
        # Main Agent special behaviors
        if isinstance(agent, ag.MainAgent):
            # Check for food
            if self.isAgentAt(newX, newY):
                agentAtLoc = self.getAgentAt(newX, newY)
                if agentAtLoc.symbol == '$':
                    agent.energy += agentAtLoc.energyGain
                    reward += 0.4
                    self.remainingFood -= 1
                    if self.remainingFood == 0:
                        if self.verbose:
                            print('Game over: Won by agent')
                        return True, reward
                # Check for enemy
                elif agentAtLoc.symbol == 'E':
                    if self.verbose:
                        print('Game over: Died to enemy')
                    return True, -1
            agent.energy -= 1
            # Check for energy
            if agent.energy == 0:
                if self.verbose:
                    print('Game over: Died to exhaustion')
                return True, -1

        if (agent.x, agent.y) in self.foodPositionsToRestore:
            self.grid[agent.y][agent.x] = self.foodPositionsToRestore[(agent.x, agent.y)].id
            self.foodPositionsToRestore.pop((agent.x, agent.y))
        else:
            self.grid[agent.y][agent.x] = ' '
        if isinstance(agent, ag.EnemyAgent) and self.isAgentAt(newX, newY):
            overwritten = self.getAgentAt(newX, newY)
            if isinstance(overwritten, ag.FoodAgent):
                self.foodPositionsToRestore[(newX, newY)] = overwritten

        self.grid[newY][newX] = agent.id
        agent.x = newX
        agent.y = newY
        return False, reward

    def getGameStateString(self):
        startEnd = '+' + ''.join('-+' for _ in range(self.cols))
        lines = [startEnd]
        for y in range(self.rows):
            newLine = [(self.getAgentAt(x, y).symbol if self.isAgentAt(x, y) else self.at(x, y)) for x in
                       range(self.cols)]
            lines.append('|' + '|'.join(newLine) + '|')
        lines.append(startEnd)
        return '\n'.join(lines)

    def printGameState(self):
        print(self.getGameStateString())

    def affPlot(self):
        convertDict = {' ': 0, 'O': 1, 'E': 2, 'I': 3, '$': 4}
        stringList = self.getGameStateString().split('\n')
        intList = []
        for w in stringList:
            line = []
            for l in w:
                if l in convertDict:
                    line.append(convertDict[l])
            if line:
                intList.append(line)
        return intList

    def step(self, action):
        obs, rwd, done = self.mainAgent.step(action)
        for k in self.agents:
            if isinstance(self.agents[k], ag.MainAgent):
                continue
            self.agents[k].step()
        return obs, rwd, done

    def sample(self):
        return np.random.choice(self.action_space)

    def envReset(self):
        self.grid = [[' ' for _ in range(self.cols)] for _ in range(self.rows)]
        self.remainingFood = self.initialFood
        self.foodPositionsToRestore = {}
        self.agents = {}
        for x, y in self.initialWalls:
            self.grid[y][x] = 'O'
        for x, y in self.initialEnemyPositions:
            self.addEnemyAgent(x, y)
        for x, y in self.initialMainAgentPosition:
            self.addMainAgent(x, y)
        self.distributeFood()

    def reset(self):
        self.envReset()
        observation = self.mainAgent.observation()
        return observation

    def getFoodEaten(self):
        return self.initialFood - self.remainingFood
