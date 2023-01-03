import random
from customExceptions import InvalidMoveError
import numpy as np
import math


class Agent:
    newId = 0

    def __init__(self, gameInst, x=0, y=0, symbol='A'):
        self.x = x
        self.y = y
        self.id = Agent.newId
        self.symbol = symbol
        self.gameInst = gameInst
        self.did_collide = False
        Agent.newId += 1

    def moveUp(self):
        try:
            return self.gameInst.doMove(self, 0, -1)
        except InvalidMoveError:
            print(InvalidMoveError)

    def moveDown(self):
        try:
            return self.gameInst.doMove(self, 0, 1)
        except InvalidMoveError:
            print(InvalidMoveError)

    def moveLeft(self):
        try:
            return self.gameInst.doMove(self, -1, 0)
        except InvalidMoveError:
            print(InvalidMoveError)

    def moveRight(self):
        try:
            return self.gameInst.doMove(self, 1, 0)
        except InvalidMoveError:
            print(InvalidMoveError)

    def step(self):
        raise NotImplementedError

    def tilesInRadiusGen(self, radius, resolution=1, minDist=1):
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
    def __init__(self, gameInst, x=0, y=0, symbol='$', energyGain=15):
        super().__init__(gameInst, x, y, symbol)
        self.energyGain = energyGain

    def step(self):
        return


class MainAgent(Agent):
    def __init__(self, gameInst, x=0, y=0, symbol='I', baseEnergy=40):
        super().__init__(gameInst, x, y, symbol)
        self.energy = baseEnergy
        self.previous_action = [0, 0, 0, 0]
        self.max_energy = baseEnergy + 15 * gameInst.initialFood

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

        self.sensY = sensorY
        self.sensO = sensorO
        self.sensX = sensorX

        self.foodSensor = [sensorX, sensorO, sensorY]
        self.enemySensor = [sensorX, sensorO]
        self.obstacleSensor = [{
            'radius': 4,
            'resolution': 1,
            'minDist': 1
        }]

        self.sensorWall = self.obstacleSensor[0]

    def doFoodSensor(self, orientation=0):
        obsList = []
        for sensor in self.foodSensor:
            for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
                effective_x, effective_y = self.orientation(x, y, orientation)
                try:
                    if self.gameInst.isAgentAt(effective_x, effective_y):
                        agent = self.gameInst.getAgentAt(effective_x, effective_y)
                        if isinstance(agent, FoodAgent):
                            obsList.append(1)
                            continue
                except IndexError:
                    obsList.append(0)
                    continue

                obsList.append(0)
        return np.array(obsList)

    def doEnemySensor(self, orientation=0):
        obsList = []
        for sensor in self.enemySensor:
            for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
                effective_x, effective_y = self.orientation(x, y, orientation)
                try:
                    if self.gameInst.isAgentAt(effective_x, effective_y):
                        agent = self.gameInst.getAgentAt(effective_x, effective_y)
                        if isinstance(agent, EnemyAgent):
                            obsList.append(1)
                            continue
                except IndexError:
                    obsList.append(0)
                    continue

                obsList.append(0)
        return np.array(obsList)

    def doObstacleSensor(self, orientation=0):
        obsList = []
        for sensor in self.obstacleSensor:
            for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
                effective_x, effective_y = self.orientation(x, y, orientation)
                try:
                    if self.gameInst.at(effective_x, effective_y) == 'O':
                        obsList.append(1)
                        continue
                except IndexError:
                    obsList.append(0)
                    continue

                obsList.append(0)
        return np.array(obsList)

    def orientation(self, x, y, orientation):
        orientation = orientation % 4
        diffx = self.x - x
        diffy = self.y - y
        new_diffx = diffx
        new_diffy = diffy

        if orientation == 3:
            new_diffx = diffy
            new_diffy = -diffx
        elif orientation == 2:
            new_diffx = -diffx
            new_diffy = -diffy
        elif orientation == 1:
            new_diffx = -diffy
            new_diffy = diffx

        return self.x + new_diffx, self.y + new_diffy

    def doAllSensors(self, orientation=0):
        """ Performs a scan of surrounding areas according to sensors defined by self.foodSensor, self.enemySensor
		and self.obstacleSensor; according to a given orientation.

		:param orientation: (0 - Facing up),
		(1 - Facing right),
		(2 - Facing down),
		(3 - Facing left)
		:return: A tuple (food, enemy, obstacle) of bit-encoded np.array objects corresponding to each sensor output
		"""
        return self.doFoodSensor(orientation), self.doEnemySensor(orientation), self.doObstacleSensor(orientation)

    def getLabeledPositions(self, orientation=0):
        out = []
        identifier = 0
        for sensor, label in zip([self.sensY, self.sensO, self.sensX, self.sensorWall], ['Y', 'O', 'X', 'W']):
            for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
                effective_x, effective_y = self.orientation(x, y, orientation)
                out.append({'coords': (effective_x, effective_y),
                            'label': label,
                            'orientation': orientation,
                            'id': identifier})
                identifier += 1
        return out


    def observ(self, orientation=0):  # TODO why the orientation is not taken in account
        """ Performs a scan of surrounding areas according to sensors defined by self.foodSensor, self.enemySensor
		and self.obstacleSensor; according to a given orientation.

		:param orientation: (0 - Facing up),
		(1 - Facing right),
		(2 - Facing down),
		(3 - Facing left)
		:return: A array [food, enemy, obstacle,energy,previous_action,colide] of bit-encoded np.array objects corresponding to each output
		"""

        food, ennemy, obstacle = self.doAllSensors(orientation)

        energy = self.energy
        max_energy = self.max_energy
        previous_action = self.previous_action

        obs = [*ennemy, *food, *obstacle]
        for i in range(16):
            if i == round(16 / max_energy * energy):
                obs.append(1)
            else:
                obs.append(0)
        obs = obs + previous_action
        if self.did_collide:
            obs.append(1)
        else:
            obs.append(0)
        return obs

    def observation(self):
        return [self.observ(i) for i in range(4)]

    def step(self, ac):
        """ac between 0 and 3 corresponding to the action taken"""
        action = [self.moveUp, self.moveRight, self.moveDown, self.moveLeft]
        self.previous_action = []
        for a in range(len(action)):
            if a == ac:
                self.previous_action.append(1)
            else:
                self.previous_action.append(0)

        obs = self.observation()
        done, rwd = action[ac]()
        return obs, rwd, done  # return the if done and the reward of the action taken


class EnemyAgent(Agent):
    def __init__(self, gameInst, x=0, y=0, symbol='E', moveChance=0.8):
        super().__init__(gameInst, x, y, symbol)
        self.moveChance = moveChance

    def step(self):  # appendix a
        if random.random() > self.moveChance:
            return
        old_pos = [self.x, self.y]
        actions = [self.moveUp, self.moveRight, self.moveDown, self.moveLeft]
        act = [[self.x, self.y - 1], [self.x + 1, self.y], [self.x, self.y - 1], [self.x - 1, self.y]]
        prob = []
        for k in act:
            prob.append(math.exp(0.33 * self.w_angle(k) * self.T_dist()))
        random.choices(actions, weights=prob)[0]()
        if [self.x, self.y] == old_pos:  # on fait quoi si il reste sur place?
            return

    def T_dist(self):
        a = np.array([self.gameInst.mainAgent.x, self.gameInst.mainAgent.y])
        b = np.array([self.x, self.y])
        dist = np.linalg.norm(a - b)
        if dist <= 4:
            return 15 - dist
        elif dist <= 15:
            return 9 - dist / 2
        else:
            return 1

    def w_angle(self, pos_aftermove):
        a = np.array([self.x, self.y])
        b = np.array(pos_aftermove)
        inner = np.inner(a, b)
        norms = np.linalg.norm(a) * np.linalg.norm(b)
        cos = inner / norms
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        angle = np.rad2deg(rad)
        return (180 - abs(angle)) / 180
