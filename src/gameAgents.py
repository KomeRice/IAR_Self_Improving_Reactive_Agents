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
    food_detector = [[(10, 0), (8, -2), (6, -4), (4, -6), (2, -8),
                    (0, -10), (-2, -8), (-4, -6), (-6, -4), (-8, -2),
                    (-10, 0), (-8, 2), (-6, 4), (-4, 6), (-2, 8),
                    (0, 10), (2, 8), (4, 6), (6, 4), (8, 2)],
                    [(6, 0), (4, -2), (2, -4),
                    (0, -6), (-2, -4), (-4, -2),
                    (-6, 0), (-4, 2), (-2, 4),
                    (0, 6), (2, 4), (4, 2),
                    (4, 0), (2, -2), (0, -4), (-2, -2), (-4, 0), (-2, 2), (0, 4), (2, 2)],
                    [(2, 0), (1, -1), (0, -2), (-1, -1), (-2, 0), (-1, 1), (0, 2), (1, 1),
                    (1, 0), (0, -1), (-1, 0), (0, 1)]]
    enemy_detector = [[(6, 0), (4, -2), (2, -4),
                     (0, -6), (-2, -4), (-4, -2),
                     (-6, 0), (-4, 2), (-2, 4),
                     (0, 6), (2, 4), (4, 2),
                     (4, 0), (2, -2), (0, -4), (-2, -2), (-4, 0), (-2, 2), (0, 4), (2, 2)],
                     [(2, 0), (1, -1), (0, -2), (-1, -1), (-2, 0), (-1, 1), (0, 2), (1, 1),
                     (1, 0), (0, -1), (-1, 0), (0, 1)]]
    obstacle_detector = [(4, 0), (3, -1), (2, -2), (1, -3),
                        (0, -4), (-1, -3), (-2, -2), (-3, -1),
                        (-4, 0), (-3, 1), (-2, 2), (-1, 3),
                        (0, 4), (1, 3), (2, 2), (3, 1),
                        (3, 0), (2, -1), (1, -2),
                        (0, -3), (-1, -2), (-2, -1),
                        (-3, 0), (-2, 1), (-1, 2),
                        (0, 3), (1, 2), (2, 1),
                        (2, 0), (1, -1), (0, -2), (-1, -1), (-2, 0), (-1, 1), (0, 2), (1, 1),
                        (1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, gameInst, x=0, y=0, symbol='I', baseEnergy=40):
        super().__init__(gameInst, x, y, symbol)
        self.energy = baseEnergy
        self.previous_action = [0, 0, 0, 0]
        self.max_energy = baseEnergy + 15 * gameInst.initialFood -gameInst.initialFood
        self.did_collide = False

        sensorY = {
            'radius': 10,
            'resolution': 2,
            'minDist': 10,
            'reach': [(0, -2),
                      (-1, -1), (0, -1), (1, -1),
                      (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                      (-1, 1), (0, 1), (1, 1),
                      (0, 2)]
        }
        sensorO = {
            'radius': 6,
            'resolution': 2,
            'minDist': 3,
            'reach': [(-1, -1), (0, -1), (1, -1),
                      (-1, 0), (0, 0), (1, 0),
                      (-1, 1), (0, 1), (1, 1)]
        }
        sensorX = {
            'radius': 2,
            'resolution': 1,
            'minDist': 1,
            'reach': [(0, -1),
                      (-1, 0), (0, 0), (1, 0),
                      (0, 1)]
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
        for sensor, reach in zip(MainAgent.food_detector, [self.sensY['reach'], self.sensO['reach'], self.sensX['reach']]):
            for x, y in sensor:
                obsList.append(self.gameInst.sensor(x, y, reach, FoodAgent))
        return np.array(obsList)

    def doEnemySensor(self):
        obsList = []
        for sensor, reach in zip(MainAgent.enemy_detector, [self.sensO['reach'], self.sensX['reach']]):
            for x, y in sensor:
                obsList.append(self.gameInst.sensor(x, y, reach, EnemyAgent))
        return np.array(obsList)

    def doObstacleSensor(self):
        obsList = []
        for x, y in MainAgent.obstacle_detector:
            effective_x, effective_y = self.x + x, self.y + y
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

    def doAllSensors(self):
        """ Performs a scan of surrounding areas according to sensors defined by self.foodSensor, self.enemySensor
		and self.obstacleSensor; according to a given orientation.

		:param orientation: (0 - Facing up),
		(1 - Facing right),
		(2 - Facing down),
		(3 - Facing left)
		:return: A tuple (food, enemy, obstacle) of bit-encoded np.array objects corresponding to each sensor output
		"""

        return self.doFoodSensor(), self.doEnemySensor(), self.doObstacleSensor()

    def getLabeledPositions(self):
        out = []
        identifier = 0
        for sensor, label in zip([self.sensY, self.sensO, self.sensX, self.sensorWall], ['Y', 'O', 'X', 'W']):
            for x, y in self.tilesInRadiusGen(sensor['radius'], sensor['resolution'], sensor['minDist']):
                effective_x, effective_y = self.orientation(x, y, 0)
                out.append({'coords': (effective_x, effective_y),
                            'label': label,
                            'id': identifier})
                identifier += 1
        return out


    def observ(self):
        """ Performs a scan of surrounding areas according to sensors defined by self.foodSensor, self.enemySensor
		and self.obstacleSensor; according to a given orientation.

		:param orientation: (0 - Facing up),
		(1 - Facing right),
		(2 - Facing down),
		(3 - Facing left)
		:return: A array [food, enemy, obstacle,energy,previous_action,colide] of bit-encoded np.array objects corresponding to each output
		"""

        food, ennemy, obstacle = self.doAllSensors()

        energy = self.energy
        max_energy = self.max_energy
        previous_action = self.previous_action

        obs = [*food, *ennemy, *obstacle]
        c = len(obs)
        n = []

        for i in range(16):
            if i <= round(16 / max_energy * energy):
                obs.append(1)
                n.append(c + len(n))
            else:
                obs.append(0)
                n.append(c + len(n))
        obs = obs + [*previous_action]
        n.extend((i + c) for i in range(len(self.previous_action)))
        if self.did_collide:
            obs.append(1)
            n.append(c + len(n))
        else:
            obs.append(0)
            n.append(c + len(n))
        return np.array(obs)

    def observation(self):
        return self.observ()

    def step(self, ac):
        """ac between 0 and 3 corresponding to the action taken"""
        action = [self.moveUp, self.moveRight, self.moveDown, self.moveLeft]
        self.previous_action = np.zeros(4)
        self.previous_action[ac] = 1
        done, rwd = action[ac]()
        obs = self.observation()
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
        if [self.x, self.y] == old_pos:
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
