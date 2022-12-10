import os.path
import sys
import tqdm
from gridReader import GridReader
from qconAgent import QconAgent
from game import GameInstance

def main(args):
	if len(args) < 2:
		filePath = 'grids/default.txt'
		print('No file path given, using grids/default.txt...')
		if not os.path.exists(filePath):
			print('Could not find grids/default.txt, restore the file or specify a grid path.')
			return
	elif len(args) > 2:
		print(f'usage: python main.py <Path To Grid file> or python main.py to use default')
		return

	filePath = args[1]
	game = GridReader.readGrid(filePath)
	game.printGameState()

	print(game.mainAgent.x, game.mainAgent.y)

	food = game.mainAgent.doFoodSensor()
	enemy = game.mainAgent.doEnemySensor()
	obst = game.mainAgent.doObstacleSensor()
	
	print('Food: ', food, len(food))
	print('Enemy: ', enemy, len(enemy))
	print('Obstacle: ', obst, len(obst))

	#For testing
	nb_run = 20 
	agent = QconAgent()
	env = GameInstance(initialFood=15)

	for i in range(nb_run):
		obs,rwd,done = env.reset()
		while True:
			action = agent.predict(obs)
			obs, reward, done = env.step(action)

def training(env,filename,nb_play=300,nb_run=20,nb_test=50,freqSave =30):
	agent = QconAgent()

	csvFile = open(f'{filename}.csv', 'w')
	csvFile.write('n,mean_rewards,mean_food_eaten\n')

	for i in tqdm(range(nb_play)):
		for _ in range(nb_run):
			rsum, foodEaten = simulation(env, agent, test=False)


		meanRsum, meanFoodEaten = 0, 0
		for _ in range(nb_test):
			rsum, foodEaten = simulation(env, agent, test=True)
			meanRsum += rsum
			meanFoodEaten += foodEaten

		meanRsum /= nb_test
		meanFoodEaten /= nb_test

		csvFile.write(f'{i+1},{meanRsum},{meanFoodEaten}\n')

		if (i+1) % freqSave == 0 or (i+1) == nb_play:
			agent.save(f'QAgent/save_{i+1}_{filename}')
	csvFile.close()

def simulation(env, agent, test=False):
	agent.test = test
	j = 0
	rsum = 0
	ob = env.reset()

	while True:
		action = agent.predict(ob)
		new_ob, reward, done = env.step(action)
		j += 1
		if not test:
			agent.store(ob, action, new_ob, reward, done, j)
			agent.learn()		
		ob = new_ob
		rsum += reward
		if done:
			break

	return rsum, env.getFood()

if __name__ == '__main__':
	main(sys.argv)