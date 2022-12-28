import os.path
import sys
from tqdm import tqdm
from gridReader import GridReader
from qconAgent import QconAgent
from utils import plot_examples, save_plot
import imageio.v2 as imageio
import time


def main(args):
    envPath = 'grids/default.txt'
    if len(args) < 2:
        filePath = 'grids/default.txt'
        print('No file path given, using grids/default.txt...')
        if not os.path.exists(filePath):
            print('Could not find grids/default.txt, restore the file or specify a grid path.')
            return
    elif len(args) > 2:
        print(f'usage: python main.py <Path To Grid file> or python main.py to use default')
        return
    startDate = time.time()
    dirPrefix = f'save_{startDate}/'
    try:
        os.mkdir(dirPrefix)
        os.mkdir(dirPrefix + 'QAgent')
        os.mkdir(dirPrefix + 'saveR')
    except FileExistsError:
        print(f'Unexpected folder already exists: {dirPrefix}')

    nb_play = 1

    print("----------------------TRAINING QCON AGENT----------------------")
    agent = QconAgent(dirPrefix)
    training(envPath, agent, dirPrefix, filename="Output", nb_play=nb_play, animation=True)
    print("----------------------DONE----------------------")

    print("----------------------TRAINING QCONR AGENT----------------------")
    agentR = QconAgent(dirPrefix + "saveR/", batch_size=32, memory_size=10000)  # action replay
    training(envPath, agentR, dirPrefix, filename="OutputR", nb_play=nb_play, animation=True)

    print("----------------------DONE----------------------")
    # For testing
    """
    agent = QconAgent(savedir, env)
    rsum, food = simulation(env, agent)
    print(f'Rewards = {rsum} food = {food}')
    while True:
        action = env.sample()
        obs, reward, done = env.step(action)
        render(env)
        if done:
            break
    """


def training(envPath, agent, dirPrefix, filename, nb_play=1, nb_run=20, nb_test=50, freqSave=30, r=False,
             animation=False):
    csvFile = open(f'{dirPrefix}{filename}.csv', 'w')
    csvFile.write('n,mean_rewards,mean_food_eaten\n')

    for i in tqdm(range(nb_play)):
        for run in range(nb_run):
            if animation:
                rsum, foodEaten = simulation(envPath, agent, test=False, r=r, save_animation=True, dirPrefix=dirPrefix,
                                             run_name=f'{filename}_play{i}_run{run}')
            else:
                rsum, foodEaten = simulation(envPath, agent, test=False, r=r)

        meanRsum, meanFoodEaten = 0, 0
        for test in range(nb_test):
            if animation:
                rsum, foodEaten = simulation(envPath, agent, test=False, r=r, save_animation=True, dirPrefix=dirPrefix,
                                             run_name=f'{filename}_play{i}_test{test}')
            else:
                rsum, foodEaten = simulation(envPath, agent, test=True)
            meanRsum += rsum
            meanFoodEaten += foodEaten

        meanRsum /= nb_test
        meanFoodEaten /= nb_test

        csvFile.write(f'{i + 1},{meanRsum},{meanFoodEaten}\n')

        if (i + 1) % freqSave == 0 or (i + 1) == nb_play:
            # TODO IO maybe change qconagent save to check the file
            agent.save(f'{dirPrefix}/QAgent/save_{i + 1}_{filename}')
    csvFile.close()


def simulation(envPath, agent, test=False, r=False, save_animation=False, dirPrefix='', run_name=''):
    agent.test = test
    rsum = 0
    env = GridReader.readGrid(envPath)
    ob = env.reset()
    agent.env = env
    step_count = 0
    path_save = ''
    if save_animation:
        path_save = f'{dirPrefix}animation/{run_name}/'
        if not os.path.exists(f'{dirPrefix}animation'):
            os.mkdir(f'{dirPrefix}animation')
        os.mkdir(path_save)
        animate(env, path_save + 'frame_0.png')

    while True:
        action = agent.act(ob)
        new_ob, reward, done = env.step(action)
        if not test:
            agent.store(ob, new_ob, action, reward, done)
            q, loss = agent.learn()
        rsum += reward
        ob = new_ob
        step_count += 1
        if r:
            render(env)
        if save_animation:
            animate(env, path_save + f'frame_{step_count}.png')
        if done:
            break

    if save_animation:
        with imageio.get_writer(path_save + 'animation.gif', mode='I') as writer:
            for i in range(step_count + 1):
                image = imageio.imread(path_save + f'frame_{i}.png')
                writer.append_data(image)
        for i in range(step_count + 1):
            os.remove(path_save + f'frame_{i}.png')

    return rsum, env.getFoodEaten()


def render(game):
    # game.printGameState()
    plot = game.affPlot()
    plot_examples(plot)
    time.sleep(1)


def animate(game, filepath):
    plot = game.affPlot()
    save_plot(plot, filepath)


if __name__ == '__main__':
    main(sys.argv)
