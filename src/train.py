import math
import os.path
import sys
from tqdm import tqdm
from gridReader import GridReader
from qconAgent import QconAgent,DQNAgent
from utils import plot_examples, save_plot
import imageio.v2 as imageio
import time


def main(args):
    envPath = 'grids/default.txt'
    if len(args) < 2:
        print('No file path given, using grids/default.txt...')
    elif len(args) > 2:
        print(f'usage: python main.py <Path To Grid file> or python main.py to use default')
        return
    elif len(args) == 2:
        if not os.path.exists(args[1]):
            print('Could not find grids/default.txt, restore the file or specify a grid path.')
            return
        envPath = args[1]
    startDate = time.time()
    dirPrefix = f'save_{startDate}/'
    try:
        os.mkdir(dirPrefix)
        os.mkdir(dirPrefix + 'QAgent')
    except FileExistsError:
        print(f'Unexpected folder already exists: {dirPrefix}')
    env = GridReader.readGrid(envPath)
    nb_play = 20
    nb_test = 50
    nb_train = 20
    anim_period = 100
    test_period = 20
    do_anim = False

    #print("----------------------TRAINING QCON AGENT----------------------")
    agentClass = QconAgent
    training(env, agentClass, dirPrefix, filename="Output", nb_play=nb_play, nb_test=nb_test, nb_train=nb_train, animation=do_anim, anim_period=anim_period, test_period=test_period)
    #print("----------------------DONE----------------------")
    print("----------------------TRAINING DQN AGENT----------------------")
    #agentClass = DQNAgent
    #training(env, agentClass, dirPrefix, filename="OutputDQN", nb_play=nb_play, nb_test=nb_test, nb_train=nb_train, animation=do_anim, anim_period=anim_period, test_period=test_period)
    print("----------------------DONE----------------------")
    #print("----------------------TRAINING QCONR AGENT----------------------")
    #agentR = QconAgent(dirPrefix + "saveR/", batch_size=32, memory_size=10000)  # action replay
    #training(env, QconAgent, dirPrefix, filename="OutputR", nb_play=nb_play, nb_test=nb_test, nb_train=nb_train, animation=do_anim, anim_period=anim_period, test_period=test_period, agent=agentR)
    #print("----------------------DONE----------------------")
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


def training(env, agentClass, dirPrefix, filename, nb_play=20, nb_train=20, nb_test=50, freqSave=30, r=False, test_period=20,
             animation=False, anim_period=1, agent=None):
    csvFile = open(f'{dirPrefix}{filename}.csv', 'w')
    csvFile.write('n,mean_rewards,mean_food_eaten\n')
    if agent is None:
        agent = agentClass(dirPrefix)

    for i in tqdm(range(nb_play)):
        meanRsum, meanFoodEaten = 0, 0
        for run in range(nb_train):
            if animation and run == nb_train -1:
                rsum, foodEaten = simulation(env, agent, test=False, save_animation=True, dirPrefix=dirPrefix,
                                             run_name=f'{filename}_play{i}_run{run}_train')
            else:
                rsum, foodEaten = simulation(env, agent, test=False)
        for test in range(nb_test):
            if animation and test == nb_test - 1:
                rsum, foodEaten = simulation(env, agent, test=True, r=r, save_animation=True,
                                             dirPrefix=dirPrefix,
                                             run_name=f'{filename}_play{i}_test{test}')
            else:
                rsum, foodEaten = simulation(env, agent, test=True)
            meanRsum += rsum
            meanFoodEaten += foodEaten

        meanRsum /= nb_test
        meanFoodEaten /= nb_test

        cur_result = f'{i + 1},{meanRsum},{meanFoodEaten}\n'
        print(f'\n{cur_result}')
        csvFile.write(cur_result)

        if (i + 1) % freqSave == 0 or (i + 1) == nb_play:
            agent.save(f'{dirPrefix}/QAgent/save_{i + 1}_{filename}')
    csvFile.close()


def simulation(env, agent, test=False, r=False, save_animation=False, dirPrefix='', run_name='', save_animation_orientation=False):
    agent.test = test
    rsum = 0
    ob = env.reset()
    agent.env = env
    step_count = 0
    path_save = ''
    if save_animation or save_animation_orientation:
        path_save = f'{dirPrefix}animation/{run_name}/'
        if not os.path.exists(f'{dirPrefix}animation'):
            os.mkdir(f'{dirPrefix}animation')
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        if save_animation_orientation:
            animateWithOrientation(env, path_save + 'frame_0.png', 0)
        else:
            animate(env, path_save + 'frame_0.png')

    while True:
        action = agent.act(ob)
        new_ob, reward, done = env.step(action)
        if not test:
            agent.store(ob, new_ob, action, reward, done)
            agent.batchlearn()
        rsum += reward
        ob = new_ob
        step_count += 1
        if r:
            render(env)
        if save_animation:
            animate(env, path_save + f'frame_{step_count}.png')
        if save_animation_orientation:
            animateWithOrientation(env, path_save + f'frame_{step_count}.png', step_count)
        if done:
            break

    if save_animation:
        with imageio.get_writer(path_save + f'animation_{step_count + 1}frames.gif', mode='I') as writer:
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
    save_plot(plot, filepath, env=game, showSensors=True)

def animateWithOrientation(game, filepath, stepCount):
    orienta = stepCount % 4
    plot = game.affPlot()
    save_plot(plot, filepath, env=game, showSensors=True, doOrientation=orienta)

if __name__ == '__main__':
    main(sys.argv)
