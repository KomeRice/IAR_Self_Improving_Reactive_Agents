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



if __name__ == '__main__':
	main(sys.argv)