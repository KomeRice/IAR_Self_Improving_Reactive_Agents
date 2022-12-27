import random
import game
from customExceptions import GridReaderError


class GridReader:
    @staticmethod
    def readGrid(filePath):
        knownSymbols = {' ', 'O', 'E', 'I'}
        with open(filePath, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith('#')]

            try:
                nbRows = int(lines[0])
            except ValueError:
                raise GridReaderError(0, filePath, 'Could not convert to int')

            try:
                nbCols = int(lines[1])
            except ValueError:
                raise GridReaderError(1, filePath, 'Could not convert to int')

            try:
                nbFood = int(lines[2])
            except ValueError:
                raise GridReaderError(2, filePath, 'Could not convert to int')

            if len(lines[3:]) != nbRows:
                raise GridReaderError(1, filePath, 'Number of rows does not match given grid')

            grid = []

            for i in range(3, nbRows + 3):
                symbols = []
                for symbol in lines[i].split(','):
                    symbolStripped = symbol.strip('\n')
                    if symbolStripped == '':
                        symbols.append(' ')
                        continue
                    if symbolStripped in knownSymbols:
                        symbols.append(symbolStripped)

                if len(symbols) != nbCols:
                    raise GridReaderError(i, filePath, 'Invalid line length or symbol')
                grid.append(symbols)

        #print(f'Successfully read file {filePath}, generating game instance...')

        g = game.GameInstance(nbRows, nbCols, nbFood)
        mainAgentAdded = False

        emptyTiles = []

        for y in range(nbRows):
            for x in range(nbCols):
                symbol = grid[y][x]
                if symbol == 'E':
                    g.addEnemyAgent(x, y)
                    g.initialEnemyPositions.append((x, y))
                elif symbol == 'I':
                    #if mainAgentAdded:
                        #print(f'Added multiple main agents, is this intended?')
                    g.addMainAgent(x, y)
                    g.initialMainAgentPosition.append((x, y))
                    mainAgentAdded = True
                elif symbol == 'O':
                    g.grid[y][x] = 'O'
                    g.initialWalls.append((x, y))
                else:
                    emptyTiles.append((x, y))

        g.initialEmptyTiles = emptyTiles

        if nbFood > len(emptyTiles):
            raise GridReaderError(2, filePath, 'Not enough empty space for food')

        g.initialFood = nbFood
        g.distributeFood()

        #print('Done.')

        return g
