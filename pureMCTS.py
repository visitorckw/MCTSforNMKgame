import math
import random
import copy

class Node:
    def __init__(self, x = -1, y = -1, color = 1, parent = None) -> None:
        self.W = 0
        self.N = 0
        self.terminal = False
        self.x = x
        self.y = y
        self.color = color
        self.parent = parent
        self.child = []
class Board:
    def __init__(self, N = 3, M = 3, K = 3) -> None:
        self.N = N
        self.M = M
        self.K = K
        self.grid = [[-1 for j in range(self.M)] for i in range(self.N)]
        self.round = 0
    def checkTerminal(self):
        played = 0
        dir = [[0,1], [1,0], [1,1], [1,-1]]
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == -1:
                    continue
                played += 1
                for d in range(len(dir)):
                    ctr = 1
                    k = 1
                    while True:
                        x = i + dir[d][0] * k
                        y = j + dir[d][1] * k
                        if not self.inBound(x, y):
                            break
                        if self.grid[i][j] == self.grid[x][y]:
                            ctr += 1
                        else:
                            break
                        k += 1
                    k = -1
                    while True:
                        x = i + dir[d][0] * k
                        y = j + dir[d][1] * k
                        if not self.canPlay(x, y):
                            break
                        if self.grid[i][j] == self.grid[x][y]:
                            ctr += 1
                        else:
                            break
                        k -= 1
                    if ctr == self.K:
                        return self.grid[i][j]
        if played == self.N * self.M:
            return 0.5 ## tie
        return -1 ## not finish
    def play(self, x : int, y : int, color : int) -> int:
        if self.grid[x][y] != -1:
            print('ERR: playing at the place already have a stone')
            return -1
        self.grid[x][y] = color
        return 0
    def inBound(self, x, y) -> bool:
        if x < 0 or y < 0 or x >= self.N or y >= self.M:
            return False
        return True
    def canPlay(self, x, y) -> bool:
        if not self.inBound(x, y):
            return False
        return self.grid[x][y] == -1
    def draw(self) -> None:
        print("BOARD:")
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == -1:
                    print(' ', end = ' ')
                if self.grid[i][j] == 0:
                    print('O', end = ' ')
                if self.grid[i][j] == 1:
                    print('X', end = ' ')
            print('')
        for i in range(self.M):
            print('-', end = '')
        print('')
class MCTS:
    def __init__(self, board : Board, color : int) -> None:
        self.c = math.sqrt(2)
        self.interations = 100
        self.board = copy.deepcopy(board)
        self.color = color
        self.root = Node(color=1-color)
    def selection(self, node : Node) -> None: ## finish
        node = self.root
        board = copy.deepcopy(self.board)
        while True:
            if node.terminal:
                result = board.checkTerminal()
                self.backpropagation(node, result)
                break
            if node.N == 0 or len(node.child) == 0:
                # print('start expand')
                # print('node N = ', node.N)
                # board.draw()
                self.expansion(board, node)
                # board.draw()
                # print('node terminal', node.terminal)
                # print('after expand')
                # print('node N = ', node.N)
                # if not node.terminal:
                #     print('child[0] N = ', node.child[0].N)
                break
            ucbMax = -1
            chose = -1
            for i in range(len(node.child)):
                child = node.child[i]
                ucb = child.W / child.N + self.c * math.sqrt(math.log(node.N) / child.N)
                if ucb > ucbMax:
                    ucbMax = ucb
                    chose = i
            if chose == -1:
                board.draw()
                print(node.x, node.y, node.N)
                print(len(node.child))
            node = node.child[chose]
            board.play(node.x, node.y, node.color)

    def expansion(self, board : Board, node : Node): # finish
        if board.checkTerminal() != -1:
            node.terminal = True
            result = board.checkTerminal()
            self.backpropagation(node, result)
            return
        color = 1 - node.color
        for i in range(board.N):
            for j in range(board.M):
                if not board.canPlay(i, j):
                    continue
                node.child.append(Node(i, j, color, node))
                newBoard = copy.deepcopy(board)
                newBoard.play(i, j, color)
                result = self.simulation(newBoard, 1 - color)
                self.backpropagation(node.child[-1], result)

    def simulation(self, board : Board, color : int): # finish
        while board.checkTerminal() == -1:
            actionSet = []
            for i in range(board.N):
                for j in range(board.M):
                    if not board.canPlay(i, j):
                        continue
                    actionSet.append([i, j])
            action = random.choice(actionSet)
            # print('random play', action)
            board.play(action[0], action[1], color)
            color = 1 - color
        return board.checkTerminal()
    def backpropagation(self, node : Node, result): # finish
        while node is not None:
            node.N += 1
            if result == 0.5:
                node.W += 0.5
            elif result == node.color:
                node.W += 1
            node = node.parent

    def play(self):
        for i in range(self.interations):
            print('step', str(i+1) + '/' + str(self.interations))
            self.selection(self.root)
        maxN = -1
        chose = -1
        for i in range(len(self.root.child)):
            if self.root.child[i].N > maxN:
                maxN = self.root.child[i].N
                chose = i
        return self.root.child[chose].x, self.root.child[chose].y

def main():
    Test = False
    if Test:
        board = Board()
        board.play(0,0,0)
        board.play(0,1,0)
        board.play(0,2,0)
        print(board.checkTerminal())
        return
    board = Board()
    color = 1
    while True:
        x = int(input(''))
        y = int(input(''))
        board.play(x, y, 1 - color)

        mcts = MCTS(board, color)
        x, y = mcts.play()
        for i in range(len(mcts.root.child)):
            node = mcts.root.child[i]
            print('action x, y, w, n, ev')
            print(node.x, node.y, node.W, node.N, node.W / node.N)
        print('play at', x, y)
        board.play(x, y, color)
        board.draw()
        
if __name__ == '__main__':
    main()