import math
import random
import copy
import numpy as np
from keras.layers import Input
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
import keras
from tensorflow import transpose
from keras.regularizers import l2

N = 8
M = 8
K = 5

class Network_Model:
    def __init__(self) -> None:

        self.InputTensor = Input(shape=(N, M, 4))
        # self.H1 = Dense(10, activation='relu')(self.InputTensor)
        self.H2 = Conv2D(32, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.InputTensor)
        self.H2 = Conv2D(64, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.H2)
        self.H2 = Conv2D(128, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.H2)
        self.H2 = Conv2D(4, 1, 1, activation = 'relu', kernel_regularizer=l2(1e-4))(self.H2)
        self.H3 = Flatten()(self.H2)
        self.strategyOutput = Dense(N * M, activation='softmax')(self.H3)
        self.valueOutput = Dense(1, activation='tanh')(self.H3)
        self.model = keras.Model(
            inputs=[self.InputTensor],
            outputs=[self.strategyOutput,self.valueOutput ]
        )
        self.model.compile(optimizer = 'adam', loss = ['categorical_crossentropy', 'mean_squared_error'])
        self.model.summary()
    def fit(self, train_X, train_Y):
        train_X = np.array(train_X)
        train_Y = [np.array(train_Y[0]), np.array(train_Y[1])]
        self.model.fit(train_X, train_Y)
    def predict(self, feature):
        return self.model.predict(feature)

class Node:
    def __init__(self, x = -1, y = -1, color = 1, parent = None, P = 0.0) -> None:
        self.W = 0
        self.N = 0
        self.P = P
        self.Q = 0
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
        self.last_action = [-1, -1]
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
            print('ERR: playing at the (' + str(x) + ', ' + str(y)+ ' already have a stone')
            self.draw()
            return -1
        self.grid[x][y] = color
        self.last_action = [x, y]
        self.round += 1
        return 0
    def inBound(self, x, y) -> bool:
        if x < 0 or y < 0 or x >= self.N or y >= self.M:
            return False
        return True
    def canPlay(self, x, y) -> bool:
        if not self.inBound(x, y):
            return False
        return self.grid[x][y] == -1
    def getFeatures(self):
        feature = np.array([[[0.0 for j in range(self.N)] for i in range(self.M)] for k in range(4)])
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] != -1:
                    feature[self.grid[i][j]][i][j] = 1.0
        if self.last_action[0] != -1:
            feature[2][self.last_action[0]][self.last_action[1]] = 1.0
        if self.round % 2:
            for i in range(self.N):
                for j in range(self.M):
                    feature[3][i][j] = 1.0
        return transpose(feature, (1,2,0))
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
    def __init__(self, board : Board, color : int, model : Network_Model) -> None:
        self.c = math.sqrt(2)
        self.epsilon = 0.25
        self.c_puct = 5
        self.interations = 400
        self.board = copy.deepcopy(board)
        self.color = color
        self.root = Node(color=1-color)
        self.model = model
        self.trainX = []
        self.trainY = []
    def selection(self, node : Node, board : Board) -> None: ## finish
        # node = self.root
        # board = copy.deepcopy(self.board)
        board = copy.deepcopy(board)
        while True:
            if node.terminal:
                result = board.checkTerminal()
                z = 0
                if result == 0:
                    z = 1
                elif result == 1:
                    z = -1
                self.backpropagation(node, z)
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
            ucbMax = -1e9
            chose = -1
            for i in range(len(node.child)):
                child = node.child[i]
                ucb = child.Q + self.c_puct * child.P * math.sqrt(node.N) / (child.N + 1)
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
            # print('Terminal node')
            # board.draw()
            # print('')
            node.terminal = True
            result = board.checkTerminal()
            z = 0
            if result == 0:
                z = 1
            elif result == 1:
                z = -1
            self.backpropagation(node, z)
            return
        # print('input shape = ', np.array([board.getFeatures()]))
        y = self.model.predict(np.array([board.getFeatures()]))
        # print('expand')
        # print('y = ', y)
        color = 1 - node.color
        for i in range(board.N):
            for j in range(board.M):
                if not board.canPlay(i, j):
                    continue
                child_idx = i * board.M + j
                # print('P = ', y[0][0][child_idx])
                node.child.append(Node(i, j, color, node, y[0][0][child_idx]))
        # print('V = ', y[1][0][0])
        self.backpropagation(node, y[1][0][0])

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
            z = result
            if node.color == 1:
                z *= -1
            node.W += z
            node.Q = node.W / node.N
            node = node.parent

    def train(self, games = 1000):
        for game in range(games):
            train_X = []
            train_Y = [],[]
            print('training games: ' + str(game+1) + '/' + str(games))
            board = copy.deepcopy(self.board)
            node = self.root
            while board.checkTerminal() == -1:
                train_X.append(board.getFeatures())
                for iter in range(self.interations):
                    self.selection(node, board)
                target_y = np.zeros(board.N * board.M)
                strategy = []
                actionSet = []
                for i in range(len(node.child)):
                    idx = node.child[i].x * board.M + node.child[i].y
                    target_y[idx] = node.child[i].N / (node.N - 1)
                    actionSet.append(i)
                    strategy.append(target_y[idx])
                train_Y[0].append(target_y)
                # print('node N = ', node.N)
                # for i in range(len(node.child)):
                #     print('child i N = ', node.child[i].N)
                strategy = np.array(strategy)
                # print(strategy.sum())
                strategy = self.epsilon * np.random.dirichlet(0.3*np.ones(len(strategy))) + (1 - self.epsilon) * np.array(strategy)
                # print(strategy.sum())
                if len(actionSet) == 0:
                    print("ERR of actionSet")
                    board.draw()
                    print(actionSet)
                    print(len(node.child))
                    print(node.terminal)
                    print(board.checkTerminal())
                action = np.random.choice(actionSet, p=strategy) # 將要玩的子節點編號
                board.play(node.child[action].x, node.child[action].y, board.round % 2)
                print('play at ', node.child[action].x, node.child[action].y)
                tmp = []
                for i in actionSet:
                    tmp.append([node.child[i].x, node.child[i].y])
                print(tmp)
                board.draw()
                print('')
                node = node.child[action]
            z = 0
            result = board.checkTerminal()
            if result == 0:
                z = 1
            elif result == 1:
                z = -1
            for i in range(len(train_Y[0])):
                train_Y[1].append([z])
            self.model.fit(train_X, train_Y)
            self.save_weights('cnn_weight.h5')

    def play(self):
        for i in range(self.interations):
            print('step', str(i+1) + '/' + str(self.interations))
            self.selection(self.root, self.board)
        maxN = -1
        chose = -1
        for i in range(len(self.root.child)):
            if self.root.child[i].N > maxN:
                maxN = self.root.child[i].N
                chose = i
        return self.root.child[chose].x, self.root.child[chose].y
    def save_weights(self, filename):
        self.model.model.save_weights(filename)
    def load_weights(self, filename):
        self.model.model.load_weights(filename)

def main():
    board = Board()
    color = 1
    while True:
        x = int(input(''))
        y = int(input(''))
        board.play(x, y, 1 - color)

        mcts = MCTS(board, color, Network_Model())
        mcts.load_weights('cnn_weight.h5')
        x, y = mcts.play()
        for i in range(len(mcts.root.child)):
            node = mcts.root.child[i]
            print('action x, y, w, n, ev')
            print(node.x, node.y, node.W, node.N, node.W / node.N)
        print('play at', x, y)
        board.play(x, y, color)
        board.draw()
        
if __name__ == '__main__':
    Train = True
    Test = False

    if Train:
        mcts = MCTS(Board(8,8,5), 0, Network_Model())
        mcts.train(1000)
        mcts.save_weights('cnn_weight.h5')
        main()

    if Test:
        model = Network_Model()
        feature = np.array([[[0 for j in range(3)] for i in range(3)] for k in range(4)])
        input = transpose(feature, (1,2,0))
        print(input)
        input = np.array([input])
        y = model.predict(input)
        print(y)
        # print(y[0].shape)
        # print(y[1].shape)
        print(len(y))
        print(len(y[0]))
        print(len(y[1]))
        print(len(y[0][0]))
        print(len(y[1][0]))
        
