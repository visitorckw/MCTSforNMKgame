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

N = 3
M = 3
K = 3

class Network_Model:
    def __init__(self) -> None:

        self.InputTensor = Input(shape=(N, M, 4))
        # self.H1 = Dense(10, activation='relu')(self.InputTensor)
        self.H2 = Conv2D(32, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.InputTensor)
        # self.H2 = Conv2D(64, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.H2)
        # self.H2 = Conv2D(128, 3, 3, activation = 'relu', kernel_regularizer=l2(1e-4), padding="same")(self.H2)
        self.H2 = Conv2D(9, 1, 1, activation = 'relu', kernel_regularizer=l2(1e-4))(self.H2)
        self.H3 = Flatten()(self.H2)
        self.strategyOutput = Dense(N * M, activation='softmax', kernel_regularizer=l2(1e-4))(self.H3)
        self.H4 = Conv2D(2, 1, 1, activation = 'relu', kernel_regularizer=l2(1e-4))(self.H2)
        self.H5 = Flatten()(self.H4)
        self.valueOutput = Dense(1, activation='tanh', kernel_regularizer=l2(1e-4))(self.H5)
        self.model = keras.Model(
            inputs=[self.InputTensor],
            outputs=[self.strategyOutput,self.valueOutput ]
        )
        self.model.compile(optimizer = 'adam', loss = ['categorical_crossentropy', 'mean_squared_error'])
        self.model.summary()
    def fit(self, train_X, train_Y):
        train_X = np.array(train_X)
        train_Y = [np.array(train_Y[0]), np.array(train_Y[1])]
        print('number of training data', len(train_X))
        self.model.fit(train_X, train_Y, epochs=5)
    def predict(self, feature):
        return self.model.predict(feature)

class Node:
    def __init__(self, x = -1, y = -1, color = 1, parent = None, P = 0.0, timestamp = 0, v = 0) -> None:
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
        self.timestamp = timestamp
        self.v = v
    def reset(self):
        self.W = 0
        self.N = 0
        self.Q = 0
class Board:
    def __init__(self, N = 3, M = 3, K = 3) -> None:
        self.N = N
        self.M = M
        self.K = K
        self.grid = [[-1 for j in range(self.M)] for i in range(self.N)]
        self.round = 0
        self.history = []
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
                        if not self.inBound(x, y):
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
        self.history.append([x, y])
        self.round += 1
        return 0
    def undo(self,) -> int:
        if len(self.history) == 0:
            print('ERR: Undoing while didn\'t have a stone')
            self.draw()
            return -1
        x = self.history[-1][0]
        y = self.history[-1][1]
        self.grid[x][y] = -1
        self.round -= 1
        self.history.pop()
        if self.round >= 1:
            self.last_action = self.history[-1]
        else:
            self.last_action = [-1, -1]
        return 0
    def inBound(self, x, y) -> bool:
        if x < 0 or y < 0 or x >= self.N or y >= self.M:
            return False
        return True
    def canPlay(self, x, y) -> bool:
        if not self.inBound(x, y):
            return False
        return self.grid[x][y] == -1
    def getFeatures(self, k = 0, flip = 0):
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
        for channel in range(4):
            if k:
                feature[channel] = np.rot90(feature[channel], k)
            if flip:
                feature[channel] = np.flipud(feature[channel])
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
        for i in range(self.M * 2):
            print('-', end = '')
        print('')
class MCTS:
    def __init__(self, board : Board, color : int, model : Network_Model) -> None:
        self.c = math.sqrt(2)
        self.epsilon = 0.25
        self.c_puct = 5
        self.temp = 1.0
        self.interations = 400
        self.board = copy.deepcopy(board)
        self.color = color
        self.root = Node(color=1-color)
        self.model = model
        self.train_X = []
        self.train_Y = [], []
        self.timestamp = 0
    def selection(self, node : Node, board : Board) -> None: ## finish
        # board = copy.deepcopy(board)
        while True:
            if node.timestamp != self.timestamp:
                node.reset()
                node.timestamp = self.timestamp
                self.backpropagation(node, node.v, board)
                break
            if node.terminal:
                result = board.checkTerminal()
                z = 0
                if result == 0:
                    z = 1
                elif result == 1:
                    z = -1
                self.backpropagation(node, z, board)
                break
            if node.N == 0 or len(node.child) == 0:
                self.expansion(board, node)
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
            node.terminal = True
            result = board.checkTerminal()
            z = 0
            if result == 0:
                z = 1
            elif result == 1:
                z = -1
            self.backpropagation(node, z, board)
            return
        y = self.model.predict(np.array([board.getFeatures()]))
        if not node.parent:
        # if False:
            noise = np.random.dirichlet(0.3 * np.ones(board.N * board.M))
        color = 1 - node.color
        for i in range(board.N):
            for j in range(board.M):
                if not board.canPlay(i, j):
                    continue
                child_idx = i * board.M + j
                if not node.parent:
                # if False:
                    node.child.append(Node(i, j, color, node, (1 - self.epsilon) * y[0][0][child_idx] + self.epsilon * noise[child_idx]))
                else:
                    node.child.append(Node(i, j, color, node, y[0][0][child_idx]))
        node.v = y[1][0][0]
        self.backpropagation(node, y[1][0][0], board)

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
    def backpropagation(self, node : Node, result, board : Board): # finish
        while node is not None:
            node.N += 1
            z = result
            if node.color == 1:
                z *= -1
            node.W += z
            node.Q = node.W / node.N
            node = node.parent
            if node is not None:
                board.undo()
    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        sum = np.sum(probs)
        probs /= sum
        return probs

    def train(self, games = 1000):
        for game in range(games):
            # train_X = []
            # train_Y = [],[]
            print('training games: ' + str(game+1) + '/' + str(games))
            board = copy.deepcopy(self.board)
            node = Node(color=1)
            self.timestamp = 0
            while board.checkTerminal() == -1:
                # train_X.append(board.getFeatures())
                for i in range(4):
                    for j in range(2):
                        self.train_X.append(board.getFeatures(i, j))
                for iter in range(self.interations):
                    self.selection(node, board)
                target_y = np.zeros(board.N * board.M)
                visits = [node.child[i].N for i in range(len(node.child))]
                actionSet = []

                if board.round < 1:
                    temp = 1.0
                else:
                    temp = 1e-2
                strategy = self.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
                for i in range(len(node.child)):
                    idx = node.child[i].x * board.M + node.child[i].y
                    actionSet.append(i)
                    target_y[idx] = strategy[i]
                sum = target_y.sum()
                for i in range(4):
                    for j in range(2):
                        y = np.array(target_y).reshape((board.N, board.M))
                        y = np.rot90(y, i)
                        if j:
                            y = np.flipud(y)
                        self.train_Y[0].append(y.reshape(board.N * board.M))
                strategy = np.array(strategy)
                # strategy = self.epsilon * np.random.dirichlet(0.3*np.ones(len(strategy))) + (1 - self.epsilon) * np.array(strategy)
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
                print('P = ', node.child[action].P)
                print('v = ', node.child[action].Q)
                # tmp = []
                # for i in actionSet:
                #     tmp.append([node.child[i].x, node.child[i].y])
                # print(tmp)
                board.draw()
                print('')
                node = node.child[action]
                node.parent = None
                noise = np.random.dirichlet(0.3 * np.ones(board.N * board.M))
                for i in range(len(node.child)):
                    idx = node.child[i].x * board.M + node.child[i].y
                    node.child[i].P = (1 - self.epsilon) * node.child[i].P + self.epsilon * noise[idx]
                self.timestamp += 1
                # node = Node(color=1 - board.round % 2)
            z = 0
            result = board.checkTerminal()
            if result == 0:
                z = 1
            elif result == 1:
                z = -1
            while len(self.train_Y[0]) > len(self.train_Y[1]):
                self.train_Y[1].append([z])
            print('z = ', z)
            if len(self.train_X) >= (5+9) // 2 * 8 * 5:
                self.model.fit(self.train_X, self.train_Y)
                self.save_weights('cnn_weight.h5')
                self.train_X = []
                self.train_Y = [], []

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
    board = Board(N, M, K)
    color = 0
    while True:
        # x = int(input(''))
        # y = int(input(''))
        # board.play(x, y, 1 - color)

        mcts = MCTS(board, color, Network_Model())
        mcts.load_weights('cnn_weight.h5')
        x, y = mcts.play()
        for i in range(len(mcts.root.child)):
            node = mcts.root.child[i]
            print('action x, y, w, n, ev, P')
            print(node.x, node.y, node.W, node.N, node.Q, node.P)
        print('play at', x, y)
        board.play(x, y, color)
        board.draw()

        x = int(input(''))
        y = int(input(''))
        board.play(x, y, 1 - color)
        
if __name__ == '__main__':
    Train = True
    Test = False

    if Train:
        mcts = MCTS(Board(N,M,K), 0, Network_Model())
        # mcts.load_weights('cnn_weight.h5')
        mcts.train(1000)
        mcts.save_weights('cnn_weight.h5')
    main()

    if Test:
        board = Board(8,8,5)
        board.play(3,4,0)
        board.play(4,3,1)
        board.draw()
        self = board
        print(self.last_action)
        board.undo()
        board.draw()
        print(self.last_action)
        board.undo()
        board.draw()
        print(self.last_action)
        board.undo()
        
