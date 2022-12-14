import numpy as np


class LinearRegression:

    def __init__(self, x, y, alpha=3e-2, max_epochs=10, epsilon=1e-2,
                 batch_size=10):
        self.x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        self.y = y
        self.alpha = alpha
        self.maxEpochs = max_epochs
        self.epsilon = epsilon
        self.batchSize = batch_size

        self.history = {
            'theta': [],
            'cost': [],
        }

        self.theta = np.random.normal(loc=0.0, scale=1.0, size=(self.x.shape[1], 1))
        self.history['theta'].append(self.theta.squeeze().tolist())

        initialPrediction = self.getPrediction(self.x, self.theta)
        initialCost = self.getCost(initialPrediction, self.y)
        self.history['cost'].append(initialCost)

        self.prevTheta = self.theta
        self.printStats()

    def printStats(self):
        print('=' * 80)
        print('[INFO]\t\tHyperparameters for Linear Regression')
        print('=' * 80)
        print(f'[INFO] Learning Rate: {self.alpha}')
        print(f'[INFO] Mini Batch Size: {self.batchSize}')
        print(f'[INFO] Maximum Epochs: {self.maxEpochs}')
        print(f'[INFO] Epsilon for checking convergence: {self.epsilon}')
        print(f'[INFO] Starting value of theta: {self.theta.tolist()} | {self.theta.shape}')
        print(f'[INFO] Shape of x data with ones: {self.x.shape}')
        print(f'[INFO] Shape of y data: {self.y.shape}')
        print(f'[INFO] Initial cost: {self.history["cost"][0]}')
        print('=' * 80)

    def runGradientDescent(self):

        xBatches = np.array_split(self.x, self.x.shape[0] // self.batchSize)
        yBatches = np.array_split(self.y, self.y.shape[0] // self.batchSize)

        for i in range(self.maxEpochs):
            for j, (x, y) in enumerate(zip(xBatches, yBatches)):
                # keeping track of prev theta for checking convergence
                self.prevTheta = self.theta

                h = self.getPrediction(x, self.theta)
                cost = self.getCost(h, y)
                gradients = (1 / x.shape[0]) * np.dot(np.transpose(x), (h - y))
                self.theta = self.theta - self.alpha * gradients

                # log metrics
                self.history['theta'].append(self.theta.squeeze().tolist())
                self.history['cost'].append(cost)

                if self.isConverged():
                    print(f'[INFO] Gradient Descent converged at Epoch: {i + 1}, iteration: {j + 1}')
                    break

            if self.isConverged():
                break

    def getThetaByNormalEquations(self):
        return np.dot(np.linalg.inv(np.dot(np.transpose(self.x), self.x)),
                      np.dot(np.transpose(self.x), self.y))

    def isConverged(self):
        return (abs(self.theta - self.prevTheta) <= self.epsilon).all()

    def getHistory(self):
        return self.history

    @staticmethod
    def getPrediction(x, theta):
        return np.dot(x, theta)

    @staticmethod
    def getCost(y_pred, y_true):
        m = y_pred.shape[0]
        return (1 / (2 * m)) * np.sum((y_pred - y_true)**2)