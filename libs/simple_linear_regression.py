import numpy as np

class SimpleLinearRegression:
    def fit(self, x, y):
        sum1 = np.sum((x - np.mean(x)) * (y - np.mean(y)))
        sum2 = np.sum((x - np.mean(x)) ** 2)
        self.w1 = sum1/sum2
        self.w0 = np.mean(y) - self.w1 * np.mean(x)

    def predict(self, x):
        return self.w0 + self.w1 * x