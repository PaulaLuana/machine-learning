import numpy as np

class Metrics:
  def MAE(self, y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

  def MSE(self, y_true, y_pred):
    return np.mean(pow((y_pred - y_true),2))

  def RMSE(self, y_true, y_pred):
    return np.sqrt(np.mean(pow((y_pred - y_true),2)))