import numpy as np
from scipy.spatial import distance
from statistics import mode

class KNeighborsClassifier:
  def __init__(self):
    self.train = []
    self.labels = []
    self.k = 5
  
  def fit(self, x, y, k = 5):
    self.train = x
    self.labels = y
    self.k = k


  def predict(self, x):
    y_pred = list()
    for elemento in x: #para cada x preciso descobrir a menor distancia e a label
      distancias = [distance.euclidean(x_i, elemento) for x_i in self.train]
      indices_sort = np.argsort(distancias)
      rotulos = self.labels[indices_sort]
      rotulos_ = rotulos[0:self.k]
      y_pred.append(mode(rotulos_))
    return y_pred
