import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class RBF:
  def __init__(self, x, y, nn, center=None, b=None):
    self.x = x
    self.y = y
    self.nn = nn
    self.input_dim = self.x.shape[1]
    self.no_samples = self.x.shape[0]
    self.center = self.Center_kmean()
    if b == None:
      self.b = 1
    else:
      self.b = b
  
  def Center_kmean(self):
    km = KMeans(self.nn)
    km.fit(self.x)
    return km.cluster_centers_
  
  def x_prime(self, x):
    xp = []
    for x in range(self.no_samples):
      x_p = []
      for c in range(self.nn):
        sum = 0
        for i in range(self.input_dim):
          sum += (self.x[x, i] - self.center[c, i])**2
        norm = sum * self.b * self.b
        x_p.append(norm)
      xp.append(x_p)  
    return np.exp(-np.array(xp))    

  def fit(self):
    a2 = self.x_prime(self.x)
    a2 = np.concatenate((a2, np.ones((a2.shape[0], 1))), axis=1)
    #Normal eq
    temp = np.linalg.inv(np.dot(a2.T, a2))
    self.w = np.dot(np.dot(temp, a2.T), self.y)
  
  def predict(self, x):
    a2 = self.x_prime(x)
    a2 = np.concatenate((a2, np.ones((a2.shape[0], 1))), axis=1)
    y = np.dot(a2, self.w)
    return y


#data set
x = np.linspace(-2, 2, 211)[:, np.newaxis]
y = np.sin((np.pi/2)*x)
print('x shape: ', x.shape, ', y shape: ', y.shape)
x2 = np.array([[0,0], [0,1], [1,0], [1,1]])
y2 = np.array([[0, 1, 1, 0]]).T

#rbf
rbf = RBF(x, y, 5, b=1)
rbf.fit()
out = rbf.predict(x)
plt.figure()
plt.plot(x, y, 'r', x, out, 'b--')
plt.show()

# a = np.array([2, 2, 2, 2, 2, 2])
# a = np.expand_dims(a, 1)
# o = np.ones((6, 1))
# b = np.concatenate((o, a), axis=0)
# print(b)
