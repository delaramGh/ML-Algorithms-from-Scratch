#hopfield
import numpy as np
import matplotlib.pyplot as plt

def hopImage(img_name):
  pass

class Hop:
  def __init__(self, X):  #X dim: no of features, no of samples
    self.a_k = None
    self.a_k_1 = None
    self.w =  np.dot(X, X.T) - X.shape[1]*np.eye(X.shape[0])  
  
  def predict(self, x, verbos=False):
    print('predict: ', x.dtype)
    print(x.T)
    self.a_k_1 = x
    for i in range(50):
      self.a_k = np.sign( np.dot(self.w, self.a_k_1) )
      # print(self.a_k.dtype)
      self.a_k[self.a_k == 0] = -1
      self.a_k = self.async_(self.a_k_1, self.a_k)
      if verbos:
        print("a(k) iter ", i, ':   ', self.a_k.T, 'diff: ', np.sum((self.a_k!=self.a_k_1).astype(np.int8)))
      if(np.all(self.a_k == self.a_k_1)):
        break
      self.a_k_1 = self.a_k
      
    return self.a_k

  def async_(self, a, b):
    ind = 0
    for i in range(a.shape[0]):
      if(a[i,0] != b[i,0]):
        ind = i
        break
    for i in range(ind+1, a.shape[0]):
      b[i, 0] = a[i, 0]
    return b
#_____________________________________________________________________
from PIL import Image
from copy import deepcopy

def hopImage(name):
  img = Image.open(name).resize((100, 100)).convert('L')
  img_arr = np.array(img, dtype=np.int16)
  img_arr = np.reshape(img_arr, (100 * 100))
  for i in range(len(img_arr)):
    if img_arr[i] >= 127:
      img_arr[i] = 1
    else:
      img_arr[i] = -1
  
  # print('hopImage: ', img_arr)
  return img_arr

def showHopImg(img__):
  img_ = img__.copy()
  for i in range(len(img_)):
    if img_[i] == 1:
      img_[i] = 255
    else:
      img_[i] = 0
  img = np.reshape(img_, (100, 100))
  # print('show', img)
  plt.figure()
  plt.imshow(img.astype(np.uint8), cmap='gray')
  plt.show()

def noisifyImg(name, percent):
  percent = 100 * percent
  img = hopImage(name)
  pixels = np.random.randint(0, 10000, (percent))
  for pixel in pixels:
    if img[pixel] == 1:
      img[pixel] = -1
    else:
      img[pixel] = 1
  return img
  
#______________________________________________________________________

# X = np.array([[1, 1, 1, 1], [-1, 1, 1, 1]]).T  #sotuni shape:(4, 2)
# hop = Hop(X)
# x = np.array([[1, 1, -1, 1]]).T
# print(hop.predict(x, 1))

# img1 = hopImage('chess.jpg')
# img2 = hopImage('girl.jfif')
img1 = hopImage('gilbert.jpeg')
img2 = hopImage('messi.jpg')
#Data set
X = np.concatenate((img1[:, np.newaxis], img2[:, np.newaxis]), axis=1)
# showHopImg(X[:,1])
print(X[:, 1])
hop2 = Hop(X)

noisy_img = noisifyImg('messi.jpg', 1)
showHopImg(noisy_img)
# n_img = deepcopy(noisy_img[:, np.newaxis])
# showHopImg(noisy_img)
# ans = hop2.predict(noisy_img[:, np.newaxis])
print(X[:, 0:1].dtype)
print(X[:, 1:2])
# ans = hop2.predict(X[:,0:1], 1)
ans = hop2.predict(noisy_img[:, np.newaxis], 1)
# print(X[:,0])
showHopImg(ans)

