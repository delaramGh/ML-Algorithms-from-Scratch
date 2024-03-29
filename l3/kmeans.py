
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class Kmeans:
  def __init__(self, x, k, range1=0, range2=255):
    self.inputs = x
    self.labels = np.zeros_like(self.inputs)
    self.prv_labes = np.zeros_like(self.inputs)
    self.centers = np.random.randint(range1, range2, k)
    self.k = k

  def equal(self, a, b):
    for i in range(a.shape[0]):
      if a[i] != b[i]:
        return False
    return True

  def center_update(self):
    new_center = np.zeros_like(self.centers, dtype='float')
    for c in range(self.centers.shape[0]):
      index = 0
      center = 0
      tedad = 1
      for l in self.labels:
        index += 1
        if l == c+1:
          tedad += 1
          center += self.inputs[index-1]
      new_center[c] = center/tedad
    return new_center
  
  def labeling(self):
    labels = np.zeros_like(self.labels)
    for i in range(self.inputs.shape[0]):
      abs = []
      for j in range(self.k):
        abs.append(np.abs(self.inputs[i] - self.centers[j]))
      temp = np.array(abs)
      label = temp.argmin()
      labels[i] = label + 1
    return labels

  def fit(self):
    for i in range(40):
      # print(i, 'th time.')
      self.prv_labes = self.labels
      self.labels = self.labeling()
      if self.equal(self.prv_labes, self.labels):
        return True
      self.centers = self.center_update()

  def output(self, dim1, dim2):
    out = np.zeros_like(self.inputs, dtype='float')
    index = 0
    for l in self.labels:
      out[index] = self.centers[l-1]
      index += 1
    return np.reshape(out, (dim1, dim2))


img = Image.open('mha.jpg').resize((450, 800)).convert('L')
img_arr = np.array(img)
img_1dim = np.reshape(img_arr, (img_arr.shape[0]*img_arr.shape[1]))
plt.figure()
plt.imshow(np.reshape(img_1dim, (img_arr.shape[0], img_arr.shape[1])), cmap='gray')
plt.show()

# x = np.array([-1, 2, 22, 20, 3, 23, 10, 11, 0, 12, -19, 90, 100, -40])
# km = Kmeans(x, 3, 0, 20)
# km.fit()
km = Kmeans(img_1dim, 10)
km.fit()
out = km.output(800,450)
plt.figure()
plt.imshow(out, cmap='gray')
plt.show()

#___________________________________________________________________________________________
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = Image.open("jik.jpeg")
img_arr = np.array(img)
print(type(img_arr), img_arr.shape)
plt.figure()
plt.imshow(img_arr)
plt.show()

img_1dim = np.reshape(img_arr, img_arr.shape[0]*img_arr.shape[1]*img_arr.shape[2])
print('img 1dim: ', img_1dim.shape)
km = KMeans(2)
img4 = img_1dim[:, np.newaxis]
km.fit(img4)
# print(km.cluster_centers_)
# print(km.labels_)

c = []
for i in range(len(km.cluster_centers_)):
  c.append(int(km.cluster_centers_[i]))
print(c)

index = 0
for l in km.labels_:
  img4[index] = c[l]
  index += 1
img5 = np.reshape(img4, (img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
plt.figure()
plt.imshow(img5)
plt.show()

