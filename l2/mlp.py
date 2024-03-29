import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, x, y, neurons=5, lr=0.01, activations=['sigmoid', 'linear']):
        self.inputs = x
        self.targets = y
        self.lr = lr
        self.neurons = neurons
        self.no_samples = self.inputs.shape[0]
        self.no_features = self.inputs.shape[1]  #input matrix colums.
        self.output_dim = self.targets.shape[1]
        self.w1 = np.random.rand(self.neurons, self.no_features)
        self.w2 = np.random.rand(self.output_dim, self.neurons)
        self.b1 = np.random.rand(self.neurons, 1)
        self.b2 = np.random.rand(self.output_dim, 1)
        self.activations = activations
        
    def sigmoid(self, n):
        return 1 / (1 + np.exp(-1 * n) )

    def f(self, n, layer):
        func = self.activations[layer-1]
        if func == 'sigmoid': 
            return self.sigmoid(n)
        else:
            return n

    def f_dot(self, n, layer):
        f = np.zeros((n.shape[0], n.shape[0]))
        if self.activations[layer-1] == 'sigmoid':
            for i in range(n.shape[0]):
                f[i, i] = self.sigmoid(n[i, 0]) * (1 - self.sigmoid(n[i, 0]))
        else:
            for i in range(n.shape[0]):
                f[i, i] = 1
        return f

    def J(self):
        j = 0
        for i in range(self.no_samples):
            sample = self.inputs[i, :][:, np.newaxis]  
            target = self.targets[i, :][:, np.newaxis]
            n1 = np.dot(self.w1, sample) + self.b1
            a1 = self.f(n1, 1)
            n2 = np.dot(self.w2, a1) + self.b2
            a2 = self.f(n2, 2)
            e = target - a2
            j += np.dot(e.transpose(), e)
        return j/self.no_samples

    def fit(self, epochs=100, verbos=True):
        for z in range(epochs):
            for _ in range(self.no_samples):
                index = np.random.randint(0, self.no_samples)
                sample = self.inputs[index, :][:, np.newaxis]  #picking a random sample
                target = self.targets[index, :][:, np.newaxis]
                #forward prop
                n1 = np.dot(self.w1, sample) + self.b1
                a1 = self.f(n1, 1)
                n2 = np.dot(self.w2, a1) + self.b2
                a2 = self.f(n2, 2)
                #back prop
                s2 = -2 * np.dot( self.f_dot(n2, 2), (target - a2) )
                s1 = np.dot( np.dot(self.f_dot(n1, 1),  self.w2.transpose()) , s2 )
                self.w1 += -1 * self.lr * np.dot(s1, sample.transpose())
                self.w2 += -1 * self.lr * np.dot(s2, a1.transpose())
                self.b1 += -1 * self.lr * s1
                self.b2 += -1 * self.lr * s2
            if verbos:
                print("epoch ", z+1, ", cost : ", self.J()[0][0])
    
    def predict(self, x):
        out = np.zeros_like(self.targets)
        for i in range(x.shape[0]):
            sample = x[i, :][:, np.newaxis]  
            n1 = np.dot(self.w1, sample) + self.b1
            a1 = self.f(n1, 1)
            n2 = np.dot(self.w2, a1) + self.b2
            a2 = self.f(n2, 2)
            out[i, 0] = a2[0, 0]
        return out


#data set
x = np.linspace(-2, 2, 21)[:, np.newaxis]  #shape: (21, 1)
y = 1 + np.sin((np.pi/2)*x)                #shape: (21, 1)
x2 = np.linspace(-3, 3, 61)[:, np.newaxis]
y2 = x2*x2*x2 + 4*np.sin(1.5*np.pi*x2)


nn = MLP(x2, y2, neurons=35)
nn.fit(epochs=1300, verbos=False)
out = nn.predict(x2)
plt.figure()
plt.plot(x2, y2, 'r', x2, out, 'b')
plt.show()

