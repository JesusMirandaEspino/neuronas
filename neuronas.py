import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar los tamaños de las capas y los pesos
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializar los pesos para la capa oculta y la capa de salida
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Inicializar los sesgos para la capa oculta y la capa de salida
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Propagación hacia adelante desde la capa de entrada a la capa oculta
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Propagación hacia adelante desde la capa oculta a la capa de salida
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        
        return self.output
    
    def backward(self, X, y, output):
        # Calcular el error en la capa de salida
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)
        
        # Calcular el error en la capa oculta
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        
        # Actualizar los pesos y los sesgos
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output)
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True)
        self.weights_input_hidden += X.T.dot(delta_hidden)
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True)
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

# Datos de entrada y salida
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[1], [0], [0], [1]])

# Crear y entrenar la red neuronal
input_size = 2
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs=10000)

# Realizar predicciones
predictions = nn.forward(X)
print("Predicciones finales:")
print(predictions)