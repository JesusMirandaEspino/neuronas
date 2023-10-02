import numpy as np

class Neuronas:
    def __init__(self, entradas, salidas, testing):
        # Función  sigmoide
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        # Derivada sigmoide
        self.derivada = lambda x: x * (1 - x)
        
        # Datos de entrada
        self.X = entradas

        # Salida esperada
        self.y = salidas
        
        # Testing
        self.testing = testing
        
        # Pesos y bias
        self.entrada_tamano = 2
        self.entrada_oculta = 3
        self.salida_tamano = 1
        
        # Pesos  capa oculta y  capa de salida
        self.in_ocult_tamano = np.random.uniform(size=(self.entrada_tamano, self.entrada_oculta))
        self.out_ocult_tamano = np.random.uniform(size=(self.entrada_oculta, self.salida_tamano))

    def entrenar(self, rango=0.1, epoca=10000):
        for epoch in range(epoca):
            # Feedforward
            capa_oculta_in = np.dot(self.X, self.in_ocult_tamano)
            capa_ocul_out = self.sigmoid(capa_oculta_in)
            s_capa_in = np.dot(capa_ocul_out, self.out_ocult_tamano)
            s_capa_s = self.sigmoid(s_capa_in)
        
            # Cálculo error
            err = self.y - s_capa_s
        
            # Retropropagación
            d_output = err * self.derivada(s_capa_s)
            capa_error = d_output.dot(self.out_ocult_tamano.T)
            capa_oc = capa_error * self.derivada(capa_ocul_out)
        
            # Actualización de pesos y bias
            self.out_ocult_tamano += capa_ocul_out.T.dot(d_output) * rango
            self.in_ocult_tamano += self.X.T.dot(capa_oc) * rango

    def preparar(self,X):
        capa_oculta_in = np.dot(X, self.in_ocult_tamano)
        capa_ocul_out = self.sigmoid(capa_oculta_in)
        s_capa_in = np.dot(capa_ocul_out, self.out_ocult_tamano)
        s_capa_s = self.sigmoid(s_capa_in)
        return s_capa_s



# Datos para usarse 
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

salidas = np.array([[1], [0], [0], [1]])
        
testing = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])



neural_network = Neuronas(entradas, salidas, testing)

# Entrenar la red neuronal
neural_network.entrenar()

# Resultado final después del entrenamiento
print("Entrenamiento:")
print(neural_network.preparar(neural_network.X))

entradas_test = neural_network.testing

print("Predicciones:")
for entrada in entradas_test:
    prediction = neural_network.preparar(entrada)
    print(entrada, "->", prediction)