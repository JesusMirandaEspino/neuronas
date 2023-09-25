import numpy as np


class Neurona:

    def  __init__(self):
        # Listas de datos
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([1, 0, 0, 1])


    # Activación tipo escalón
    def pasos(self, z):
        return 1 if z >= 0 else 0

    # Perceptrón
    def perceptron(self, rate=0.1, ep=100):
        ejemplos, caracteristicas = self.X.shape
        peso = np.zeros(caracteristicas)
        bias = 0

        for _ in range(ep):
            for i in range(ejemplos):
                y_p = self.pasos(np.dot(self.X[i], peso) + bias)
                err = self.y[i] - y_p
                peso += rate * err * self.X[i]
                bias += rate * err

        return peso, bias

    def iniciar(self):

        # Entrenar
        peso, bias = self.perceptron()
        
        # Predicciones
        resultP = self.prediccion(peso, bias)

        # Resultados
        print("Entrada   Deseado   Prediccion")
        for i in range(self.X.shape[0]):
            print(f"  {self.X[i]}       {self.y[i]}         {resultP[i]}")

    # Red neuronal
    def prediccion(self, peso, bias):
        listaP = []
        for i in range(self.X.shape[0]):
            y_d = self.pasos(np.dot(self.X[i], peso) + bias)
            listaP.append(y_d)
        return np.array(listaP)


comenzar = Neurona()
comenzar.iniciar()