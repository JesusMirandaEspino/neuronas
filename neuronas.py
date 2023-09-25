import numpy as np

class Neurona:
    def __init__(self):
        
        # Inicializar los Datos
        self.entrada = 2
        self.ocultar = 4
        self.salida = 1
        
        self.X = np.array([[0, 0], [0, 1],[1, 0],[1, 1]])
        self.y = np.array([[1], [0], [0], [1]])
        
        #  Preparar pesos para la capa oculta y la capa de salida
        self.pesos_oculto = np.random.randn(self.entrada, self.ocultar)
        self.pesos_salida = np.random.randn(self.ocultar, self.salida)
        
        # Preparar los sesgos para la capa oculta y la capa de salida
        self.bias_oculto = np.zeros((1, self.ocultar))
        self.bias_salida = np.zeros((1, self.salida))
        
    def comenzar(self):
        self.entrenar(rango=10000)
        pre = iniciar.continuar()
        print("Resultados:")
        print(pre)
        
    def entrenar(self, rango):
        for r in range(rango):
            salida = self.continuar()
            self.atras(salida)
            
            if r % 1000 == 0:
                loss = np.mean(np.square(self.y - salida))
                print(f"Rango {r}, Perdida: {loss}")
    
    def continuar(self):
        # Propagación hacia adelante desde la capa de entrada a la capa oculta
        self.entrada_oculta = np.dot(self.X, self.pesos_oculto) + self.bias_oculto
        self.salida_oculta = self.sigmoid(self.entrada_oculta)
        
        # Propagación hacia adelante desde la capa oculta a la capa de salida
        self.entrada_salida = np.dot(self.salida_oculta, self.pesos_salida) + self.bias_salida
        self.salida = self.sigmoid(self.entrada_salida)
        
        return self.salida
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivado(self, x):
        return x * (1 - x)

    def atras(self, output):
        # Error en salida
        err = self.y - output
        delta = err * self.sigmoid_derivado(output)
        
        # Error oculto
        err_oculto = delta.dot(self.pesos_salida.T)
        deltaH = err_oculto * self.sigmoid_derivado(self.salida_oculta)
        
        # Actualizar los pesos y los sesgos
        self.pesos_salida += self.salida_oculta.T.dot(delta)
        self.bias_salida += np.sum(delta, axis=0, keepdims=True)
        self.pesos_oculto += self.X.T.dot(deltaH)
        self.bias_oculto += np.sum(deltaH, axis=0, keepdims=True)
    

iniciar = Neurona()
iniciar.comenzar()


