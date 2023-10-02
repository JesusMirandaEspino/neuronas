import tensorflow as tf
import tensorflow_datasets as tfds

class redImagen:
    def __init__(self):
        # Cargar MNIST
        (self.lista_entrenar, self.lista_test), self.info = tfds.load(
            'mnist',
            split=['train', 'test'],
            with_info=True,
            as_supervised=True,
        )

        # Preprocesar
        def preprocesar(imagen, etiqueta):
            imagen = tf.image.convert_image_dtype(imagen, tf.float32)
            imagen = tf.image.resize(imagen, (28, 28))
            return imagen, etiqueta

        TAMANO = 64
        self.lista_entrenar = self.lista_entrenar.map(preprocesar).batch(TAMANO)
        self.lista_test = self.lista_test.map(preprocesar).batch(TAMANO)

        # Crear
        self.modelo = self._construir_modelo()
    
    def probar(self):
        self.entrenar(epoc=10)
        self.evaluar()
        self.guardar_modelo()
        
    def entrenar(self, epoc=10):
        self.modelo.fit(self.lista_entrenar, epochs=epoc)

    def _construir_modelo(self):
        modelo = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return modelo

    def guardar_modelo(self, nombre='red.h5'):
        self.modelo.save(nombre)
        

    def evaluar(self):
        perdida_prueba, pre_test = self.modelo.evaluate(self.lista_test)
        print(f'Precisión las pruebas: {pre_test * 100:.2f}%')


red = redImagen()
red.probar()
