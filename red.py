import tensorflow as tf
import tensorflow_datasets as tfds

class Red:
    def __init__(self):
        # Cargar la base de datos MNIST
        (self.train_dataset, self.test_dataset), self.info = tfds.load( 'mnist', split=['train', 'test'],with_info=True, as_supervised=True,)

        # Preprocesar los datos
        def pre(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)  # Normalizar las imágenes
            image = tf.image.grayscale_to_rgb(image)  # Convertir a 3 canales (RGB)
            image = tf.image.resize(image, (28, 28))  # Redimensionar a 28x28
            return image, label

        BATCH_SIZE = 64
        self.train_dataset = self.train_dataset.map(pre).batch(BATCH_SIZE)
        self.test_dataset = self.test_dataset.map(pre).batch(BATCH_SIZE)

        # Crear el modelo
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Compilar el modelo
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
    def iniciar(self):
        self.entrenar(epochs=10)
        self.evaluar()
        self.guardar_modelo()

    def evaluar(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)
        print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')
        
    def entrenar(self, epochs=10):
        self.model.fit(self.train_dataset, epochs=epochs)

    def guardar_modelo(self, filename='red.h5'):
        self.model.save(filename)


comprobar = Red()
comprobar.iniciar()