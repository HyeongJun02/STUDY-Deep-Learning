import tensorflow as tf

def build_CNN_model(self):
    input_layer = tf.keras.Input(shape=[self.img_shape_x, self.img_shape_y, 1, ])

    hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),
    padding='valid', activation='relu')(input_layer)
    hidden_layer = tf.keras.layers.MaxPooling2D((2, 2))(hidden_layer)

    hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
    padding='valid', activation='relu')(hidden_layer)
    hidden_layer = tf.keras.layers.MaxPooling2D((2, 2))(hidden_layer)

    hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(units=64, activation='relu')(hidden_layer)

    output = tf.keras.layers.Dense(units=self.num_labels, activation='softmax')(hidden_layer)
    
    classifier_model = tf.keras.Model(inputs=input_layer, outputs=output)
    classifier_model.summary()
    
    opt_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_cross_e = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    classifier_model.compile(optimizer=opt_alg, loss=loss_cross_e, metrics=['accuracy'])
    self.classifier = classifier_model