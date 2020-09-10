import tensorflow as tf 
from model import model 


model = model() 

optimizer = tf.keras.optimizers.SGD(momentum = 0.9, lr = 0.045)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics= ['accuracy'])

model.summary()





