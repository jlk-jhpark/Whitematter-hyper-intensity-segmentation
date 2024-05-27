from utils import dice_coef, dice_coef_loss, get_unet, DataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


import os
import numpy as np

params = {'dim': (256,256),
          'batch_size': 4,
          'class_num': 1,
          'n_channels': 1,
          'shuffle': True}

partition_path = ''
partition = np.load(os.path.join(partition_path,'partion.npy'))
training_generator = DataGenerator(partition[0], **params)
validation_generator = DataGenerator(partition[1], **params)

learning_rate = 1e-3
epoch_num = 50
decay_rate = learning_rate/epoch_num
model = get_unet(256,256)

model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=dice_coef_loss, metrics=[dice_coef])


save_path = ''
model_name = 'model.h5'
model_checkpoint = ModelCheckpoint(os.path.join(save_path,model_name) , monitor='val_loss', save_best_only=True)

steps_e = len(partition[0])//8
steps_v = len(partition[1])//8

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    workers=6, steps_per_epoch=steps_e, validation_steps=steps_v, epochs=epoch_num, callbacks=[model_checkpoint])    
