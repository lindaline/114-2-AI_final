import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Dropout,Activation,BatchNormalization,ReLU,add
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Input,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras import optimizers,regularizers
from matplotlib import image, pyplot as plt

model_name = 'resnet18'
folder_path = 'database/train'
cls_list = []

for folder in os.listdir(folder_path):
    cls_list.append(folder)

def block(x,out_filters,k_size=(3,3),downsample=0):
  if(downsample==1):
    x1 = Conv2D(filters=out_filters,kernel_size=(1,1),strides=(2,2),padding='same')(x)
    x1 = BatchNormalization()(x1)
    x = Conv2D(filters=out_filters,kernel_size=k_size,strides=(2,2),padding='same')(x)
  else:
    x1 = x
    x = Conv2D(filters=out_filters,kernel_size=k_size,strides=(1,1),padding='same')(x)

  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(filters=out_filters,kernel_size=k_size,strides=(1,1),padding='same')(x)
  x = BatchNormalization()(x)
  x = add([x1,x])
  x = ReLU()(x)
  return x

def resnet18(x,num_class=2):
  x = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding='same')(x)
  x = MaxPooling2D()(x)
  x = block(x,out_filters=64,downsample=0)
  x = block(x,out_filters=64,downsample=0)
  x = block(x,out_filters=128,downsample=1)
  x = block(x,out_filters=128,downsample=0)
  x = block(x,out_filters=256,downsample=1)
  x = block(x,out_filters=256,downsample=0)
  x = block(x,out_filters=512,downsample=1)
  x = block(x,out_filters=512,downsample=0)
  x = GlobalAveragePooling2D()(x)
  x = Dense(num_class,activation='softmax')(x)
  return x

def show_history(history):
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train','Test'],loc='upper left')
  plt.savefig(f'./{model_name}_acc.png')

  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train','Test'],loc='upper left')
  plt.savefig(f'./{model_name}_loss.png')

  if 'learning_rate' in history.history:
    plt.plot(history.history['learning_rate'])
    plt.title('Model lr'); plt.ylabel('Learning Rate'); plt.xlabel('Epoch')
    plt.legend(['Train'],loc='upper left')
    plt.savefig(f'./{model_name}_lr.png')

def train(num_class):
  data_path = r'./database'
  image_size = (224,224)
  num_class = num_class
  batch_size = 32
  num_epochs = 20
  weights_final = f'./{model_name}.h5'

  # 重點：train 與 val 都要 rescale=1./255，把像素正規化到 0~1
  train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=10,
                    horizontal_flip=True,
                    fill_mode='nearest')
  train_batches = train_datagen.flow_from_directory(data_path+'/train',
                            target_size=image_size,
                            interpolation='bicubic',
                            class_mode='categorical',
                            shuffle=True,
                            batch_size=batch_size)
  valid_datagen = ImageDataGenerator(rescale=1./255)
  valid_batches = valid_datagen.flow_from_directory(data_path+'/val',
                            target_size=image_size,
                            interpolation='bicubic',
                            class_mode='categorical',
                            shuffle=False,
                            batch_size=batch_size)

  for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

  img_input = Input(shape=(image_size[0],image_size[1],3))
  output = resnet18(img_input, num_class=num_class)
  model = Model(img_input,output)
  print(model.summary())

  model.compile(optimizer=Adam(learning_rate=1e-3),
          loss='categorical_crossentropy',
          metrics=['acc'])
  csv_logger = CSVLogger(f'./{model_name}.csv')
  reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,min_lr=1e-6,verbose=1)
  earlystop = EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True,verbose=1)
  cbks = [csv_logger,reduce_lr,earlystop] 

  history = model.fit(train_batches,
          validation_data=valid_batches,
          epochs=num_epochs,
          callbacks=cbks)
  model.save(weights_final)

  show_history(history)

train(num_class=len(cls_list))