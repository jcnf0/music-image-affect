import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import pandas as pd



## variables
taille_batch = 32
main_path= "C:\python\PAr"
va_path=main_path+"\DEAM_Annotations\annotations_averaged_per_song\song_level"


## Traitement espace 2D Valence et Arousal
va=pd.read_csv(r'C:\python\PAr\DEAM_Annotations\annotations_averaged_per_song\song_level\static_annotations_averaged_songs_1_2000.csv')

valence=va[' valence_mean']
arousal=va[' arousal_mean']

valence=tf.convert_to_tensor(valence)
arousal=tf.convert_to_tensor(arousal)

#agglomère en un espace 2D
espace = tf.stack([valence, arousal])
espace = tf.transpose(espace)

#divise en batch
espace = tf.data.Dataset.from_tensor_slices(espace)



## Traitement musique

# dataset avec path to music
music = pd.read_csv('C:\\python\\PAr\\DEAM_audio\\file_paths_45.csv')
music = music['Path']
music = tf.convert_to_tensor(music)
music = tf.data.Dataset.from_tensor_slices(music)

# charge musiques
def load_and_preprocess_audio(path):
  # Load the audio and the sample rate
  aud = tfio.audio.AudioIOTensor(path, dtype='float32')

  #Convert to tensor, transform to mono and keep 1s to 44s
  aud = aud.to_tensor()[44100:1940400, 0]

  #Transform the sample rate from 44.1khz to 16khz
  aud = tfio.audio.resample(aud, rate_in=44100, rate_out = 1000)


  return aud

music = music.map(load_and_preprocess_audio)


dataset = tf.data.Dataset.zip((music, espace)).batch(taille_batch)



# Division entrainement/test
test_data=dataset.take(12)
train_data=dataset.skip(12)


input_shape = (1000*43, 1)

## Modèle
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])


# Compilation modèle avec fonction de perte, l'optimiseur et les métriques
model.compile(loss='mse', optimizer='adam', metrics=['mae'], run_eagerly=True)

# Entraînement
model.fit(train_data, epochs=50, batch_size = taille_batch, validation_data=(test_data), verbose=1)

# Sauvegarde
model.save('music_regression_model.h5')

# Evaluation performance
test_loss, test_mae = model.evaluate(test_data)


# # Faites des prédictions sur de nouvelles données en utilisant predict()
# predictions = model.predict(X_new_data)