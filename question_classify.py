import tensorflow as tf
import tensorflow.keras as keras

# constants

BATCH_SIZE      = 32
SEED            = 42
MAX_FEATURES    = 10000
SEQUENCE_LENGTH = 512
AUTOTUNE        = tf.data.experimental.AUTOTUNE
EMBEDDING_DIM   = 128
EPOCHS          = 20

# make the text vectorisation layer

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens  = MAX_FEATURES,
    output_mode = 'int',
    output_sequence_length = SEQUENCE_LENGTH )

# functions

def vectorize_text(text, label):

	text = tf.expand_dims(text, -1)
	return vectorize_layer(text), label

def get_text_dataset(directory, subset, split):

	return keras.preprocessing.text_dataset_from_directory(
		directory, 
		batch_size = BATCH_SIZE, 
		validation_split = split, 
		subset = subset, 
		seed = SEED )

# prepare training, validation and test datasets

raw_training_dataset   = get_text_dataset("train", "training",   0.2)
raw_validation_dataset = get_text_dataset("train", "validation", 0.2)
raw_test_dataset       = get_text_dataset("test",  None, None)

# set the vectorize_layer's vocabulary based on the training set

training_text = raw_training_dataset.map( lambda text, label: text )
vectorize_layer.adapt( training_text )

# vectorise the datasets

training_dataset   = raw_training_dataset.map( vectorize_text )
validation_dataset = raw_validation_dataset.map( vectorize_text )
test_dataset       = raw_test_dataset.map( vectorize_text )

# cache the datasets to improve performance

training_dataset   = training_dataset.cache().prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size = AUTOTUNE)
test_dataset       = test_dataset.cache().prefetch(buffer_size = AUTOTUNE)

# build, compile and train model

model = tf.keras.Sequential( [
	keras.layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
	keras.layers.Dropout(0.2),
	keras.layers.GlobalAveragePooling1D(),
	keras.layers.Dropout(0.2),
	keras.layers.Dense(4)] )

model.compile(
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer = 'adam',
	metrics = ["accuracy"] )

history = model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = EPOCHS )
