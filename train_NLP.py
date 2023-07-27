# import required packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalAveragePooling1D
from keras import Sequential
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_dataset(path, types):
    texts, labels = [],[]
    for index,label in enumerate(types):
        for fname in glob.glob(os.path.join(path, label, '*.*')):
            texts.append(open(fname, 'r', encoding="utf8").read())
            labels.append(index)
    return texts, np.array(labels).astype(np.int64)


if __name__ == "__main__": 

	# Load your training data
	path ='./data/aclImdb/'
	types = ['neg','pos']
	x_train, y_train = create_dataset(f'{path}train', types)

	# Train your network
	vocab_size = 20000    
	max_length = 400     
	trunc_type = 'post'
	oov_tok = "<OOV>"
	num_epochs = 2

	tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) # Initialized the tokenizer
	tokenizer.fit_on_texts(x_train)  # applying tokenizer on input
	word_index = tokenizer.word_index   # obtaining word index
	sequences = tokenizer.texts_to_sequences(x_train)    # converting input to word_indexes
	xtrain_padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type) # padding

	pickle.dump(tokenizer, open("./data/tokenizer.pkl", 'wb')) # Save tokenizer to use in test file

	X_train, X_val, y_train, y_val = train_test_split(xtrain_padded, y_train, test_size=0.3, random_state=42)

	model_nlp = Sequential()
	model_nlp.add(Embedding(vocab_size, 16))
	model_nlp.add(Dropout(0.1))
	model_nlp.add(Conv1D(filters=16, kernel_size=2, padding='valid', activation='relu'))
	model_nlp.add(GlobalAveragePooling1D())
	model_nlp.add(Dropout(0.1))
	model_nlp.add(Dense(32, activation='relu'))
	model_nlp.add(Dropout(0.1))
	model_nlp.add(Dense(1, activation='sigmoid'))

	model_nlp.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

	history_nlp = model_nlp.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
	
	print('Training accuracy :', history_nlp.history['accuracy'][-1])
	print('Validation Accuracy :', history_nlp.history['val_accuracy'][-1])

	# Save your model
	model_nlp.save("./models/Group_30_NLP_model.h5")