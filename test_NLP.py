# import required packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
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
	# Load your saved model
	model_nlp = keras.models.load_model("./models/Group_30_NLP_model.h5")

	# Load your testing data
	path ='./data/aclImdb/'	
	types = ['neg','pos']
	x_test, y_test = create_dataset(f'{path}test', types)

	# Preprocessing for test data
	tokenizer = pickle.load(open("./data/tokenizer.pkl", 'rb'))
	seq_test = tokenizer.texts_to_sequences(x_test) # tokenizing test data
	xtest_padded = pad_sequences(seq_test, maxlen=400) # padding

	# Run prediction on the test data and print the test accuracy
	predictions = model_nlp.predict(xtest_padded) # to check predicted values
	test_loss, test_acc = model_nlp.evaluate(xtest_padded,  y_test)
	print('Test accuracy is :', test_acc)