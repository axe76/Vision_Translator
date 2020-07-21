from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from ocr import ocr
import numpy as np

res = ocr('simple1.jpg')
print(res)
#res = ' '.join(res)

def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
def encode_sequences(tokenizer, length, lines):
	X = tokenizer.texts_to_sequences(lines)
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

dataset = load_clean_sentences('english-german-both.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

model = load_model('my_model.h5')

print(res[0].lower())

for i in range(len(res)):
    text = np.array([res[i].lower()])
    
    text = encode_sequences(ger_tokenizer, ger_length, text)
    #print(text)
    
    translation = predict_sequence(model, eng_tokenizer, text)
    print(translation)


	







    
    
    

