# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:47:31 2019

@author: Lord Tanmay
"""
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, SpatialDropout1D, Bidirectional
from keras.layers import Conv1D, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model

with open("data_array.pkl","rb") as file:
    input_train = pickle.load(file)
    
EMBEDDING_FILE='../Glove Data/glove.6B.50d.txt'
MODEL_WEIGHTS_FILE = 'toxic_model.h5'

embed_size = 50
max_features = 20000
maxlen = 100

list_sentences_train = input_train["comment_text"].fillna("_na_").values
y = input_train['target'].values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE,'rb'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: 
        continue 
    embedding_vector = embeddings_index.get(word) 
    
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector 
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
preds = Dense(1, activation="relu")(x)

model = Model(inp, preds)
model.compile(loss='mse',optimizer='nadam')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00000001)

callbacks = [learning_rate_reduction,EarlyStopping('val_loss', patience=3), ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]

history = model.fit(X_t, y, batch_size=128, epochs=20, validation_split=0.2, callbacks=callbacks)

model.save("toxic_model.h5")
model = load_model("toxic_model.h5")
