import pickle 
import streamlit as st   
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
with open('model_next_word.pkl','rb') as f:
    model=pickle.load(f)

with open('text.txt','r') as f:
        text=f.read()
t = Tokenizer()
t.fit_on_texts([text])

st.title(" Next Word Suggestion Model")

a=st.chat_input("write your  words")
#st.header(" Here is Word Suggestion")
try:
    def predict(seed):
        token_list = t.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen = 15, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        return t.index_word[predicted[0]]
    
    pred=predict(a)
    print(pred)
    st.header(" Here is Word Suggestion :")
    st.success(pred)

except Exception as e:
    print(e)
