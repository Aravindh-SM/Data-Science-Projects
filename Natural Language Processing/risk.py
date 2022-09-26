
import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer

def sent_to_word(sent):
    
    words = RegexpTokenizer('\w+').tokenize(sent)
    words = [re.sub(r'([xx]+)|([XX]+)|(\d+)', '', w).lower() for w in words]
    words = list(filter(lambda a: a != '', words))
    return words

pickledmodel = open('model_RF.pkl',"rb")
nlp_classifier = pickle.load(pickledmodel)

#model_XGB = xgboost.Booster()
#model_XGB.load_model('model_XGB.JSON')


def model_pipeline(problem):
  embeddings = {}

  f = open('glove.6B.300d.txt', 'r', encoding = 'utf-8')

  for line in f:
    values = line.split()
    words = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings[words] = coefs
  f.close()

  data_list = list()
  sentence = np.zeros(300)
  count = 0
  dataset_word = sent_to_word(problem)
  for x in dataset_word:
       try:
           sentence += embeddings[x]
           count += 1
       except KeyError:
           continue
  data_list.append(sentence / count)

  return  model_RF.predict(np.array(data_list))



def main():
    st.title("Risk Analysis Chatbot")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Risk Analysis Chatbot </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    problem = st.text_input("Problem","Type Here")

    result=""

    if st.button("Predict"):
        result = model_pipeline(problem)
    st.success('The output is {}'.format(result))

if __name__=='__main__':
    main()