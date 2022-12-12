import difflib
import re
import string

import nltk
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.chrome.options import Options
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
import aspose.words as aw
import librosa
import numpy as np
import soundfile as sf
import torch
from IPython.display import Audio
from scipy.io import wavfile
from transformers import (HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer)
from io import StringIO
from PIL import Image
import warnings
from textblob import TextBlob, Word
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import warnings
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import SessionState 
warnings.filterwarnings("ignore")

options = Options()
options.add_argument("headless")
def googleSearch(query):
    driver = webdriver.Chrome(options=options, executable_path='chromedriver')
    search_engine = "https://www.google.com/search?q="
    query = query.replace(" ","+")
    driver.get(search_engine + query + "&start=" + "0")

    df = {}
    
    stopwords=nltk.corpus.stopwords.words('english')
    def clean_text(text):
        edited_text=re.sub('\W'," ",text) #replace any sumbol with whitespace
        edited_text=re.sub("  "," ",edited_text) #replace double whitespace with single whitespace
        edited_text=edited_text.split(" ") #split the sentence into array of strings
        edited_text=" ".join([char for char in edited_text if char!= ""]) #remove any empty string from text
        edited_text=edited_text.lower() #lowercase
        edited_text=re.sub('\d+',"",edited_text) #Removing numerics
        edited_text=re.split('\W+',edited_text) #spliting based on whitespace or whitespaces
        edited_text=" ".join([word for word in edited_text if word not in stopwords]) #Snowball Stemmer
        return edited_text

    s_len = 10

    for s_block in range(1 , s_len+1):
        content_block_xpath1 = f'''//*[@id="rso"]/div[{s_block}]/div/div/div[1]/div/div'''
        content_block_xpath2 = f'''//*[@id="rso"]/div[{s_block}]/div/div/div'''
        content_block_xpath3 = f'''//*[@id="rso"]/div[{s_block}]/div/div'''
        content_block_xpath4 = f'''//*[@id="rso"]/div[{s_block}]/div[2]/div'''

        # xpaths
        xpath_url1 = f"""{content_block_xpath1}/div[1]/div/a"""
        xpath_url2 = f"""{content_block_xpath2}/div[1]/div/a"""
        xpath_url3 = f"""{content_block_xpath3}/div[1]/div/a"""
        xpath_url4 = f"""{content_block_xpath4}/div[1]/div/a"""

        try:
            block = {}
            links = []
            try:
                url1 = driver.find_elements('xpath', xpath_url1)
                for url in url1:
                    links.append(url.get_attribute("href"))
            except:
                pass
            try:
                url2 = driver.find_elements('xpath', xpath_url2)
                for url in url2:
                    links.append(url.get_attribute("href"))
            except:
                pass
            try:
                url3 = driver.find_elements('xpath', xpath_url3)
                for url in url3:
                    links.append(url.get_attribute("href"))
            except:
                pass
            try:
                url4 = driver.find_elements('xpath', xpath_url4)
                for url in url4:
                    links.append(url.get_attribute("href"))
            except:
                pass
            pattern =  r"""(https?:\/\/)?(([a-z0-9-_]+\.)?([a-z0-9-_]+\.[a-z0-9-_]+))"""
            if len(links) == 0:
                domain = re.search(pattern,'')
            else: 
                domain = re.search(pattern, links[0])
            if len(links)==0:
                final_text = ''
            else:
                def getdata(url): 
                    try:
                        r = requests.get(url)
                        return r.text
                    except:
                        r = ''
                        return r 
                htmldata = getdata(links[0]) 
                soup = BeautifulSoup(htmldata, 'html.parser') 
                data = '' 
                content = []
                for data in soup.find_all("p"): 
                    content.append(data.get_text())
                text_content = ' '.join(content)
                final_text = clean_text(text_content)
            block["domain"] = domain
            if len(links)==0:
                block["url"] = ''
            else:
                block["url"] = links[0]
            block["description"] = final_text

            df[f'{s_block}'] = block

        except exceptions.NoSuchElementException:
            continue

        if len(df) == 0:
            raise Exception("No data found")

    driver.close()
    return df

page_bg_image = """
<style>
[data-testid="stReportViewContainer"]{
    background: linear-gradient(90deg, rgba(101,58,180,1) 0%, rgba(253,171,29,1) 83%, rgba(252,133,69,1) 100%);
}
div[class="css-ng1t4o e1fqkh3o1"]{
    background-color: rgba(0,0,0,0);
}
[data-testid="stHeader"]{
    background-color: rgba(0,0,0,0);
}
button[class="css-1bvhuai edgvbvh1"] {
  background-color: #ffffff;
  color: #000000;
  font-weight: bold;
  font-size: 16px
}
</style>
"""
st.markdown(page_bg_image, unsafe_allow_html=True)

st.title("Plagiarism Checker and Paraphraser")
image = Image.open('Self-plagiarism.jpg')
st.image(image)
st.sidebar.title("Please Make a Selection")
session_state = SessionState.get(checkboxed=False, button_sent=False)
session_state = SessionState.get(checkboxed=False)
button_sent = st.sidebar.button("Check Plagiarism")
parap = st.sidebar.button("Paraphrase Sentence")
if button_sent or session_state.button_sent:
    session_state.button_sent = True
    session_state = SessionState.get(text = False)
    st.subheader("Plagiarism Finder")
    text = st.text_area(label="Paste your Text", max_chars=2500,height=200)
    uploaded_file = st.file_uploader("Choose a file",type= ['txt', 'docx','wav'])
    if text is not None:
        file = open('Output.txt', 'w')
        file.write(str(text))
        file.close()

    if uploaded_file is not None:
        text = None
        File = uploaded_file.name
        st.write(uploaded_file)

        if File.endswith('.docx'):
            doc = aw.Document(uploaded_file)
            doc.save("Output.txt")
        elif File.endswith('.txt'):
            tx_up = str(uploaded_file.getvalue())
            tx_upf = tx_up[1:]
            edited_up_text=re.sub('\W'," ",tx_upf)
            file = open('Output.txt', 'w',encoding='utf8')
            file.write(edited_up_text)
            file.close()
        elif File.endswith('.wav'):
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
            speech, rate = librosa.load(uploaded_file, sr=16000)
            input_values = processor(speech, return_tensors="pt", sampling_rate=rate).input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
            audio_text = ' '.join(transcription)
            file = open('Output.txt', 'w')
            file.write(str(audio_text))
            file.close()
        up_file = open('Output.txt', 'r')
        up_text = up_file.read()
        st.text_area(label="Your uploded text", value=up_text, height=180)
    Ch_Plagirism = st.button("Check plagiarism")
    clear = st.button("Clear")
    if(clear):
        st.experimental_rerun()

    if(Ch_Plagirism):
        files = ['Output.txt'] #  you can add file or bunch of files
        da = {}
        for f in files:
            with open (f, "r",encoding="utf8") as myfile:
                all_lines = myfile.read().splitlines()
                da['Text'] = all_lines
        df = pd.DataFrame(da)
        for i in range(len(df['Text'])):
            if len(df['Text'][i]) < 100:
                df = df.drop(i)
        stopwords=nltk.corpus.stopwords.words('english')
        def clean_text(text):
            edited_text=re.sub('\W'," ",text) #replace any sumbol with whitespace
            edited_text=re.sub("  "," ",edited_text) #replace double whitespace with single whitespace
            edited_text=edited_text.split(" ") #split the sentence into array of strings
            edited_text=" ".join([char for char in edited_text if char!= ""]) #remove any empty string from text
            edited_text=edited_text.lower() #lowercase
            edited_text=re.sub('\d+',"",edited_text) #Removing numerics
            edited_text=re.split('\W+',edited_text) #spliting based on whitespace or whitespaces
            edited_text=" ".join([word for word in edited_text if word not in stopwords])
            return edited_text
        df['Treated_Text']=df.Text.apply(lambda x: clean_text(x))

        final = {}
        for i in range(len(df['Treated_Text'])):
            final[f'{i}'] = googleSearch(f'{df.Text[i]}')

        webs = []
        plt = []
        count = 0
        for i in final:
            for j in final[i]:
                arr = [df.Treated_Text[int(i)],final[i][j]["description"]]
                vectorizer = TfidfVectorizer()
                obj = vectorizer.fit_transform(arr)
                vectors = obj.toarray() 
                sim = round(cosine_similarity(vectors)[0][1],2)
                if sim>0.30:
                    count = 1
                    plt.append(sim)
                    link = str(final[i][j]["domain"])
                    link_list = link.split(',')
                    webs.append(link_list[2])

        if count == 1:
            st.subheader("Plagiarism Found")
            list_of_tuples = list(zip(webs, plt))
            data = pd.DataFrame(list_of_tuples,columns=['Source URL', 'Similarity score'])
            st.dataframe(data)
        else:
            st.subheader("No Plagiarism Found")       

if parap or session_state.checkboxed:
    session_state.checkboxed = True
    st.subheader("Paraphraser")
    sentence = st.text_area(label="Paste your Text", max_chars=500,height=200)
    para_button = st.button("Paraphrase Sentence",key='Show')
    tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

    def paraphraser(para):
        sentence = "paraphrase: " + para + " </s>"
        encoding = tokenizer.encode_plus(sentence,padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks,
                                max_length=256,
                                do_sample=True,
                                top_k=120,
                                top_p=0.90,
                                early_stopping=True,
                                num_return_sequences=1)
        para_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        return(para_sentence)
    if(para_button):
        output = " ".join([paraphraser(sent) for sent in sent_tokenize(str(sentence))])
        st.text_area(label="Paraphrase Sentence", value=output, height=150)
        



