# %%
# import some package
import requests
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# %%
# Donwload json file
r = requests.get('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json')
with open('sarcasm.json', 'wb') as f:
    f.write(r.content)
    f.close()
# %%
# Read json file
with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)
    f.close()
# %%
# Analyze json file
vocab = list(datastore[0].keys())
sentences = [] 
labels = []
urls = []
for item in datastore:
    urls.append(item[vocab[0]])
    sentences.append(item[vocab[1]])
    labels.append(item[vocab[2]])
# %%
# Word based encodings
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
# %%
# Texts to sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)
