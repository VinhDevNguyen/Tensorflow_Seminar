# %%
# Import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
# %%
# Word based encoding
sentences = [
             'I love my dog',
             'I love my cat',
             'You love my dog!',
             'Do you think my dog is amazing'
]

tokenizer = Tokenizer(num_words= 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\nWord Index = ", word_index)
