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

tokenizer = Tokenizer(num_words= 100, oov_token="<OOV_tok>") # Using out of vocabulary token
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\nWord Index = ", word_index)
# %%
# Text to sequence
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print('\nTexts to sequences:')
for i in range(len(test_data)):
    print(test_data[i], ' -> ', test_seq[i])
