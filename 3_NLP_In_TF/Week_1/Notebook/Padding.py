# %%
# Import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# %%
# Word based encoding
sentences = [
             'I love my dog',
             'I love my cat',
             'You love my dog!',
             'Do you think my dog is amazing'
]
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

tokenizer = Tokenizer(num_words= 100, oov_token="<OOV_tok>") # Using out of vocabulary token
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\nWord Index = ", word_index)
# %%
# Text to sequence
seq = tokenizer.texts_to_sequences(sentences)
test_seq = tokenizer.texts_to_sequences(test_data)
print('\nTexts to sequences:')
for i in range(len(sentences)):
    print(sentences[i], ' -> ', seq[i])
print('\nTexts on test data to sequences:')
for i in range(len(test_data)):
    print(test_data[i], ' -> ', test_seq[i])
# %%
# Padding
padded = pad_sequences(seq)
print('\nPadded Test Sequence:')
print(padded)
Test_padded = pad_sequences(test_seq)
print('\nPadded Test Sequence:')
print(Test_padded)
