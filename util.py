from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tfidf_transform(x_train, x_test):
    """Convert input raw texts into tfidf encoded floating point matrix
    Args:
        x_train: array of input text, the raw input training text data
        x_test: array of input text, the raw input test text data
    Returns:
        A tuple of two tfidf encoded matrix, (train matrix, test matrix)
    """
    kwargs = {
            'ngram_range': (1,1),  # Use 1-grams + 2-grams.
            'analyzer': 'word',  # Split text into word tokens.
            'min_df': 1,
            'stop_words': "english",
    }
    vectorizer = TfidfVectorizer(**kwargs)
    # Learn vocabulary from training texts and vectorize training texts.
    x_train_transformed = vectorizer.fit_transform(x_train)
    # Vectorize validation texts.
    x_test_transformed = vectorizer.transform(x_test)
    return x_train_transformed, x_test_transformed


def pad_sequence_transform(x_train, x_test, vocab_size=50000, max_len=20000):
    """Convert input raw tests into pad sequence encoded integer matrix
    Args:
        x_train: array of input text, the raw input training text data
        x_test: array of input text, the raw input test text data
        vocab_size: maximum number of vocabulary used for tokenization
                    default to 50000
        max_len: maximum length of padded sequences
                default to 20000
    """
    oov_tok = '<OOV>'
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    x_seq = tokenizer.texts_to_sequences(x_train)
    train_padded = pad_sequences(x_seq, padding='post', maxlen=max_len)
    test_padded = pad_sequences(tokenizer.texts_to_sequences(x_test), padding='post', maxlen=max_len)
    return train_padded, test_padded
