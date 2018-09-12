import tensorflow as tf
import re
import math

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    
    # Changing to lower case
    review = review.lower()

    # Removing punctuation and numbers
    review = re.sub('[^a-z ]','',review)

    # Remove stop words
    wordlist = review.split()
    remaining_words = [word for word in wordlist if word not in stop_words]
    processed_review = ' '.join(remaining_words)
    

    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    """ MODEL DESIGN
    Word embedding - model places each word in some positive/negative space
    Overall positioning of words is used to predict label

    1 embedding layer
    1 LSTM
    1 fully connected
    """

    NUM_CLASSES = 2
    LSTM_SIZE = 100

    dropout_keep_prob = tf.placeholder(tf.float32, shape = [1])

    input_data = tf.placeholder(tf.float32, shape = [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
    input_data = tf.reshape(input_data, [-1, MAX_WORDS_IN_REVIEW])
    input_data = tf.split(input_data, MAX_WORDS_IN_REVIEW, 1)
    
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE, NUM_CLASSES])
    
    lstm_weight = tf.Variable(tf.truncated_normal([LSTM_SIZE, NUM_CLASSES]))
    lstm_bias = tf.Variable(tf.constant(0.1, shape = [NUM_CLASSES]))

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = dropout_keep_prob)

    values, states = tf.nn.static_rnn(lstm_cell, input_data, dtype = tf.float32)

    pred = tf.matmul(values[-1], lstm_weight) + lstm_bias

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels = labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
