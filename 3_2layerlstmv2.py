import tensorflow as tf
import re
import math

BATCH_SIZE = 50
MAX_WORDS_IN_REVIEW = 250  # Maximum length of a review to consider
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

    OUTPUT_SIZE = 2
    LSTM_SIZE = 100
    LEARN_RATE = 1e-3

    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name = "input_data")
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_SIZE], name = "labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape = [], name = "dropout_keep_prob")

    lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE)
    lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_cell1,
    						output_keep_prob = dropout_keep_prob)
    						
    lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE)
    lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_cell2,
    						output_keep_prob = dropout_keep_prob)
    						
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2])
    init_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
	
    value, _ = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype = tf.float32, initial_state = init_state)
    logits = tf.layers.dense(value[:,-1,:], OUTPUT_SIZE, activation = None)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
    loss = tf.identity(loss, name = "loss")
    optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
    
    Accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    Accuracy = tf.identity(Accuracy, name = "accuracy")
    
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
