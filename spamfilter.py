from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import time

# Access nltk's stopword list for English
stoplist = stopwords.words('english')

# Iteratively read files from spam and ham subfolders and keep them
# in two separate lists.
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r')
        a_list.append(f.read())
    f.close()
    return a_list

# Preprocess sentences via tokenization, lemmatization and making it lowercase
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

# Apply bow model
def get_features(text, setting):
    if setting == 'bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

# Take the set of features and the proportion of the examples assigned to the training set as arguments
# Check if data is split correctly: Training set should contain 4137 emails and the test set - 1035
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # Initialize the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    # Train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

# Evaluate performance/accuracy of how classifier performs
def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy on the test set = ' + str(classify.accuracy(classifier, test_set)))
    # Checks which 20 words were the most informative for the classifier
    classifier.show_most_informative_features(20)

# Initialize data
if __name__ == '__main__':
    start_time = time.time()
    # Spam and Ham lists of emails
    spam = init_lists('enron1/spam/')
    ham = init_lists('enron1/ham/')
    # Single list of spam and ham with two tuples 'email text' and 'label'
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    # Shuffle all emails to make organizing the training data more evenly distributed so any portio fo all_emails will contain both types.
    random.shuffle(all_emails)
    #Check if data loaded correctly by checking size of data structure (Total number of emails (spam+ham = 5175))
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')

    # Extract all features and pair them with the email class label spam or ham.
    all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]
    print ('Collected ' + str(len(all_features)) + ' feature sets')

    # Train the classifier
    train_set, test_set, classifier = train(all_features, 0.8)

    # Check performance/accuracy of how classifier performs
    evaluate(train_set, test_set, classifier)

    # Runtime
    print ('Runtime: ---%s seconds ---' %(time.time() - start_time))
