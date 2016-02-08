#
# Clean text into vector of words, and vector of sentences.
#

import nltk
from nltk.corpus import stopwords
import codecs
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from __future__ import division

def load(text_path, filename):
    # Set up raw text.
    f = codecs.open(text_path+filename, encoding="utf-8")
    text = f.read()
    return text

def clean(text):
    text = text.replace("\n", " ")
    text = text.encode("ascii", "ignore")
    return text

def head(text):
    # Inspect head of text file
    t = text[:502]
    return t

def wordlist_to_postags(wordlist):
    # Convert list of words into corresponding list of POS tags.
    tag_list = [tag for (_, tag) in nltk.pos_tag(wordlist)]
    return tag_list

def shingle(data_list, w):
    # Make windows of size w.
    num_items = len(data_list)
    sh = [data_list[i:i+w] for i in range(num_items - w + 1)]
    return sh

def test_human(sents, num_context_sents):
    # Test a human.
    context = num_context_sents
    i = random.randint(context, len(sents)-context)  # Choose a center.
    test_sents = sents[i-context:i+context+1]  # Fetch surrounding sentences.
    target_tokens = nltk.word_tokenize(test_sents[context])  # Tokenize center.
    removed_i = random.randint(0, len(target_tokens))  # Remove word from center.
    missing_word = target_tokens[removed_i]
    target_tokens[removed_i] = '*****'  # Replace word from center.
    test_sents[context] = ' '.join(target_tokens)  # Join new text all together.
    read_test = ' '.join(test_sents)

    print
    print read_test
    raw_input('\nClick enter to see missing word:\n\n')
    print missing_word
    print

def run():
    # Text location.
    text_path = '/Users/mauricediesendruck/Google Drive/fitb/'
    filename = 'prince.txt'
    filename = 'eol.txt'
    filename = 'meta.txt'

    # Create test text.
    text = clean(load(text_path, filename))
    t = head(text)

    # Compute text stats.
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", "''", '``', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w not in stop_words]
    counts = Counter(words).most_common()
    frequencies = [c for (_,c) in counts]
    # Below format... {a: b} means there are b unique words which occur a times.
    freq_freq = Counter(frequencies)
    print dict(freq_freq)[1]/len(counts)

    # Make big list of word w-shingles, and POS w-shingles.
    # Do each on a sentence-by-sentence basis.
    sents = nltk.sent_tokenize(text)
    wordlist_by_sent = [nltk.word_tokenize(s) for s in sents]
    poslist_by_sent = [wordlist_to_postags(w) for w in wordlist_by_sent]

    # Make word shingles.
    word_shingle_size = 5
    word_shingles = []
    for wordlist in wordlist_by_sent:
        word_shingles.append(shingle(wordlist, word_shingle_size))

    # Make POS shingles.
    pos_shingle_size = 5
    pos_shingles = []
    for poslist in poslist_by_sent:
        pos_shingles.append(shingle(poslist, pos_shingle_size))

    # Test a human
    num_context_sents = 1
    test_human(sents, num_context_sents)

run()






