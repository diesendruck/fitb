#
# Clean text into vector of words, and vector of sentences.
#

import nltk
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.style.use('ggplot')

# Text location.
text_path = "/Users/mauricediesendruck/Google Drive/fitb/"
filename = "eol.txt"

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

def test_human(num_context_sents):
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
    # Create test text.
    text = clean(load(text_path, filename))
    t = head(text)

    # Make big list of word w-shingles, and POS w-shingles.
    # Do each on a sentence-by-sentence basis.
    sents = nltk.sent_tokenize(text)
    wordlist_by_sent = [nltk.word_tokenize(s) for s in sents]
    poslist_by_sent = [wordlist_to_postags(w) for w in wordlist_by_sent]

    # Make word shingles.
    word_shingles = []
    for wordlist in wordlist_by_sent:
        word_shingles.append(shingle(wordlist, 5))

    # Make POS shingles.
    pos_shingles = []
    for poslist in poslist_by_sent:
        pos_shingles.append(shingle(poslist, 5))

    # Test a human
    test_human(1)




run()






