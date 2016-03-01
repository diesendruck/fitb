#
# Clean text into vector of words, and vector of sentences.
#

import nltk
from nltk.corpus import stopwords
import codecs
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from __future__ import division
import matplotlib
matplotlib.style.use('ggplot')

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

def compute_text_stats(text):
    # Compute text stats.
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", "''", '``', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w not in stop_words]
    counts = Counter(words).most_common()
    frequencies = [c for (_,c) in counts]
    # Below format... {a: b} means there are b unique words which occur a times.
    freq_freq = Counter(frequencies)
    print "Percentage of unique words occuring only once:"
    return dict(freq_freq)[1]/len(counts)

def is_match(sh, q):
    # Given two lists of strings, checks if a subset of those strings is equal.
    if (sh[0]==q[0] and sh[2]==q[2]):  # This test is a 3-shingle PREFERENCE.
        return True
    else:
        return False

def predict_pos(pos_sh, word_sh, i, verbose=True):
    # Given lists of pos and word shingles, and random query i; predict POS.
    without_i = pos_sh[:i]+pos_sh[i+1:]
    # NOTE: This is under the 3-tuple/middle-missing regime.
    query = pos_sh[i]
    true_POS = pos_sh[i][1]
    context_matches = [sh for sh in without_i if is_match(sh, query)]
    all_candidates = [c for (_,c,_) in context_matches]
    n = len(all_candidates)
    f = Counter(all_candidates).most_common()
    if len(f)>=2:
        top_guess, top_score = f[0][0], f[0][1]/n
        second_guess, second_score = f[1][0], f[1][1]/n
        delta = top_score - second_score
    elif len(f)==1:
        top_guess, top_score = f[0][0], f[0][1]/n
        second_guess, second_score = [0] * 2
        delta = 0
    else:
        top_guess, top_score, second_guess, second_score, delta = [0] * 5
    context_width = 15
    if verbose==True:
        print '\n ~~~~ SUMMARY ~~~~ '
        print 'WORD TUPLE: {}'.format(word_sh[i])
        print 'POS TUPLE: {}'.format(pos_sh[i])
        print '\nCONTEXT:'
        print ' '.join([s for (_,_,s) in word_sh[i-context_width:i+context_width]])
        print '\nMATCH FREQUENCIES: n={}, {}'.format(n, f[:10])
        print '\nTRUTH: {}'.format(true_POS)
        print '\nTOP TWO GUESSES: {}, {} ({}, {}, delta={})'.format(
            top_guess, second_guess, round(top_score, 3), round(second_score, 3),
            round(delta, 3))
    return top_guess, second_guess, delta, true_POS

def test_POS_prediction(num_trials):
    results = [[], [], []]
    for d in np.arange(0, 1, 0.1):  # Vary delta threshold.
        num_guesses = 0
        num_correct = 0
        for _ in range(num_trials):  # Do many trials for that delta.
            ind = random.randrange(len(pos_sh))
            q = pos_sh[ind]
            p1, p2, delta, truth = predict_pos(pos_sh, word_sh, ind, verbose=False)
            if delta > d:  # Only guess if delta is big enough.
                num_guesses += 1
                if p1 == truth:
                    num_correct += 1
        if num_guesses >= 1:
            precision = num_correct/num_guesses
        print 'DELTA VALUE: {}; PRECISION: {}; GUESS PROPORTION: {}/{}'.format(
            round(d, 3), round(precision, 3), num_guesses, num_trials)
        results[0].append(d)
        results[1].append(precision)
        results[2].append(num_guesses)
    return results

def plot_POS_pred_results(results):
    trace1 = go.Scatter(
        x=results[0],
        y=results[1],
        mode='markers',
        marker=dict(size=[n/10 for n in results[2]])
    )
    trace2 = go.Scatter(
        x=results[0],
        y=results[1],
        hoverinfo='none'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='POS Prediction Precision (3-tuple)',
        xaxis=dict(title='Delta Threshold'),
        yaxis=dict(title='Precision'),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='pos-prediction-delta-threshold')

def run():
    # Text location.
    text_path = '/Users/mauricediesendruck/Google Drive/fitb/'
    filename = 'prince.txt'
    filename = 'eol.txt'
    filename = 'meta.txt'

    # Create test text.
    text = clean(load(text_path, filename))
    t = head(text)

    # Show proportion of vocabulary that occurs only once.
    print compute_text_stats(text)

    # Make lists of word and POS w-shingles, each on a per-sentence basis.
    sents = nltk.sent_tokenize(text)
    wordlist_by_sent = [nltk.word_tokenize(s) for s in sents]
    poslist_by_sent = [wordlist_to_postags(w) for w in wordlist_by_sent]

    # Make word shingles.
    word_shingle_size = 3
    word_shingles = []
    for wordlist in wordlist_by_sent:
        word_shingles.append(shingle(wordlist, word_shingle_size))
    word_sh = [sh for sent_list in word_shingles for sh in sent_list]

    # Make POS shingles.
    pos_shingle_size = 3
    pos_shingles = []
    for poslist in poslist_by_sent:
        pos_shingles.append(shingle(poslist, pos_shingle_size))
    pos_sh = [sh for sent_list in pos_shingles for sh in sent_list]

    # Predict POS.
    num_trials = 1000
    results = test_POS_prediction(num_trials)
    pos_prediction_results = plot_POS_pred_results(results)

    # Test a human
    num_context_sents = 1
    test_human(sents, num_context_sents)

run()






