import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords

# cow = 'There is no abaci level'
#
# cow_list = word_tokenize(cow)

lemma = WordNetLemmatizer()

# for idx in range(len(tokened)):
#     tokened[idx] = lemma.lemmatize(tokened[idx])

#
# test = np.array([
#     cow_list
# ])

# def lemming(word):
#     return lemma.lemmatize(word)
#
# test = lemming(test)


def read_data(file_name):
    all_words = []
    data = []
    with open(file_name) as f:
        for line in f:
            tokened_list = word_tokenize(line.decode('utf8').lower())
            data.append(tokened_list)
            all_words += tokened_list
    return data, all_words


def prep_data(data, l2, classification):
    featureset = []
    features = np.zeros(len(l2))
    l2_set = set(l2)
    for line in data:
        for word in line:
            if word in l2_set:
                idx = l2.index(word)
                features[idx] += 1
        featureset.append([list(features), classification])
    return featureset

def remove_common_words(lex, min=None, max=None):
    #  de dupe
    word_counts = Counter(lex)
    if not min and not max:
        stop = set(stopwords.words('english'))
        return [w for key, w in enumerate(word_counts) if w not in stop]
    else:
        l2 = []
        for w in word_counts:
            if max > word_counts[w] > min:
                l2.append(w)
        return l2

# print l2.index('greater')

def create_feature_sets_labels(min=None, max=None, test_size=0.1):
    pos_data, pos_lexicon = read_data('pos.txt')
    neg_data, neg_lexicon = read_data('neg.txt')
    lexicon = []
    lexicon += pos_lexicon
    lexicon += neg_lexicon
    print 'Lexicon total length', len(lexicon)
    lexicon = [lemma.lemmatize(w) for w in lexicon]
    print 'Lexicon length after lemm', len(lexicon)
    l2 = remove_common_words(lexicon, min, max)
    print len(l2)

    features = prep_data(pos_data, l2, [1, 0]) + prep_data(neg_data, l2, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    print 'Number of features:', len(features)
    testing_size = int(test_size * len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])
    print 'Train lengths:', len(train_x), len(train_y)
    print 'Test lengths:', len(test_x), len(test_y)
    return train_x, train_y, test_x, test_y

# huh = ['apple', 'bannana', 'peach', 'orange']
#
# test_size = int(0.25 * len(huh))
#
# print huh[:test_size]
# print huh[:-test_size]
#
# print huh[test_size:]
# print huh[-test_size:]


if __name__ == '__main__':
    # create_feature_sets_labels()
    train_x, train_y, test_x, test_y = create_feature_sets_labels(50, 1000)
    # print 'Writing pickle'
    # with open('sentiment_set_stop_words.pickle', 'wb') as f:
    #     pickle.dump([train_x, train_y, test_x, test_y], f)
    print 'Done'

# create_feature_sets_labels()
