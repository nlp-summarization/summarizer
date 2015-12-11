from nltk import tokenize
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from itertools import combinations
import cPickle
import re
import os
import sys
import kenlm
import nltk
import marshal
import cloudpickle
from nltk.collocations import *

test = ["the", "cat", "sat", "in", "the", "hat"]
#combs = combinations(test, 1)

def load_paraphrase_dict():
  f=open("../data/paraphrase_large")
  paraphrase = {}
  for line in f:
    split_struct = line.split("|||")
    source = ' '.join(re.sub(r'\[[^)]*\]', '', split_struct[1]).split()).lower()
    target = ' '.join(re.sub(r'\[[^)]*\]', '', split_struct[2]).split()).lower()
    if len(source) > len(target):
      paraphrase[source] = target
  f.close()
  return paraphrase

def get_paraphrase_dict():
  paraphrase_dict = None
  file_path = '../data/pickled_paraphrase'
  if os.path.isfile(file_path):
    print("loading from file")
    f = open(file_path, 'rb')
    paraphrase_dict = cPickle.load(f)
  else:
    print("loading paraphrase dictionary")
    paraphrase_dict = load_paraphrase_dict()
    print("writing to file")
    f = open(file_path, 'wb')
    cPickle.dump(paraphrase, f)
  return paraphrase_dict

# accepts list of words in sentence
def ordered_combinations(sentence):
  paraphrase_beginning = []
  paraphrase_middle = []
  paraphrase_end = []
  for x in xrange(len(sentence)):
    for i in range(x,len(sentence)):

      # beginning paraphrase split
      beginning = []
      for j in range(0,x):
        beginning.append(sentence[j])
      paraphrase_beginning.append(beginning)

      # middle paraphrase split
      middle = []
      for j in range(x,i+1):
        middle.append(sentence[j]) 
      paraphrase_middle.append(middle)

      # end paraphrase split
      end = []
      for j in range(len(middle) + x, len(sentence)):
        end.append(sentence[j])
      paraphrase_end.append(end)

  paraphrase_full = []
  for i,j,k in zip(paraphrase_beginning, paraphrase_middle, paraphrase_end):
    # join middle and end
    joined = [' '.join(j), ' '.join(k)]
    # insert beginning correctly
    for b in reversed(i):
      joined = [b] + joined
     # remove empty string
    joined = [i for i in joined if i != '']
    paraphrase_full.append(joined)

  return paraphrase_full

def get_trigram_score(words):
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder = TrigramCollocationFinder.from_words(words)
  score = 0.0
  fndr = finder.score_ngrams(trigram_measures.pmi)
  size = len(fndr)
  for i in fndr:
    score = score + i[1]
  return score/float(size)

def get_lowest_score_sentence(sentence)


def main():
  model = kenlm.LanguageModel("../data/language_model.klm")

  # model = cPickle.load(open('model.p', 'rb'))

  paraphrase_dict = get_paraphrase_dict()
  print len(paraphrase_dict)

  #test = ["the", "cat", "sat"]
  #test = ["this", "man", "is", "a", "liar"]
  #test = ["three", "soldiers", "died", "today"]
  #test = "both were personally likable policy aficionados who inspired tremendous loyalty from aides"
  #test = "in november he spent parts of 14 days in florida, including a break for thanksgiving"
  words = word_tokenize(test)
  print "Original sentence:"
  print test, abs(model.score(' '.join(words)))
  print ""

  combinations = ordered_combinations(words)

  #print "Combinations:"
  #for c in combinations: print c
  #print ""

  paraphrased_sentences = []
  for split_sentence in combinations:
    sent = []
    for phrase_to_check in split_sentence:
      if phrase_to_check in paraphrase_dict and len(phrase_to_check) > 0:
        sent.append(paraphrase_dict[phrase_to_check])
      else:
        sent.append(phrase_to_check)
    paraphrased_sentences.append(sent)

  # sentence with lowest score
  index_of_lowest_score = 0
  min_score = sys.maxint
  for i,p in enumerate(paraphrased_sentences):
    #length = 0
    #for l in p:
    #  length = length + len(l)
    score = abs(model.score(' '.join(p)))
    #score = length_weight*length + score_weight*s
    if score < min_score:
      index_of_lowest_score = i
      min_score = score

  # sentence with smallest length
  index_of_smallest = 0
  min_length = sys.maxint
  for i,p in enumerate(paraphrased_sentences):
    length = 0
    for l in p:
      length = length + len(l)
    if length < min_length:
      index_of_smallest = i
      min_length = length


  #for x in paraphrased_sentences:
  #  print x

  # print "Paraphrased sentences:"
  #for c,p in zip(paraphrased_sentences, combinations):
    #score = get_trigram_score(word_tokenize(' '.join(c)))
  #  print p, ' => ',  c, ' score = ', abs(model.score(' '.join(c))), get_trigram_score(' '.join(c))

  lowest_score_word_list = paraphrased_sentences[index_of_lowest_score]
  paraphrased_sentence = ' '.join(lowest_score_word_list)
  print "Solution with lowest score:"
  print paraphrased_sentence, abs(model.score(paraphrased_sentence))

  smallest_length_word_list = paraphrased_sentences[index_of_smallest]
  paraphrased_sentence = ' '.join(smallest_length_word_list)
  print ""
  print "Solution with smallest length:"
  print paraphrased_sentence, abs(model.score(paraphrased_sentence))

  print ""
  print "Intersection of lowest score and lowest length:"
  lowest_score_word_list = ' '.join(lowest_score_word_list).split()
  smallest_length_word_list = ' '.join(smallest_length_word_list).split()
  intersected_words = [itm for itm in lowest_score_word_list if itm in smallest_length_word_list]
  #intersected_words = list(set(lowest_score_word_list).intersection(smallest_length_word_list))
  print ' '.join(intersected_words), abs(model.score(' '.join(intersected_words)))

  # language model
  
  #sentence = "this is sentence a."
  #print model.score(sentence)





if __name__ == "__main__":
  main()















