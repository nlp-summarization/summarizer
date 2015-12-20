import numpy
from numpy.linalg import svd as singular_value_decomposition
from nltk import tokenize, word_tokenize
from gensim.parsing import preprocess_string
from gensim import corpora, models, similarities
from gensim.summarization import summarize
from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_short, strip_punctuation

import math
import os, glob, sys
import codecs

parent_dir = os.pardir
sys.path.insert(0, parent_dir + "/Rouge")
from PythonROUGE import *

global ROUGE_path, data_path
global reference_summary_list
global system_summary_list

def save_word_dict(text):
  proc_text = []
  
  sentences = text 
  sentences = tokenize.sent_tokenize(sentences)

  for sentence in sentences:
    sentence_without_stops = remove_stopwords(sentence)
    sentence_without_stops = stem_text(sentence_without_stops)
    sentence_without_stops = strip_short(sentence_without_stops)
    sentence_without_stops = strip_punctuation(sentence_without_stops)
    
    proc_sentence = word_tokenize(sentence_without_stops.lower())

    if(len(proc_sentence) == 0):
      continue
    proc_text.append(proc_sentence)

  dictionary = corpora.Dictionary(proc_text)
  return [dictionary, proc_text, sentences]

def compute_tf_idf(dictionary, proc_text):
  
  words_count = len(dictionary)
  sentences_count = len(proc_text)
  
  raw_corpus = [dictionary.doc2bow(t) for t in proc_text]
  tf_matrix = numpy.zeros((words_count, sentences_count), dtype=numpy.float)

  for sentence_id, word_counts in enumerate(raw_corpus):
    for (word_id, count) in word_counts:
      tf_matrix[word_id, sentence_id] = count

  idf_values = {}
  for (word_id, word) in dictionary.iteritems():
    word_count = 0
    for sentence in proc_text:
      if word in sentence:
        word_count = word_count + 1

    if(word_count == 0):
      term_ratio = 0
    else:
      term_ratio = float(sentences_count)/float(word_count)
    
    idf_values[word_id] = math.log(1 + term_ratio)

  return [tf_matrix, idf_values]

def start_lex_rank(article_id, limit, reference_summary, text=None):
  if(text == None):
    text = "Thomas A. Anderson is a man living two lives. By day he is an " + \
      "average computer programmer and by night a hacker known as " + \
      "Neo. Neo has always questioned his reality, but the truth is " + \
      "far beyond his imagination. Neo finds himself targeted by the " + \
      "police when he is contacted by Morpheus, a legendary computer " + \
      "hacker branded a terrorist by the government. Morpheus awakens " + \
      "Neo to the real world, a ravaged wasteland where most of " + \
      "humanity have been captured by a race of machines that live " + \
      "off of the humans' body heat and electrochemical energy and " + \
      "who imprison their minds within an artificial reality known as " + \
      "the Matrix. As a rebel against the machines, Neo must return to " + \
      "the Matrix and confront the agents: super-powerful computer " + \
      "programs devoted to snuffing out Neo and the entire human " + \
      "rebellion."

  [dictionary, proc_text, sentences] = save_word_dict(text)


  [tf_matrix, idf_values] = compute_tf_idf(dictionary, proc_text)

  sentence_matrix = compute_sentence_similarity_matrix(dictionary, proc_text, tf_matrix, idf_values, 0.3)
  # print sentence_matrix
  scores = power(sentence_matrix, 0.25)
  
  ranked_sentences = sorted(range(len(scores)),key=lambda x:scores[x], reverse=True)
  
  result_summary = ''
  for i in range(0, limit):
    result_summary = result_summary + ' ' + sentences[ranked_sentences[i]]

  system_summary = result_summary

  sys_dir = os.pardir + "/test-summarization/system/" + article_id + "_" + "system.txt"
  ref_dir = os.pardir + "/test-summarization/reference/" + article_id + "_" + "reference.txt"

  write_to_file(ref_dir, reference_summary)
  reference_summary_list.append([ref_dir])

  # write system summary to file
  write_to_file(sys_dir, system_summary)
  system_summary_list.append(sys_dir)
  # test_print(reference_summary, system_summary)
  return ranked_sentences

def test_print(reference_summary, system_summary):
  print "\n### reference_summary ###"
  print reference_summary
  print "\n### system_summary ###"
  print system_summary
  print "\n"

def write_to_file(filename, summary):
  with codecs.open(filename, 'w', encoding='utf8') as f:
    f.write(str(summary.encode('ascii', errors='ignore')))

def power(m, e):
  mT = m.T
  num_sentences = len(m)
  pVc = numpy.array(num_sentences*[1.0/num_sentences])
  l = 1.0
  while l > e:
      pNex = numpy.dot(mT, pVc)
      l = numpy.linalg.norm(numpy.subtract(pNex, pVc))
      pVc = pNex
  return pVc

def compute_sentence_similarity_matrix(dictionary, proc_text, tf_matrix, idf_values, threshold=0.01):

  sentence_length = len(proc_text)
  sentence_matrix = numpy.zeros((sentence_length, sentence_length), dtype=numpy.float)
  degree_matrix   = numpy.zeros((sentence_length, ), dtype=numpy.float)

  raw_corpus = [dictionary.doc2bow(t) for t in proc_text]

  for s_1, sentence1 in enumerate(raw_corpus):
    s1 = [i[0] for i in sentence1]

    for s_2, sentence2 in enumerate(raw_corpus):
      
      if(s_1 == s_2):
        continue

      s2 = [i[0] for i in sentence2]
      
      # print proc_text[s_1], " => " , proc_text[s_2]
      common_words = intersect(s1, s2)

      if(len(common_words) == 0):
        continue

      sentence_matrix[s_1, s_2] = idf_modified_cosine(common_words, s_1, s_2, s1, s2, tf_matrix, idf_values)

      if(sentence_matrix[s_1, s_2] > threshold):
        degree_matrix[s_1] = degree_matrix[s_1] + 1.0
      else:
        sentence_matrix[s_1, s_2] = 0.0


  for s_1 in range(0, sentence_length):
   for s_2 in range(0, sentence_length):
     if(degree_matrix[s_1] > 0.0):
       sentence_matrix[s_1, s_2] = sentence_matrix[s_1, s_2]/degree_matrix[s_1]

  return sentence_matrix

def idf_modified_cosine(common_words, s_1, s_2, s1, s2, tf_matrix, idf_values):
  # print common_words
  sentence_score = 0.0
  for word_id in common_words:
    sentence_score = sentence_score + (tf_matrix[word_id, s_1] * tf_matrix[word_id, s_2] * pow(idf_values[word_id], 2) )

  sentence1_sum = 0.0
  for word_id in s1:
    sentence1_sum = sentence1_sum  + pow((tf_matrix[word_id, s_1] * idf_values[word_id] ), 2)

  sentence2_sum = 0.0
  for word_id in s2:
    sentence2_sum = sentence2_sum  + pow((tf_matrix[word_id, s_2] * idf_values[word_id] ), 2)


  if(sentence1_sum == 0.0 or sentence2_sum == 0.0):
    return 0.0
  else:
    return sentence_score/(sentence2_sum * sentence1_sum)

  # print "sentence_score : ", sentence_score
  # print "sentence_score 1: ", sentence1_sum
  # print "sentence_score 2: ", sentence2_sum

def intersect(a, b):
  return list(set(a) & set(b))

def main():
  global system_summary_list, reference_summary_list

  system_summary_list = []
  reference_summary_list = []

  start_lex_rank("test", 3, None, None)

  return
  recall_list, precision_list, F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))

if __name__ == "__main__":
  main()