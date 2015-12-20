import numpy
from numpy.linalg import svd as singular_value_decomposition
from nltk import tokenize
from gensim.parsing import preprocess_string
from gensim import corpora, models, similarities
from gensim.summarization import summarize
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
    proc_sentence = preprocess_string(sentence)

    if(len(proc_sentence) == 0):
      continue
    proc_text.append(proc_sentence)

  dictionary = corpora.Dictionary(proc_text)
  return [dictionary, proc_text, sentences]

def create_tf_matrix(proc_text, dictionary):

  words_count = len(dictionary)
  sentences_count = len(proc_text)
  
  raw_corpus = [dictionary.doc2bow(t) for t in proc_text]
  matrix = numpy.zeros((words_count, sentences_count), dtype=numpy.float)

  for sentence_id, word_counts in enumerate(raw_corpus):
    for (word_id, count) in word_counts:
      matrix[word_id, sentence_id] = count

  return matrix

def normalize_tf_matrix(matrix, alpha=0.7):
  rows = len(matrix)
  cols = len(matrix[0])

  max_freq = numpy.max(matrix, axis=0)
    
  for i in range(0, rows):
    for j in range(0, cols):
      max_f = max_freq[j]
      if max_f == 0:
        continue

      # dividing by max term freq so as to not penalize long sentences
      term_freq = matrix[i, j] / max_f
      matrix[i, j] = alpha + (1.0 - alpha) * term_freq

  return matrix

def start_lsa(article_id, limit, text, reference_summary):
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
  

  tf_matrix = create_tf_matrix(proc_text, dictionary)
  tf_matrix = normalize_tf_matrix(tf_matrix, 0.3)

  # decompose in U x S X V matrices using SVD
  [u, s, v] = singular_value_decomposition(tf_matrix, full_matrices=False)

  reduction_ratio = 1.0
  dimension = s.shape[0]
  reduced_dimension = int(dimension * reduction_ratio)

  min_dimension = 1

  if(reduced_dimension < min_dimension):
    reduced_dimension = min_dimension

  s2 = numpy.array(s, copy=True)
  s2 = numpy.square(s2)

  if(reduced_dimension < dimension):
    for i in range(reduced_dimension, dimension):
      s2[i] = 0

  # http://textmining.zcu.cz/publications/PhDThesis-Steinberger.pdf
  # see page 25 - Sk = sqrt(sum(v * sigma^2 ))
  ranks = numpy.sqrt(numpy.square(v.T*s2).sum(axis=1))
  #print "ranks " , ranks
  ranked_sentences = sorted(range(len(ranks)),key=lambda x:ranks[x], reverse=True)
  #print "ranked_sentences ", ranked_sentences

  
  result_summary = ''
  for i in range(0, limit):
    result_summary = result_summary + ' ' + sentences[ranked_sentences[i]]

  system_summary = result_summary

  # if(reference_summary != None):
  #   try:
  #     reference_summary = summarize(text)
  #   except (ValueError, ZeroDivisionError):
  #     return -1

  # if(reference_summary == None or len(reference_summary) == 0 or len(reference_summary) > 140):
  #   return -1

  # write reference summary to file
  
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

def main():
  global system_summary_list, reference_summary_list

  system_summary_list = []
  reference_summary_list = []

  start_lsa("test", 2, None, None)
  recall_list, precision_list, F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))


if __name__ == "__main__":
  main()