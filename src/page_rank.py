#!/usr/bin/python
import sys
import pdb
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import os, glob
import codecs
from gensim.summarization import summarize
from gensim.parsing import preprocess_string
from gensim import corpora, models, similarities
from nltk import tokenize

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
    proc_sentence = preprocess_string(sentence) # ' '.join(preprocess_string(sentence))

    if(len(proc_sentence) == 0):
      continue
    proc_text.append(proc_sentence)

  dictionary = corpora.Dictionary(proc_text)
  # dictionary.save(os.pardir + '/data/text.dict')
  return [dictionary, proc_text, sentences]

def start_page_rank(article_id, text=None):
  global reference_summary_list, system_summary_list

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
    "rebellion. "

  [dictionary, proc_text, sentences] = save_word_dict(text)
  raw_corpus = [dictionary.doc2bow(t) for t in proc_text]

  if(len(raw_corpus) <= 1):
    return -1

  tfidf        = models.TfidfModel(raw_corpus)
  corpus_tfidf = tfidf[raw_corpus]
  simMat       = similarities.MatrixSimilarity(tfidf[raw_corpus])
  similarityMatrix = simMat[corpus_tfidf]
  
  converged = False
  
  W = {}
  for i, s1 in enumerate(simMat):
    for j, s2 in enumerate(s1):
      W[(i, j)] = s2

  ranks = [1.0]*len(proc_text)
  temp_ranks = [1.0]*len(proc_text)
  damping_factor = 0.5
  num_sent = len(proc_text)

  while(not converged):
    for i in range(0, num_sent):
      total_sum  = 0.0
      for j in range(0, num_sent):
        if(j == i):
          continue
        # print "[" + str(i) + ", " + str(j) +"] : ", s2, " ",
        if(j < i):
          W[(j, i)] = similarityMatrix[j][i]
        else:
          W[(j, i)] = similarityMatrix[i][j]
        
        den_sum = 0.0
        for k in range(0, num_sent):
          if(k == j):
            continue
          if(j < k):
            W[(j, k)] = similarityMatrix[j][k]
          else:
            W[(j, k)] = similarityMatrix[k][j]
          den_sum = den_sum + W[(j, k)]
        # end k loop
        
        if(den_sum != 0.0):
          total_sum = total_sum + (W[(j, i)] * ranks[j] / den_sum)
      # end j loop
      rank = (1 - damping_factor) + (damping_factor * total_sum)
      temp_ranks[i] = rank
    # end i loop

    if(temp_ranks == ranks):
      converged = True  
    else:
      ranks = temp_ranks
    
  # end while loop

  ranked_sentences = sorted(range(len(ranks)),key=lambda x:ranks[x], reverse=True)
  limit = 1 # to limit number of sentences
  
  result_summary = ''
  for i in range(0, limit):
    result_summary = result_summary + ' ' + sentences[ranked_sentences[i]]

  # print result_summary
  try:
    reference_summary = summarize(text, word_count = 20)
  except (ValueError, ZeroDivisionError):
    return -1

  if(reference_summary == None or len(reference_summary) == 0 or len(reference_summary) > 140):
    return -1

  # system_summary = "Morpheus wakes Neo into the real world that has been captured by machines."
  system_summary = result_summary
  # article_id = "test" # this should be filename id
  
  # write reference summary to file
  ref_dir = os.pardir + "/test-summarization/reference/" + article_id + "_" + "reference.txt"
  sys_dir = os.pardir + "/test-summarization/system/" + article_id + "_" + "system.txt"
  write_to_file(ref_dir, reference_summary)
  reference_summary_list.append([ref_dir])

  # write system summary to file
  write_to_file(sys_dir, system_summary)
  system_summary_list.append(sys_dir)
  test_print(reference_summary, system_summary)
  return 1

def test_print(reference_summary, system_summary):
  print "\n### reference_summary ###"
  print reference_summary
  print "\n### system_summary ###"
  print system_summary
  print "\n"

def write_to_file(filename, summary):
  with codecs.open(filename, 'w', encoding='utf8') as f:
    f.write(str(summary.encode('ascii', errors='ignore')))

def generate_score():
  global reference_summary_list, system_summary_list
  training = ET.parse('../data/training.xml')

  files_read = 0
  # reference_summary_list = []
  # system_summary_list = []

  for i, t in enumerate(training.findall("article")):
    article_id = t.get('id')
    reference_summary = t.find("summary").text
    text = t.find("text").text

    # call page_rank
    result_code = start_page_rank(article_id, text)

    if (files_read > 10):
      break

    if(result_code != -1):
      files_read = files_read + 1

def main():
  global system_summary_list, reference_summary_list
  system_summary_list = []
  reference_summary_list = []

  generate_score()
  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))

if __name__ == "__main__":
  main()