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

  print proc_text

  dictionary = corpora.Dictionary(proc_text)
  # dictionary.save(os.pardir + '/data/text.dict')
  return [dictionary, proc_text]

def start_page_rank():
  global reference_summary_list, system_summary_list

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

  [dictionary, proc_text] = save_word_dict(text)

  raw_corpus = [dictionary.doc2bow(t) for t in proc_text]

  print raw_corpus

  tfidf = models.TfidfModel(raw_corpus)
  
  # print tfidf
  corpus_tfidf = tfidf[raw_corpus]
  print corpus_tfidf

  print corpus_tfidf[1]
  # print dictionary
  # load from   

  print 'Input text:'
  print text
  print

  reference_summary = summarize(text)
  system_summary = "Morpheus wakes Neo into the real world that has been captured by machines."
  article_id = "test" # this should be filename id
  
  # write reference summary to file
  ref_dir = os.pardir + "/test-summarization/reference/" + article_id + "_" + "reference.txt"
  sys_dir = os.pardir + "/test-summarization/system/" + article_id + "_" + "system.txt"
  write_to_file(ref_dir, reference_summary)
  reference_summary_list.append([ref_dir])

  # write system summary to file
  write_to_file(sys_dir, system_summary)
  system_summary_list.append(sys_dir)

  test_print(reference_summary, system_summary)

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

  start_page_rank()
  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))

if __name__ == "__main__":
  main()