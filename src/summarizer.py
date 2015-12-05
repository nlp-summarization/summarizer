#!/usr/bin/python
import sys
import pdb
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import os, glob
import codecs
from gensim.summarization import summarize
  
parent_dir = os.pardir
sys.path.insert(0, parent_dir + "/Rouge")
from PythonROUGE import *

global ROUGE_path, data_path
global reference_summary_list
global system_summary_list

def generate_summary():
  global reference_summary_list, system_summary_list
  training = ET.parse('../data/training.xml')

  files_read = 0

  reference_summary_list = []
  system_summary_list = []
  for i, t in enumerate(training.findall("article")):
    article_id = t.get('id')
    reference_summary = t.find("summary").text
    text = t.find("text").text
    try:
      system_summary = summarize(text, word_count = 20)
    except (ValueError, ZeroDivisionError):
      continue 

    if(system_summary == None or len(system_summary) > 140 or len(system_summary) == 0 ):
      continue

    # write reference summary to file
    ref_dir = os.pardir + "/test-summarization/reference/" + article_id + "_" + "reference.txt"
    sys_dir = os.pardir + "/test-summarization/system/" + article_id + "_" + "system.txt"
    write_to_file(ref_dir, reference_summary)
    reference_summary_list.append([ref_dir])

    # write system summary to file
    write_to_file(sys_dir, system_summary)
    system_summary_list.append(sys_dir)

    if (files_read > 10):
      break

    files_read = files_read + 1

def write_to_file(filename, summary):
  with codecs.open(filename, 'w', encoding='utf8') as f:
    f.write(str(summary.encode('ascii', errors='ignore')))

def main():
  global system_summary_list, reference_summary_list
  generate_summary()
  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))

if __name__ == "__main__":
  main()
