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
from gensim.models import ldamodel
from nltk import tokenize
import itertools
from HTMLParser import HTMLParser

parent_dir = os.pardir
sys.path.insert(0, parent_dir + "/Rouge")
from PythonROUGE import *

import lsa
import baseline

global ROUGE_path, data_path
global reference_summary_list
global system_summary_list

def parse_text(folder_path, num_files=1):

  num_files_read = 0
  root_tag = ET.Element("root")

  raw_texts = {}

  for root, dirs, files in os.walk(folder_path):
    path = root.split('/')
    
    for file in files:
      
      if not file.startswith('.') and os.path.isfile(os.path.join(root, file)) and num_files_read < num_files:
        
        
        e = xml.etree.ElementTree.parse(os.path.join(root, file))
        
        text = ''

        text_tag = e.findall("TEXT")
        text_ptags = e.findall("TEXT/P")

        if len(text_tag) > 0:
          for t in text_tag:
            text = text + t.text

        if len(text_ptags) > 0:
          for t in text_ptags:
            text = text + t.text

        raw_texts[file] = text

        num_files_read = num_files_read + 1
      else:
        continue

  return raw_texts

def parse_summary(folder_path, num_files, allowed_keys):

  num_files_read = 0
  raw_texts = {}

  for root, dirs, files in os.walk(folder_path):
    
    summary_file = "perdocs"
    
    if summary_file in files:
      with open(os.path.join(root, summary_file)) as f:
        it = itertools.chain('<root>', f, '</root>')

        parser = ET.XMLParser(encoding="us-ascii")
        root = ET.fromstringlist(it, parser=parser)
        
        text_tags = root.findall("SUM")

        for text_tag in text_tags:
          if text_tag.get("DOCREF") in allowed_keys:
            raw_texts[text_tag.get("DOCREF")] = text_tag.text
          
          if raw_texts.keys() == allowed_keys:
            break
      
    else:
      continue

  return raw_texts

def parse_summary_documents(num_files=20):
  directory = os.pardir
  document_path = os.path.join(directory, "data", "DUC2002_Summarization_Documents", "docs")
  summary_path = os.path.join(directory, "data", "summaries")
  
  raw_texts = parse_text(document_path, num_files)
  summ_texts = parse_summary(summary_path, num_files, raw_texts.keys())

  return [raw_texts, summ_texts]

def generate_lsa_score(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list

  lsa.system_summary_list = []
  lsa.reference_summary_list = []

  for text_id in raw_texts.keys():
    lsa.start_lsa(text_id, 5, raw_texts[text_id], summ_texts[text_id])

  system_summary_list    = lsa.system_summary_list
  reference_summary_list = lsa.reference_summary_list

def run_summary_algos(num_files=10):
  [raw_texts, summ_texts] = parse_summary_documents(num_files)
  
  # run baseline
  run_baseline(raw_texts, summ_texts, num_files)
  # run lsa
  run_lsa(raw_texts, summ_texts, num_files)


def run_baseline(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list
  system_summary_list = []
  reference_summary_list = []

  baseline.system_summary_list = []
  baseline.reference_summary_list = []

  for text_id in raw_texts.keys():
    baseline.run_baseline(text_id, 2, summ_texts[text_id], raw_texts[text_id])
    # lsa.start_lsa(text_id, 5, raw_texts[text_id], summ_texts[text_id])

  system_summary_list    = baseline.system_summary_list
  reference_summary_list = baseline.reference_summary_list

  print "BASELINE"  
  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))
  print "BASELINE END"  

def run_lsa(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list
  system_summary_list = []
  reference_summary_list = []

  generate_lsa_score(raw_texts, summ_texts, num_files)

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  print ('recall = ' + str(recall_list))
  print ('precision = ' + str(precision_list))
  print ('F = ' + str(F_measure_list))

def main():
  run_summary_algos()

if __name__ == "__main__":
  main()