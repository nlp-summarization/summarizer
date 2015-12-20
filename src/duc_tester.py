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
import numpy
import math
from nltk.tokenize import sent_tokenize,word_tokenize

parent_dir = os.pardir
sys.path.insert(0, parent_dir + "/Rouge")
from PythonROUGE import *

import lsa
import baseline
import page_rank
import lex_rank

import sentence_paraphrase

import kenlm
import cloudpickle

global ROUGE_path, data_path
global reference_summary_list
global system_summary_list
global rouge_score_summaries

def parse_text(folder_path, start_offset, num_files=1):

  
  num_files_read = 0
  root_tag = ET.Element("root")

  raw_texts = {}

  for root, dirs, files in os.walk(folder_path):
    path = root.split('/')
    
    for file in files:
      
      if not file.startswith('.') and os.path.isfile(os.path.join(root, file)) and num_files_read < num_files:
        
        if num_files_read < start_offset:
          num_files_read = num_files_read + 1
          continue  

        try:
          e = xml.etree.ElementTree.parse(os.path.join(root, file))
        except xml.etree.ElementTree.ParseError as e:
          print (file)
          print e
          continue
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

def parse_summary_documents(start, end):
  num_files = end - start
  directory = os.pardir
  document_path = os.path.join(directory, "data", "DUC2002_Summarization_Documents", "docs")
  summary_path = os.path.join(directory, "data", "summaries")
  
  raw_texts = parse_text(document_path, start, end)
  summ_texts = parse_summary(summary_path, num_files, raw_texts.keys())

  return [raw_texts, summ_texts]

def generate_lsa_score(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list

  lsa.system_summary_list = []
  lsa.reference_summary_list = []

  ranks = []
  for text_id in raw_texts.keys():
    ranks.append(lsa.start_lsa(text_id, 5, raw_texts[text_id], summ_texts[text_id]))

  system_summary_list    = lsa.system_summary_list
  reference_summary_list = lsa.reference_summary_list

  return ranks

def run_summary_algos(start, end, lambda1, lambda2, lambda3):
  num_files = end - start
  [raw_texts, summ_texts] = parse_summary_documents(start, end)


  ranks = {}
  # ranks["lex"] = run_lex_rank(raw_texts, summ_texts, num_files)
  # ranks["baseline"] = run_baseline(raw_texts, summ_texts, num_files)
  # ranks["lsa"] = run_lsa(raw_texts, summ_texts, num_files)
  # ranks["page_rank"] = run_page_rank(raw_texts, summ_texts, num_files)

  
  # final_ranks = rank_interpoliation(ranks)
  # sentences = sent_tokenize(raw_texts["AP880911-0016"])
  # limit = 1
  # summary = ""
  # for i in xrange(limit):
  #   summary = summary + " " + sentences[final_ranks[i]]
  # print summary



  # print final_ranks

  ranks = {}
  ranks["lex"] = run_lex_rank(raw_texts, summ_texts, num_files)
  ranks["baseline"] = run_baseline(raw_texts, summ_texts, num_files)
  ranks["lsa"] = run_lsa(raw_texts, summ_texts, num_files)
  ranks["page_rank"] = run_page_rank(raw_texts, summ_texts, num_files)

  system_summary_list = []
  reference_summary_list = []
  first_sentences = []

  for doc_id, text_id in enumerate(raw_texts.keys()):
  # for doc_id in range(0, num_files):
    rank = {}
    rank["lex"] = ranks["lex"][doc_id]
    #rank["baseline"] = ranks["baseline"][doc_id]
    rank["lsa"] = ranks["lsa"][doc_id]
    rank["page_rank"] = ranks["page_rank"][doc_id]
    final_ranks = rank_interpoliation(rank, lambda1, lambda2, lambda3)

    sentences = sent_tokenize(raw_texts[text_id])
    
    limit = 5
    summary = ""
    for i in xrange(limit):
      if i == 0:
        first_sentences.append(sentences[final_ranks[i]])
      summary = summary + " " + sentences[final_ranks[i]]
    
    '''
    print "\n################# FINAL SUMMARY ##############################\n" + text_id
    print summary

    print "\n################# REF SUMMARY ##############################\n" + text_id
    print summ_texts[text_id]
    '''
    

    system_summary = summary
    # call compression
    reference_summary = summ_texts[text_id]

    sys_dir = os.pardir + "/test-summarization/system/" + text_id + "_" + "system.txt"
    ref_dir = os.pardir + "/test-summarization/reference/" + text_id + "_" + "reference.txt"

    write_to_file(ref_dir, reference_summary)
    reference_summary_list.append([ref_dir])

    # write system summary to file
    write_to_file(sys_dir, system_summary)
    system_summary_list.append(sys_dir)

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  
  score_summary = {}
  score_summary["recall"] = str(recall_list)
  score_summary["precision"] = str(precision_list)
  score_summary["F"] = str(F_measure_list)

  rouge_score_summaries["FINAL"] = score_summary

  #return ranks
  return first_sentences

def write_to_file(filename, summary):
  with codecs.open(filename, 'w', encoding='utf8') as f:
    f.write(str(summary.encode('ascii', errors='ignore')))

def rank_interpoliation(ranks, lambda1, lambda2, lambda3):
  
  scores = {}
  weights = []
  for algo, rank in ranks.iteritems():
    

    if algo == "lex" and len(rank) > 0: weights.append(lambda2)
    #elif algo == "baseline" and len(rank) > 0: weights.append(0.1)
    elif algo == "page_rank" and len(rank) > 0: weights.append(lambda1)
    elif algo == "lsa" and len(rank) > 0: weights.append(lambda3)
    else: weights.append(0) 

    for position, sentence_id in enumerate(rank):
      if sentence_id in scores:
        scores[sentence_id].append(position)
      else:
        scores[sentence_id] = [position]

  avg_var = []

  for (sentence_id, positions) in scores.iteritems():
    if len(positions) == 3:
      [average, variance] = weighted_avg_and_std(positions, weights)
      avg_var.append((sentence_id, round(average), variance))
  
  
  avg_var = sorted(avg_var, key = lambda x : (x[1], x[2]))
  
  final_ranks = []
  for s in avg_var:
    final_ranks.append(s[0])

  return final_ranks

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    
    average = numpy.average(values, weights=weights)
    variance = numpy.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))



def run_page_rank(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list, rouge_score_summaries
  system_summary_list = []
  reference_summary_list = []

  page_rank.system_summary_list = []
  page_rank.reference_summary_list = []

  ranks = []
  for text_id in raw_texts.keys():
    ranks.append(page_rank.start_page_rank(text_id, 3, summ_texts[text_id], raw_texts[text_id]))

  system_summary_list    = page_rank.system_summary_list
  reference_summary_list = page_rank.reference_summary_list

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)

  score_summary = {}
  score_summary["recall"] = str(recall_list)
  score_summary["precision"] = str(precision_list)
  score_summary["F"] = str(F_measure_list)
  rouge_score_summaries["Page Rank"] = score_summary

  return ranks

def run_baseline(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list, rouge_score_summaries
  system_summary_list = []
  reference_summary_list = []

  baseline.system_summary_list = []
  baseline.reference_summary_list = []

  ranks = []
  for text_id in raw_texts.keys():
    ranks.append(baseline.run_baseline(text_id, 2, summ_texts[text_id], raw_texts[text_id]))

  system_summary_list    = baseline.system_summary_list
  reference_summary_list = baseline.reference_summary_list

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)

  score_summary = {}
  score_summary["recall"] = str(recall_list)
  score_summary["precision"] = str(precision_list)
  score_summary["F"] = str(F_measure_list)
  rouge_score_summaries["Baseline"] = score_summary

  return ranks

def run_lex_rank(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list, rouge_score_summaries
  system_summary_list = []
  reference_summary_list = []

  lex_rank.system_summary_list = []
  lex_rank.reference_summary_list = []

  ranks = []
  for text_id in raw_texts.keys():
    ranks.append(lex_rank.start_lex_rank(text_id, 2, summ_texts[text_id], raw_texts[text_id]))


  system_summary_list    = lex_rank.system_summary_list
  reference_summary_list = lex_rank.reference_summary_list

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)

  score_summary = {}
  score_summary["recall"] = str(recall_list)
  score_summary["precision"] = str(precision_list)
  score_summary["F"] = str(F_measure_list)
  rouge_score_summaries["Lex Rank"] = score_summary

  return ranks

def run_lsa(raw_texts, summ_texts, num_files=20):
  global system_summary_list, reference_summary_list, rouge_score_summaries
  system_summary_list = []
  reference_summary_list = []

  ranks = generate_lsa_score(raw_texts, summ_texts, num_files)

  recall_list,precision_list,F_measure_list = PythonROUGE(parent_dir, system_summary_list,reference_summary_list, 1)
  
  score_summary = {}
  score_summary["recall"] = str(recall_list)
  score_summary["precision"] = str(precision_list)
  score_summary["F"] = str(F_measure_list)

  rouge_score_summaries["LSA"] = score_summary
  return ranks

def main():
  global rouge_score_summaries
  rouge_score_summaries = {}
  #lambda1 = 0.4; 
  #lambda2 = 0.4;
  #lambda3 = 0.2;
  all_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9, 1.0];

  lambdas = {}
  for lambda1 in all_lambda:
    for lambda2 in all_lambda:
      for lambda3 in all_lambda:
        if (abs(1.0 - (lambda1 + lambda2 + lambda3)) < 0.0001):
          lambdas[(lambda1, lambda2, lambda3)] = 1


  '''
  for values in lambdas:
    lambda1 = values[0]
    lambda2 = values[1]
    lambda3 = values[2]
    system_summary = run_summary_algos(0,160,lambda1, lambda2, lambda3)
    print values, ' ' , ('F = ' + str(rouge_score_summaries['FINAL']['F']) + "\n")
  #paraphrase_dict = sentence_paraphrase.get_paraphrase_dict()
  '''

  '''
  for i, s in enumerate(system_summary):
    [score_sent, length_sent, inter_sent] = sentence_paraphrase.get_compressed_sentence(s, paraphrase_dict)
    print "####### ORIGINAL #######"
    print len(s), " ", s
    print "####### COMPRESSED #######"
    print len(score_sent), " ", score_sent, " ---- ", len(s)
    print ""
    '''

  lambda1 = 0.4
  lambda2 = 0.3
  lambda3 = 0.3
  # Training set
  system_summary = run_summary_algos(160,200,lambda1, lambda2, lambda3)

  '''
  cleaned_sentences = []
  for s in system_summary:
    words = word_tokenize(s)
    cleaned_sentences.append(' '.join(words))

  paraphrase_dict = sentence_paraphrase.get_paraphrase_dict()
  average_compression_percentage = 0.0
  for i, c in enumerate(cleaned_sentences):
    [score_sent, length_sent, inter_sent] = sentence_paraphrase.get_compressed_sentence(c, paraphrase_dict)
    average_compression_percentage = average_compression_percentage + (1.0/float(len(cleaned_sentences)))*(float(len(score_sent))/float(len(c)))
    if len(score_sent) < 145 and len(c) >= 145:
      print i
      print "original sentence:   ", c
      print "compressed sentence: ", score_sent
      print ""
    #print "completion: ", (float(i)/200.0) * 100.0, ' %'

  #print "compression: ", (1.0 - average_compression_percentage)*100.0, '%'
  '''
  
  for model in rouge_score_summaries.keys():
    print ("########## Model : " + str(model) + " ################ \n")
    print ('recall = ' + str(rouge_score_summaries[model]['recall']))
    print ('precision = ' + str(rouge_score_summaries[model]['precision']))
    print ('F = ' + str(rouge_score_summaries[model]['F']) + "\n")
  
  
  

if __name__ == "__main__":
  main()