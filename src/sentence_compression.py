from spacy.en import English
import cloudpickle
import pickle
from nltk import stem
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import pdb
from pulp import LpVariable
import sys
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import os, glob
import codecs
from gensim.summarization import summarize
from gensim.parsing import preprocess_string
from gensim import corpora, models, similarities
from gensim.models import ldamodel
from nltk import tokenize
import copy

def generate_parse_tree(nlp, sentence):
  # porter stemmer changes meaning of the sentence and reduces
  # quality of the dependency tree. Don't use it.
  #porter = stem.porter.PorterStemmer()
  #sentence = ' '.join([porter.stem(i).lower() for i in sentence.split()])
  doc = nlp(sentence)
  root_token = None
  for token in doc:
    if token.head is token:
      root_token = token
      break
  return root_token

def pretty_print_tree(token):
  pretty_print_tree_helper(token,0)

def pretty_print_tree_helper(token,level):
  if token.head is token:
    print (token.text + " (" + token.tag_ + ")")
  for t in  token.children:
    print(('\t' * level) + "|__" + t.text + " (" + t.tag_ + ")")
    pretty_print_tree_helper(t,level + 1)

def save_word_dict(text):
  proc_text = []
  
  sentences = text 
  sentences = tokenize.sent_tokenize(sentences)

  tokenized_sentences = []
  for s in sentences:
    tokenized_sentences.append(word_tokenize(s))
  
  for sentence in sentences:
    proc_sentence = preprocess_string(sentence) # ' '.join(preprocess_string(sentence))

    if(len(proc_sentence) == 0):
      continue
    proc_text.append(proc_sentence)

  dictionary = corpora.Dictionary(tokenized_sentences)
  # dictionary.save(os.pardir + '/data/text.dict')
  return [dictionary, proc_text, sentences, tokenized_sentences]

def sentence_compression(nlp):
  document = "Thomas A. Anderson is a man living two lives. By day he is an " + \
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

  [dictionary, proc_text, sentences, tokenized_sentences] = save_word_dict(document)
  raw_corpus = [dictionary.doc2bow(t) for t in tokenized_sentences]
  tfidf        = models.TfidfModel(raw_corpus)
  tfidf.save("tfidf_model")
  corpus_tfidf = tfidf[raw_corpus]

  #print(tokenized_sentences)

  
  #print (dictionary)
  #print("tfidf:")
  #for d in corpus_tfidf:
  #  print (d)
  #print()

  #print("processed text:")
  #print(proc_text)
  #print()

  #print("depths:")
  depths = []
  for i,s in enumerate(sentences):
    root_node = generate_parse_tree(nlp, s)
    t_d = get_depth(root_node)
    #print(str(i) + ": ", end="")
    #print(t_d)
    depths.append(t_d)

  #print ("depths:")
  #print (depths)

  # PROC_TEXT AND CORPUS_TFIDF ARE ORDERED
  # DEPTHS IS NOT ORDERED
  weights = []
  for i, (text, tf_idf, depth) in enumerate(zip(tokenized_sentences, corpus_tfidf, depths)):
    temp_list = []
    for word, freq_idf in zip(text, tf_idf):
      freq = freq_idf[1]
      weight = abs((freq - 0.4) * (depth_for_word(word, depth) - 0.5))
      temp_list.append((word,weight))
    weights.append(temp_list)

  #print("weights:")
  #for i,w in enumerate(weights):
  #  print(str(i) + ": ", end="")
  #  print (w)
  print(weights)

    #print(i)
    #print(proc_text)
    #print(tf_idf)
    #print(depth)
    #print()

def depth_for_word(word, depth_list):
  for d in depth_list:
    w = d[0]
    if w == word:
      return d[1]
  return 0



def get_depth(root_node):
  queue = []
  depths = []
  depth_val = 0
  queue.append({"node": root_node, "depth": depth_val})
  depths.append((root_node.text, depth_val))

  while(len(queue) > 0):
    token = queue.pop(0)
    current_depth = token["depth"]
    for t in token["node"].children:
      queue.append({"node": t, "depth": current_depth+1})
      depths.append((t.text, current_depth+1))
  return depths

def main():
  '''
  nlp = None
  if os.path.isfile('nlp.p'):
    print("loading from file")
    f = open('nlp.p', 'rb')
    nlp = pickle.load(f)
  else:
    nlp = English(parser=True, tagger=True, entity=False)
    f = open('nlp.p', 'wb')
    p = cloudpickle.CloudPickler(f)
    p.dump(nlp)
  '''
  nlp = English(parser=True, tagger=True, entity=False)

  #sentence = "He said that he lived in Paris and Berlin"
  #sentence2 = "After sone time, he moved to London"
  #root_token = generate_parse_tree(nlp, sentence)
  #pretty_print_tree(root_token)
  #root_token = generate_parse_tree(nlp, sentence2)
  #pretty_print_tree(root_token)
  #x = LpVariable("x",0,3)
  sentence_compression(nlp)

if __name__ == "__main__":
  main()
