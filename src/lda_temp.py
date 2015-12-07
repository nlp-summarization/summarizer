from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]

dictionary   = corpora.Dictionary(texts)
corpus       = [dictionary.doc2bow(text) for text in texts]
tfidf        = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# I can print out the topics for LSA
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus]

for l,t in izip(corpus_lsi,corpus):
  print l,"#",t
print
for top in lsi.print_topics(2):
  print top
