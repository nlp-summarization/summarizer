# from pulp import LpVariable
import pulp
from spacy.en import English
from nltk import stem
import pdb
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
from nltk.tokenize import word_tokenize
import copy
def save_word_dict_full(text):
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

'''
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
  '''

#[dictionary, proc_text, sentences] = save_word_dict(document)
[dictionary, proc_text, sentences, tokenized_sentences] = save_word_dict_full(document)




#sentences_with_word_scores = [[('thoma', 0.1068085771573899), ('anderson', 0.0356028590524633), ('man', 0.08891846327106773), ('live', 0.1068085771573899)], [('dai', 0.08736120966474198), ('averag', 0.08736120966474198), ('programm', 0.02912040322158066), ('night', -0.17854640387819373), ('hacker', 0.02912040322158066), ('known', -0.5300571488650518), ('neo', -0.4166082757157854)], [('neo', -0.17570712204508632), ('question', -0.0387375079278687), ('realiti', 0.0387375079278687), ('truth', -0.16085630161861778), ('far', 0.19368753963934351), ('imagin', 0.19368753963934351)], [('neo', -0.09643169940472064), ('find', 0.1828121949901654), ('target', -0.03108737703232642), ('polic', -0.4821584970236032), ('contact', -0.09326213109697926), ('morpheu', -0.21761163922628493), ('legendari', -0.1554368851616321), ('hacker', -0.09326213109697926), ('brand', -0.03108737703232642), ('terrorist', -0.09326213109697926), ('govern', -0.21761163922628493)], [('morpheu', -0.12727346777760223), ('awaken', 0.18793057868387), ('neo', -0.12727346777760223), ('real', -0.6363673388880111), ('world', -0.3818204033328067), ('ravag', -0.3818204033328067), ('wasteland', -0.08138813472441814), ('human', -1.14546120999842), ('captur', -0.2441644041732544), ('race', -0.569716943070927), ('machin', -0.8952694819685996), ('live', -1.0580457514174357), ('human', -0.7324932125197632), ('bodi', -1.7091508292127808), ('heat', -1.5463745597639447), ('electrochem', -1.8719270986616172), ('energi', -1.7091508292127808), ('imprison', -1.2208220208662721), ('mind', -1.3835982903151083), ('artifici', -1.7091508292127808), ('realiti', -1.036391775548884), ('known', -1.7091508292127808)], [('rebel', -0.5124920745309579), ('machin', -0.7848222633253705), ('neo', -0.11211746618933864), ('return', 0.11211746618933864), ('matrix', -0.17000928236682844), ('confront', -0.05666976078894281), ('agent', -0.17000928236682844), ('super', -0.3966883255225997), ('power', -0.28334880394471407), ('program', -0.17000928236682844), ('devot', -0.28334880394471407), ('snuf', -0.5100278471004853), ('neo', -0.05666976078894281), ('entir', -0.8500464118341422), ('human', -0.8500464118341422)]]
sentences_with_word_scores = [[('Thomas', -0.0355300469051906), ('A.', -0.0355300469051906), ('Anderson', -0.011843348968396866), ('is', 0.15742120627124823), ('a', -0.0355300469051906), ('man', -0.011843348968396866), ('living', -0.0355300469051906), ('two', -1.265978051806257), ('lives', -0.05921674484198433)], [('By', -0.15840803580887503), ('day', -0.5438932363613784), ('he', -0.08730610339466148), ('is', 0.08730610339466148), ('an', -0.5438932363613784), ('average', -0.04861251294134389), ('computer', -0.38669420275735944), ('programmer', -0.12889806758578648), ('and', -0.08730610339466148), ('by', -0.016204170980447963), ('night', -0.04861251294134389), ('a', -0.38669420275735944), ('hacker', -0.08730610339466148), ('known', -0.2619183101839844), ('as', -0.08102085490223981), ('Neo', -0.11342919686313574)], [('Neo', -0.16804813535720556), ('has', -0.1856325153813483), ('always', -0.05880398858402472), ('questioned', 0.16804813535720556), ('his', -0.17641196575207416), ('reality', -0.05880398858402472), (',', -0.16804813535720556), ('but', -0.05880398858402472), ('the', -0.29401994292012357), ('truth', -0.17641196575207416), ('is', 0.08239202283195057), ('far', -0.5671303098530757), ('beyond', -0.17641196575207416), ('his', -0.17641196575207416)], [('Neo', -0.16859859054883874), ('finds', 0.17176006644136274), ('himself', -0.5576400996620442), ('targeted', -0.14631893597888956), ('by', -0.3447525795831849), ('the', -0.27269765555668046), ('police', -0.5745876326386414), ('when', -0.6859859054883876), ('he', -0.8429929527441937), ('is', -0.30618231253308925), ('contacted', -0.18370938751985355), ('by', -0.18370938751985355), ('Morpheus', -0.42865523754632495), (',', -0.18370938751985355), ('a', -0.30618231253308925), ('legendary', -0.5745876326386414), ('computer', -0.30618231253308925), ('hacker', -0.18370938751985355), ('branded', -0.06123646250661785), ('a', -0.30618231253308925)], [('Morpheus', -0.18397494664944425), ('awakens', 0.1517190373780683), ('Neo', -0.1517190373780683), ('to', -0.19198747332472213), ('the', -0.6953810415075189), ('real', -0.8476905207537594), ('world', -0.4551571121342049), (',', -0.14654279968194925), ('a', -0.5465427996819493), ('ravaged', -0.4551571121342049), ('wasteland', -0.1517190373780683), ('where', -0.606285707644101), ('most', -0.606285707644101), ('of', -0.8487999907017413), ('humanity', -1.0913142737593817), ('have', -0.606285707644101), ('been', -0.606285707644101), ('captured', -0.36377142458646056), ('by', -0.606285707644101), ('a', -0.36377142458646056), ('race', 0.25360002789477626), ('of', -0.8487999907017413), ('machines', -1.333828556817022), ('that', -1.8188571229323027), ('live', -1.5763428398746624), ('off', -1.8188571229323027), ('of', -0.8487999907017413), ('the', -0.606285707644101), ('humans', -3.1860997849394344), ("'", -2.7889142551628643), ('body', -2.546399972105224), ('heat', -2.3038856890475836), ('and', -1.8188571229323027), ('electrochemical', -2.7889142551628643), ('energy', -3.1860997849394344), ('and', -1.8188571229323027), ('who', -2.0613714059899433), ('imprison', -1.8188571229323027), ('their', -2.3038856890475836), ('minds', -2.0613714059899433), ('within', -2.579223635427161), ('an', -2.546399972105224)], [('As', -0.1886023584031412), ('a', -0.8860235840314119), ('rebel', -0.34001177007860894), ('against', -0.7833431417321741), ('the', -0.4930554495570539), ('machines', -1.2225694073449689), (',', -0.06264280164857502), ('Neo', -0.1313214008242875), ('must', -0.1313214008242875), ('return', 0.08799002917072235), ('to', -0.08799002917072235), ('the', -0.43995014585361175), ('Matrix', -0.26397008751216705), ('and', -0.08799002917072235), ('confront', -0.08799002917072235), ('the', -0.43995014585361175), ('agents', -0.26397008751216705), (':', -0.26397008751216705), ('super-powerful', 0.08799002917072235), ('computer', -0.43995014585361175), ('programs', -0.26397008751216705), ('devoted', -0.43995014585361175), ('to', -0.08799002917072235), ('snuffing', -0.7919102625365011), ('out', -0.9678903208779459)]]
#sentences_with_word_scores = [[('Thomas', 0.0355300469051906), ('A.', 0.0355300469051906), ('Anderson', 0.15742120627124823), ('is', 0.011843348968396866), ('a', 0.0355300469051906), ('man', 0.011843348968396866), ('living', 0.0355300469051906), ('two', 1.265978051806257), ('lives', 0.05921674484198433)], [('By', 0.15840803580887503), ('day', 0.5438932363613784), ('he', 0.1812977454537928), ('is', 0.016204170980447935), ('an', 0.26191831018398437), ('average', 0.38669420275735944), ('computer', 0.26191831018398437), ('programmer', 0.08730610339466147), ('and', 0.016204170980447935), ('by', 0.016204170980447935), ('night', 0.38669420275735944), ('a', 0.38669420275735944), ('hacker', 0.016204170980447935), ('known', 0.26191831018398437), ('as', 0.08102085490223968), ('Neo', 0.6111427237626302)], [('Neo', 0.16804813535720556), ('has', 0.1856325153813483), ('always', 0.05880398858402472), ('questioned', 0.05880398858402472), ('his', 0.5041444060716167), ('reality', 0.05880398858402472), (',', 0.05880398858402472), ('but', 0.05880398858402472), ('the', 0.29401994292012357), ('truth', 0.24717606849585172), ('is', 0.05880398858402472), ('far', 0.29401994292012357), ('beyond', 0.5041444060716167), ('his', 0.34027818591184544)], [('Neo', 0.16859859054883874), ('finds', 0.17176006644136274), ('himself', 0.5576400996620442), ('targeted', 0.03895680793666864), ('by', 0.3447525795831849), ('the', 1.024232551852227), ('police', 0.5745876326386414), ('when', 0.8429929527441937), ('he', 0.6859859054883876), ('is', 0.30618231253308925), ('contacted', 0.18370938751985355), ('by', 0.18370938751985355), ('Morpheus', 0.42865523754632495), (',', 0.3447525795831849), ('a', 0.30618231253308925), ('legendary', 0.30618231253308925), ('computer', 0.30618231253308925), ('hacker', 0.18370938751985355), ('branded', 0.06123646250661785), ('a', 0.30618231253308925)], [('Morpheus', 0.18397494664944425), ('awakens', 0.19198747332472213), ('Neo', 0.1517190373780683), ('to', 0.1695381041507519), ('the', 0.7585951868903416), ('real', 0.695381041507519), ('world', 0.4551571121342049), (',', 0.18218093322731643), ('a', 0.4396283990458477), ('ravaged', 0.4551571121342049), ('wasteland', 0.1517190373780683), ('where', 0.6062857076441011), ('most', 0.6062857076441011), ('of', 0.8487999907017415), ('humanity', 1.091314273759382), ('have', 0.6062857076441011), ('been', 0.6062857076441011), ('captured', 0.3637714245864606), ('by', 0.6062857076441011), ('a', 0.3637714245864606), ('race', 0.8487999907017415), ('of', 1.062033261646478), ('machines', 1.3338285568170223), ('that', 1.8188571229323032), ('live', 1.5763428398746626), ('off', 1.8188571229323032), ('of', 0.8487999907017415), ('the', 0.6062857076441011), ('humans', 2.5463999721052244), ("'", 2.7889142551628647), ('body', 2.5463999721052244), ('heat', 2.8826617101832976), ('and', 1.8188571229323032), ('electrochemical', 2.7889142551628647), ('energy', 2.5463999721052244), ('and', 1.8188571229323032), ('who', 0.61588578203017), ('imprison', 1.8188571229323032), ('their', 2.303885689047584), ('minds', 2.0613714059899437), ('within', 2.0613714059899437), ('an', 3.1860997849394344)], [('As', 0.1886023584031412), ('a', 0.8860235840314119), ('rebel', 0.4700058850393045), ('against', 0.5666862834643482), ('the', 0.8732638623892636), ('machines', 0.6902776293798755), (',', 0.1313214008242875), ('Neo', 0.1313214008242875), ('must', 0.06264280164857502), ('return', 0.08799002917072235), ('to', 0.08799002917072235), ('the', 0.43995014585361175), ('Matrix', 0.26397008751216705), ('and', 0.08799002917072235), ('confront', 0.08799002917072235), ('the', 0.43995014585361175), ('agents', 0.26397008751216705), (':', 0.26397008751216705), ('super-powerful', 0.08799002917072235), ('computer', 0.43995014585361175), ('programs', 0.26397008751216705), ('devoted', 0.43995014585361175), ('to', 0.08799002917072235), ('snuffing', 0.7919102625365011), ('out', 0.9678903208779459)]]

x = pulp.LpVariable.dict('x_%s', dictionary.values(), lowBound =0)#, cat = pulp.LpInteger)
#print(x)

prob = pulp.LpProblem("Summarizer", pulp.LpMaximize)

word2lpvar = {}
# objective function

# prob += sum([x[w][b]*costs[w][b] for (w,b) in routes]), "Sum_of_Transporting_Costs"
sum_list = []

for sent_id, sent in enumerate(sentences_with_word_scores):
  for (word, score) in sent:
    if(score > 0):
      sum_list.append(x[word]*score) 

prob += sum(sum_list), "SUM_OF_SELECTED_WORDS"
# print (sum_list)

for sent_id, sent in enumerate(sentences_with_word_scores):
  sum_list = []
  
  sums = 0
  for (word, score) in sent:
    if(score > 0):
      sum_list.append(x[word]*score) 
      sums = sums + score
      print (sums)

  #print (sum_list)
  #if(len(sum_list) > 0):
  prob += sum(sum_list) <= sums , "SUM_OF_WORDS_IN_SENT_%s"%str(sent_id)
  #print("SUM_OF_WORDS_IN_SENT_%s"%str(sent_id))


# print (x["realiti"])
prob.solve()




print (prob)

for v in prob.variables():
  print (v.name, "=", v.varValue)

print ("Status:", pulp.LpStatus[prob.status])

print ("Total Weigt of Words = ", prob.objective.value())

print ("done")

# pdb.set_trace()

# LpVariable("example", None, 100)



