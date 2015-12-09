from spacy.en import English
from nltk import stem
import pdb

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

def main():
  nlp = English(parser=True, tagger=True, entity=True)
  sentence = "He said that he lived in Paris and Berlin"
  sentence2 = "After sone time, he moved to London"
  root_token = generate_parse_tree(nlp, sentence)
  pretty_print_tree(root_token)
  root_token = generate_parse_tree(nlp, sentence2)
  pretty_print_tree(root_token)

if __name__ == "__main__":
  main()
