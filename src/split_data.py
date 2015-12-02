#!/usr/bin/python
import sys
import pdb
import xml.etree.cElementTree as ET
from xml.dom.minidom import parse,parseString
import os, glob
import re

def parse_files():

  e = ET.parse('../data/parsed_summaries.xml')
  r = e.getroot()
  size = get_size()
  training_size = int(size*0.8)
  test_size = size - training_size

  root_tag_training = ET.Element("root")
  root_tag_test = ET.Element("root")

  for i, t in enumerate(r.findall("article")):
      article_id = t.get('id')
      summary = t.find("summary").text
      
      if(re.search("^Correction*", str(summary)) 
              or re.search("\*Correction*", str(summary))
              or re.search("^A Correction*", str(summary))
              or re.search("\* Correction*", str(summary))
              or re.search("\*\* Correction*", str(summary))
              or re.search("\*\*Correction*", str(summary))):
        continue

      text = t.find("text").text

      text = re.sub("Correction:.*$", "", str(text))

      if i < training_size:
        root_tag_training = make_xml(root_tag_training, article_id, summary, text)
      else:
        root_tag_test = make_xml(root_tag_test, article_id, summary, text)

  tree = ET.ElementTree(root_tag_training)
  tree.write("../data/training.xml")
  tree = ET.ElementTree(root_tag_test)
  tree.write("../data/test.xml")

def make_xml(root_tag, a_id, summary, text):
  global parse_file_name
  article_id = ET.Element("article", id=a_id)
  
  summary_tag = ET.SubElement(article_id, "summary")
  summary_tag.text = summary 
  
  text_tag = ET.SubElement(article_id, "text")
  text_tag.text = text
  
  root_tag.append(article_id)
  return root_tag

def get_size():

  file = open('../data/parsed_summaries.xml','r')
  data = file.read()
  file.close()
  dom = parseString(data)
  return len(dom.getElementsByTagName('text'))

def main():

  parse_files()
  

if __name__ == "__main__":
  main()
