#!/usr/bin/python
import sys
import pdb
import xml.etree.ElementTree
import os, glob

def list_files(num_files=10):
  num_files_read = 0
  for root, dirs, files in os.walk("."):
    path = root.split('/')
    # print (len(path) - 1) *'---' , os.path.basename(root)       
    for file in files:
      if file.endswith(".xml") and num_files_read < num_files:

        e = xml.etree.ElementTree.parse(os.path.join(root, file))
        
        abstract_text = ''

        abstract_tag = e.findall("body/body.head/abstract/p")
        if not len(abstract_tag) > 0:
          continue

        for paragraph in abstract_tag:
          abstract_text = abstract_text + paragraph.text
        
        full_text = ''
        for paragraph in e.findall("body/body.content/block[@class='full_text']/p"):
          full_text = full_text + paragraph.text

        num_files_read = num_files_read + 1
        # e.clear()
        print(os.path.join(root, file))
        print abstract_text.encode('utf-8').strip()
        print "\n\n\n"
        print full_text.encode('utf-8').strip()


def main():  
  list_files(20)
  

if __name__ == "__main__":
  main()