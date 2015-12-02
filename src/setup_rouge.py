#!/usr/bin/python
import sys
import pdb
import xml.etree.cElementTree as ET
import os, glob
import re

def parse_files():

  e = ET.parse("../data/test.xml")
  r = e.getroot()

  # Reference
  for t in r.findall("article"):
      article_id = t.get('id')
      summary = t.find("summary").text
      summary = re.sub("\. ","\n", str(summary))
      reference_file_name = "../test-summarization/reference/" + str(article_id) + "_reference.txt"
      system_file_name = "../test-summarization/system/" + str(article_id) + "_system.txt"
      write_to_file(reference_file_name, summary)
      write_to_file(system_file_name, summary)

def write_to_file(filename, summary):
  f = open(filename, "w")
  f.write(summary)
  f.close()

def main():

  parse_files()
  
if __name__ == "__main__":
  main()
