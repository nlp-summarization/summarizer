#!/usr/bin/python
import sys
import pdb
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import os, glob

def parse_files(folder_path, num_files=1000000):

  num_files_read = 0
  root_tag = ET.Element("root")

  for root, dirs, files in os.walk(folder_path):
    path = root.split('/')
    
    for file in files:
      if file.endswith(".xml") and num_files_read < num_files:

        try:
          e = xml.etree.ElementTree.parse(os.path.join(root, file))
        except xml.etree.ElementTree.ParseError:
          continue

        abstract_text = ''

        abstract_tag = e.findall("body/body.head/abstract/p")
        if not len(abstract_tag) > 0:
          continue

        for paragraph in abstract_tag:
          if paragraph.text != None:
            abstract_text = abstract_text + paragraph.text
        
        if len(abstract_text) > 140:
          continue

        full_text = ''
        for paragraph in e.findall("body/body.content/block[@class='full_text']/p"):
          if paragraph.text != None:
            full_text = full_text + paragraph.text

        num_files_read = num_files_read + 1
        root_tag = make_xml(root_tag, os.path.splitext(file)[0], abstract_text, full_text)
        print_progress(num_files_read)

  tree = ET.ElementTree(root_tag)
  tree.write(parse_file_name)

def print_progress(progress):
  global total_files

  sys.stdout.write("Percent files read: %d%%   \r" % ( round(float(progress*100.00)/float(total_files)) ))
  sys.stdout.flush()

def make_xml(root_tag, a_id, summary, text):
  global parse_file_name
  article_id = ET.Element("article", id=a_id)
  
  summary_tag = ET.SubElement(article_id, "summary")
  summary_tag.text = summary 
  
  text_tag = ET.SubElement(article_id, "text")
  text_tag.text = text
  
  root_tag.append(article_id)
  return root_tag

def main():

  directory = "."
  if len(sys.argv) > 1:
    directory = argv[1]
  
  global parse_file_name, total_files
  total_files = 1000000
  parse_file_name = "parsed_summaries.xml"
  parse_files(directory, total_files)
  

if __name__ == "__main__":
  main()
