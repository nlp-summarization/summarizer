#!/usr/bin/python
import sys
import pdb
import os
from pyrouge import Rouge155

def test():
	# print "hello world"
	r = Rouge155()
	r.system_dir = '/Users/chandradhar/WebProjects/NLP-FinalProject/Rouge/test-summarization/system/'
	r.model_dir = '/Users/chandradhar/WebProjects/NLP-FinalProject/Rouge/test-summarization/reference/'

	r.system_filename_pattern = 'some_name.(\d+).txt'
	r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'

	output = r.convert_and_evaluate()
	print(output)

def main():
  test()
  
if __name__ == "__main__":
  main()

#pyrouge_set_rouge_path /Users/chandradhar/WebProjects/ROUGE-perl