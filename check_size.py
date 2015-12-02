#!/usr/bin/python
import sys
import pdb
import os, glob
from xml.dom.minidom import parseString

file = open('parsed_summaries.xml','r')
data = file.read()
file.close()
dom = parseString(data)
print(len(dom.getElementsByTagName('text')))
