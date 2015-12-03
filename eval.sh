#!/bin/bash

ulimit -S -n 1000000

pd=$(dirname "$dir")

#find "$pd/test-summarization/reference" -name "*.txt" \
#    | awk '!(++cnt%10000) {"mkdir '$pd'/test-summarization/reference/sub_" ++d|getline}'
#
#for f in "$pd/test-summarization/reference/*.txt" do echo "$f"; done

cd "$pd/test-summarization/reference"
i=0; for f in *; do d=dir_$(printf %03d $((i/10000+1))); mkdir -p $d; mv "$f" $d; let i++; done

#cd Rouge
#java -jar rouge2.0.jar
#cd ..
