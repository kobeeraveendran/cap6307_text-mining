# Automatic Summarization Evaluation

## Installation and Environment Setup

### Install PyROUGE

My workspace is Ubuntu 18.04, so I more or less followed the instructions listed <a href = "https://stackoverflow.com/a/57686103/9464919" target = "_blank">here</a>.

If you're on Windows instead, see the instructions listed <a href = "https://stackoverflow.com/a/47045437/9464919" target = "_blank">here</a> (note: I have not managed to test this on Windows, so using WSL is recommended if this method fails).

A quick overview of the steps/commands I used are:

1. Install `pyrouge` from source at it's official repository: 

```bash
# install pyrouge from source at it's official repository
cd ~/
git clone https://github.com/bheinzerling/pyrouge.git
cd pyrouge && python setup.py install
cd ~/

# install the ROUGE Perl toolkit from this unofficial repo, and to a different directory than the wrapper above
git clone https://github.com/andersjo/pyrouge.git rouge

# set the path to the ROUGE Perl toolkit, which PyROUGE will require
pyrouge_set_rouge_path ~/rouge/tools/ROUGE-1.5.5/

# if you don't have an XML parser for perl already, install one
sudo apt update
sudo apt upgrade
sudo apt install libxml-parser-perl

# if you run into errors involving the WordNet-2.0 exceptions db file like I did...

# delete it, then...
cd rouge/tools/ROUGE-1.5.5/data && rm WordNet-2.0.exc.db

# generate a new one
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db


# at this point, everything should be set up and run fine

# run the pyrouge tests with 
python -m pyrouge.test

# or see the next steps for running my code
```

### Running Multi-System Summary Evaluation

Before running this code, ensure that you have placed the folders `Human_Summaries` and `System_Summaries` in this directory (one level above `src/`). No modifications need to be made to the directory structure or any of the summary files/filenames.

Now, go to the source directory and run the summary evaluation:

```bash
cd src/
python rouge_summary_eval.py
```

The results for each system will be printed as they are made available, along with their respective execution times. Additionally, the results will be organized and saved to a JSON file in the previous directory (as `rouge_scores.json`) for easier viewing. For comparison and validity purposes, my results are also available in `rouge_scores_kr.json`.