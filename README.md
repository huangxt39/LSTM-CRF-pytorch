# Introduction #
This is a simple implementation of the paper [(Lample et al., 2016)](http://arxiv.org/abs/1603.01360), with all the code in a single file "lstm-crf-pytorch.py". This implementation supports batch training and batch inference.

# Quick Start #

First, you need to download the [GloVe Word Embedding](http://nlp.stanford.edu/data/glove.6B.zip) and unzip it. Then put the file "glove.6B.100d.txt" into the directory "dataset_ner"

Now you may run it on a GPU via

    python lstm-crf-pytorch.py

Then the model will be trained from scratch and be evaluated every epoch, using CoNLL evaluation script.

If you want to run it on CPU, just remove all the ".cuda()" in the code and use the command above. This may sound sort of primitive, but it's for simplicity.

# Result #

The model achieves 90.08 F1 score on test split, which is comparable to (Lample et al., 2016). You may achieve better results by selecting better hyperparameters.