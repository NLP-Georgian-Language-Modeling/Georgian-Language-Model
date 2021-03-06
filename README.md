# Abstract

We have implemented a Georgian language model that can generate text according to the given context.

We tried several model types to solve the problem, but the best result was given by 2 layers bidirectional LSTM. Which takes into account both the previous and upcoming context in the learning process and gives good results despite the limited computational resources.

For comparison, we calculated the perplexity on the test data set, which was 415,000 on the static n-gram model, while it was 729 on the transformer model and just 24 on the 2 layers bidirectional LSTM.

As a result, you can use our model for Georgian text generation, as well as in the case of a large computing resource, you can train the model on a larger Georgian text to get better results.


# Notes

* For more information about our project, you can read report.pdf
* As Github has a restriction on file size, we attach our project [google drive link](https://drive.google.com/drive/folders/1raR7yj-4kIZMEsJ4e4v46WR8J3LXbxZY?usp=sharing), where you can find used data sets, trained word2vecs, and trained models.



