# sentence-similarity

## Training Data
 - ./data/doc_frequencies.tsv
 
    wget https://raw.githubusercontent.com/nlptown/nlp-notebooks/master/data/sentence_similarity/doc_frequencies.tsv 
 - ./data/frequencies.tsv

    wget  https://raw.githubusercontent.com/nlptown/nlp-notebooks/master/data/sentence_similarity/frequencies.tsv

 - ./model/GoogleNews-vectors-negative300.bin

    wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    gunzip GoogleNews-vectors-negative300.bin.

 - ./model/glove.840B.300d.txt
    wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip

## How to run

 1. sentence-similarity.py

    python sentence-similarity.py

 2. Google_Sentence_Encode.ipynb
 
    run in Google Jupyter Notebook

## Performance

Pearson Correlation:
![](http://ww1.sinaimg.cn/large/e9a223b5ly1g1swpqj9d0j20ps0h5t8z.jpg)
Spearman Correlation:
![](http://ww1.sinaimg.cn/large/e9a223b5ly1g1swpqj9d0j20ps0h5t8z.jpg)

## References

More details about this project can be found in [this blog](https://jijeng.github.io/2019/04/06/The-evaluation-of-sentence-similarity/)
and [this repository](https://github.com/nlptown/nlp-notebooks)
