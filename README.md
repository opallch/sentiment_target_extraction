# SentimentTargetExtraction
This repository contains the implementation and the documentation of the sentiment target extraction project, which is based on [Wiegand et al's paper pulished in 2019](https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/9321/file/Wiegand_etal._A_supervised_learning_approach_2019.pdf). The pipeline looks like the following: data preprocessing (including annotation), training and evaluation. 

## Data, Annotation and Corpus Reader
Both the [MPQA corpus](https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/) and the [UNSC corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH) are used as the data source for the project. Since the MPQA corpus is an opinion corpus, it already comes with annotations related to sentiment analysis i.e. with the annotations of sentiment expressions and their corresponding targets (in span), which was done by using [GATE](http://mpqa.cs.pitt.edu/annotation/). Meanwhile, as only raw texts are available for the the UNSC corpus, we have conducted the annotation (also in span, on sentence level) with [Prodigy](https://prodi.gy/). 

The corpus readers in `corpus_reader/` create a panda DataFrame for the sentence-level annotations and stores it in the `csv` format. The format looks like:
||sentence|sentexprStart|sentexprEnd|targetStart|targetEnd|
|--|-----|-----|-----|-----|-----|
|0|"The President: I should like to inform the Council that I have received a letter from the representative of Georgia, in which he requests to be invited to participate in the discussion of the item on the Council's agenda."|17|28|32|220|

`corpus_reader_GATE.py` processes the `xml`-files produced by GATE for the MPQA corpus and `corpus_reader_prodigy.py` the `jsonl`-files produced by prodigy for the UNSC corpus. Feel free to use them if your annotations conform to one of these formats. The following shows their usage:

```bash
python3 corpus_reader_prodigy.py -a path/to/annotation/jsonl/files/root/ -o path/to/csv/file
python3 corpus_reader_GATE.py -a path/to/annotation/xml/files/root/ -rt path/to/raw/texts/root/ -o path/to/csv/file
```

## Training


## Evaluation

[//]: # (should we relocate tree_utils?)