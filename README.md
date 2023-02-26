# SentimentTargetExtraction
This repository contains the implementation and the documentation of the sentiment target extraction project, which is based on [Wiegand et al's paper pulished in 2019](https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/9321/file/Wiegand_etal._A_supervised_learning_approach_2019.pdf). The pipeline looks like the following: data preprocessing (including annotation), training and evaluation. 

## Data, Annotation and Corpus Reader

### Annotation Files with INCEpTION
`corpus_reader_inception.py` processes the annotation files in `json` format from INCEpTION. You may use it in two ways:

1. via command line:

```bash
python3 corpus_reader_prodigy.py -a path/to/annotation/jsonl/files/root/ -o path/to/csv/file
python3 corpus_reader_GATE.py -a path/to/annotation/xml/files/root/ -rt path/to/raw/texts/root/ -o path/to/csv/file
```
2. directly create the Corpus Reader object and use its `items_df_to_csv()` method in the script.

Each line of the output dataframe represents one relation between a sentiment expression and a target/source: 

||rawTextFilename|sentenceID|sentence|sentexprStart|sentexprEnd|targetStart|targetEnd|sourceStart|sourceEnd|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|0|UNSC_2014_SPV.7154_spch004_sentsplit.txt|36|Mr. Churkin (Russian Federation) (spoke in Russian): The Russian Federation has called for this emergency meeting of the Security Council because of the serious dangerous evolution of the situation in south-eastern Ukraine.|153|170|171|180|-1|-1|

-1 in this example indicates that it is not a relation to a source, but to a target; vice versa. `sentenceID` is the id of the corresponding sentence in the annotation file.

### Annotation Files with GATE or Prodigy
Both the [MPQA corpus](https://mpqa.cs.pitt.edu/corpora/mpqa_corpus/) and the [UNSC corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH) are used as the data source for the project. Since the MPQA corpus is an opinion corpus, it already comes with annotations related to sentiment analysis i.e. with the annotations of sentiment expressions and their corresponding targets (in character span), which was done by using [GATE](http://mpqa.cs.pitt.edu/annotation/). Meanwhile, as only raw texts are available for the the UNSC corpus, we have conducted the annotation (also in character span, on sentence level) with [Prodigy](https://prodi.gy/). 

The corpus readers in `corpus_reader/` create a panda DataFrame for the sentence-level annotations and stores it in the `csv` format. The format looks like:

| |sentence|sentexprStart|sentexprEnd|targetStart|targetEnd|
|---|---|---|---|---|---|
|0|"The President: I should like to inform the Council that I have received a letter from the representative of Georgia, in which he requests to be invited to participate in the discussion of the item on the Council's agenda."|17|28|32|220|

`corpus_reader_GATE.py` processes the `xml`-files produced by GATE for the MPQA corpus and `corpus_reader_prodigy.py` the `jsonl`-files produced by prodigy for the UNSC corpus. Feel free to use them if your annotations conform to one of these formats. You may use it in two ways:
1. via command line:

```bash
python3 corpus_reader_prodigy.py -a path/to/annotation/jsonl/files/root/ -o path/to/csv/file
python3 corpus_reader_GATE.py -a path/to/annotation/xml/files/root/ -rt path/to/raw/texts/root/ -o path/to/csv/file
```
2. directly create the Corpus Reader object and use its `items_df_to_csv()` method in the script.


## Feature Engineering
Please refer to the README in `features/`.

## Training
`classifier/`, `creating_training_data.py`,  `model.py`
<!-- should we relocate tree_utils? -->

## Evaluation
`eval.py`
<!-- majority baseline? -->