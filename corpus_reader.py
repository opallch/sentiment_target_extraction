import os
from xml.dom import minidom

import numpy as np
import pandas as pd


class CorpusReader:

    SENTEXPR = {"sentiment-neg", "sentiment-pos"}

    def __init__(self, anno_dir_path, doc_dir_path):
        self.annotations = self._align_files(anno_dir_path, doc_dir_path)
        self.sent_df = None
        self.sentexpr_df = None
        self.target_df = None
        self.items = self.create_items_for_corpus()

    @staticmethod
    def _align_files(xml_dir_path, text_dir_path):
        """Align matching xml and text files of mpqa corpus.

        It is assumed that the xml directory structure places each xml file in
        an "ID directory" that has the same name as the according text document
        in the text document directory.

        Args:
            xml_dir_path (str): path to directory containing annotation xml's
            text_dir_path (str): path to directory containing corpus documents

        Returns:
            zip object: zipped file lists with aligned files

        Raises:
            AssertionError: if files cannot be aligned due to varying file
                numbers in directories
        """
        # get annotation files
        xml_files = [
            os.path.join(root, file) for root, _, files in os.walk(xml_dir_path)
            for file in files if file.endswith(".xml")
        ]
        doc_ids = set([file.split(os.sep)[-2] for file in xml_files])
        text_files = [
            os.path.join(root, file) for root, _, files in os.walk(text_dir_path)
            for file in files if file.split(os.sep)[-1] in doc_ids
        ]
        assert len(text_files) == len(xml_files), "directories contain varying file numbers"
        xml_files.sort(key=lambda x: x.split(os.sep)[-2])
        text_files.sort(key=lambda x: x.split(os.sep)[-1])
        return zip(xml_files, text_files)

    def create_items_for_corpus(self):
        """Collect items over all files of the corpus.

        Returns:
            pd.DataFrame: columns=[sentence, start index of sentiment
                expression, end index of sentiment expression, start index of
                target phrase, end index of target phrase, expressed sentiment,
                intensity of sentiment]
        """
        items = pd.DataFrame(columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd",
                                      "sentiment",
                                      "intensity"])
        for xml_file, text_file in self.annotations:
            try:
                items = pd.concat(
                    [items, self.create_items_for_file(xml_file, text_file)],
                    ignore_index=True
                )
            except AttributeError:
                print(f"File {xml_file} skipped due to issue with annotation")
        items.dropna(inplace=True)
        items = items.astype({"sentexprStart": "int",
                              "sentexprEnd": "int",
                              "targetStart": "int",
                              "targetEnd": "int"})
        return items

    def create_items_for_file(self, xml_path, text_path):
        """
        converts xml to tsv
        returns None

        Example of an annotation in xml:
        <Annotation Id="423" Type="targetFrame" StartNode="1012" EndNode="1012">
        <Feature>
        <Name className="java.lang.String">sTarget-link</Name>
        <Value className="java.lang.String">none</Value>
        </Feature>
        <Feature>
        <Name className="java.lang.String">id</Name>
        <Value className="java.lang.String">tfose11</Value>
        </Feature>
        <Feature>
        <Name className="java.lang.String">newETarget-link</Name>
        <Value className="java.lang.String">none</Value>
        </Feature>
        </Annotation>

        Example output:
        98	1012,1012	targetFrame	id="tfose11" sTarget-link="none" newETarget-link="none"  

        Important functions to read from xml:
        (1) get all elements by tag name: getElementsByTagName(<tag name>)
        (2) get attribute value of an element: <elem aka node>.attributes[<attribute name>].value
        (3) get text in a tag (btw BOT and EOT): <elem aka node>.firstChild.data
        """
        xml_file = minidom.parse(xml_path)  # throws FileNotFoundError
        #output_file = basename(xml_path).replace(".xml", ".tsv")
        anns = xml_file.getElementsByTagName('Annotation')
        i = 1  # for annotation index
        ann_list = [[i[1] for i in ann.attributes.items()] for ann in anns]
        df = pd.DataFrame(ann_list, columns=["Id", "Type", "StartNode", "EndNode"])
        df[["StartNode", "EndNode"]] = df[["StartNode", "EndNode"]].astype(int)
        # add text
        with open(text_path, encoding="utf-8") as txt_file:
            text = txt_file.read()
            df.loc[:,"Text"] = df.apply(
                lambda x: text[int(x["StartNode"]):int(x["EndNode"])], axis=1
            )
        # split df
        sent_df = self.get_sent_df(df)
        sentexpr_df =  self.get_sentexpr_df(df, anns)
        target_df = self.get_target_df(df, anns)
        # create item df
        sentexpr_df = sentexpr_df.apply(
            lambda x: self._add_linked_information(x, sent_df, target_df),
            axis=1,
            result_type="expand"
        )
        if not sentexpr_df.empty:
            sentexpr_df.columns = ["sentence", "sentexprStart", "sentexprEnd",
                                   "targetStart", "targetEnd", "sentiment",
                                   "intensity"]
            return sentexpr_df
        return pd.DataFrame(columns=["sentence", "sentexprStart", "sentexprEnd",
                                     "targetStart", "targetEnd", "sentiment",
                                     "intensity"])

    def _add_linked_information(self, attitude, sent_df, target_df):
        """Collect the linked information for each sentiment expression.

        This linked information is the information that is added to the items.

        Args:
            attitude (pd.Series): row from sentexpr_df
            sent_df (pd.DataFrame): DataFrame containing sentence objects of
                document
            target_df (pd.DataFrame): DataFrame containing target objects of
                document
        """
        sent = sent_df[(sent_df["StartNode"] <= attitude["StartNode"]) & \
                       (sent_df["EndNode"] >= attitude["EndNode"])].iloc[0]
        target = target_df.loc[attitude["TargetLink"]]
        # add item tuple to items
        target_start = target["StartNode"] - sent["StartNode"]
        target_end = target["EndNode"] - sent["StartNode"]
        if isinstance(target_start, pd.Series):
            target_start = target_start[0]
            target_end = target_end[0]
        if target_start < 0:
            return [np.nan]*7
        return (
            sent["Text"],
            attitude["StartNode"] - sent["StartNode"],
            attitude["EndNode"] - sent["StartNode"],
            target_start,
            target_end,
            attitude["Sentiment"],
            attitude["Intensity"]
        )

    @staticmethod
    def get_sent_df(df):
        """Create a sub dataframe for sentences."""
        return df[df["Type"] == "sentence"]

    def get_sentexpr_df(self, df, anns):
        """Create a sub dataframe for sentiment expressions."""
        # create targetframe df to collect target links
        targetframe_df = self.get_targetframe_df(df, anns)
        # slice df and annotations to get senti expressions
        sentexpr_df, ann_features = self._get_annos_per_type(
            df, anns, df["Type"] == "attitude"
        )
        idx = []
        intensity, sentiment, link = [], [], []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # add features to df - maybe this can be done with apply - i'll check
            if feature_dict.get("attitude-type") in self.SENTEXPR:
                idx.append(True)
                link.append(feature_dict["targetFrame-link"])
                intensity.append(feature_dict["intensity"])
                sentiment.append(feature_dict["attitude-type"].split("-")[-1])
            elif "attitude-type" in feature_dict:
                idx.append(False)
        # add to df
        sentexpr_df = sentexpr_df[idx]
        sentexpr_df.loc[:,"Sentiment"] = sentiment
        sentexpr_df.loc[:,"Intensity"] = intensity
        # add target links
        links = [targetframe_df.loc[i]["TargetLink"]
                 if i in targetframe_df.index else None for i in link]
        sentexpr_df.loc[:,"TargetLink"] = links
        return sentexpr_df[[i is not None for i in sentexpr_df["TargetLink"]]]

    def get_target_df(self, df, anns):
        target_df, ann_features = self._get_annos_per_type(
            df, anns, df["Type"].str.endswith("Target")
        )
        idx = []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # collect idxs
            idx.append(feature_dict["id"])
        target_df.loc[:,"idx"] = idx
        target_df.set_index("idx", inplace=True)
        return target_df

    def get_targetframe_df(self, df, anns):
        targetframe_df, ann_features = self._get_annos_per_type(
            df, anns, df["Type"] == "targetFrame"
        )
        link, idx = [], []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # collect links
            if feature_dict.get("sTarget-link") not in {"none", None}:
                link.append(feature_dict["sTarget-link"].split(",")[-1])
            elif feature_dict.get("newETarget-link") not in {"none", None}:
                link.append(feature_dict["newETarget-link"].split(",")[-1])
            elif feature_dict.get("eTarget-link") not in {"none", None}:
                link.append(feature_dict["eTarget-link"].split(",")[-1])
            else:
                link.append("none")
            idx.append(feature_dict["id"])
        targetframe_df.loc[:,"TargetLink"] = link
        targetframe_df.loc[:,"idx"] = idx
        targetframe_df = targetframe_df[targetframe_df["TargetLink"] != "none"]
        targetframe_df.set_index("idx", inplace=True)
        return targetframe_df

    @staticmethod
    def _get_feature_dict(feature_obj):
        # collect features
        feature_dict = {}
        for feature in feature_obj:
            name = feature.getElementsByTagName('Name')[0].firstChild.data
            value = feature.getElementsByTagName('Value')[0].firstChild.data
            feature_dict[name] = value
        return feature_dict

    @staticmethod
    def _get_annos_per_type(df, anns, condition):
        label_df = df[condition].copy()
        anns = [anns[i] for i in np.where(condition)[0]]
        ann_features = [ann.getElementsByTagName('Feature') for ann in anns]
        return label_df, ann_features


if __name__ == "__main__":
    anno_dir_path = "mpqa_corpus/gate_anns"
    text_dir_path = "mpqa_corpus/docs"
    corpus_reader = CorpusReader(anno_dir_path, text_dir_path)
    print(corpus_reader.items)
    i = corpus_reader.items.iloc[1110]
    print('.......................................................')
    print(i.sentence)
    print()
    print(i.sentence[i.sentexprStart:i.sentexprEnd])
    print(i.sentence[i.targetStart:i.targetEnd])
