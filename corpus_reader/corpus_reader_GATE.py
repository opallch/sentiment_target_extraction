# -*- coding: utf-8 -*-
"""
Reader class for the MPQA corpus extracting sentiment expressions as well as 
their corresponding targets and the sentences they are in.
"""
import os
from xml.dom import minidom

import numpy as np
import pandas as pd

class GATECorpusReader:

    SENTEXPR = {"sentiment-neg", "sentiment-pos"}

    def __init__(self, anno_dir_path, doc_dir_path):
        """Constructor of the CorpusReader class.

        Args:
            anno_dir_path (str): path of the directory containing the xml
                annotation files created with GATE
            doc_dir_path (str): path of the directory containing the according
                text documents

        Attributes:
            self.annotations: zip object with aligned annotation and text files
            self.items: pd.DataFrame with sentiment items
        """
        self.annotations = self._align_files(anno_dir_path, doc_dir_path)
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
        # get document files
        doc_ids = set([file.split(os.sep)[-2] for file in xml_files])
        text_files = [
            os.path.join(root, file) for root, _, files in os.walk(text_dir_path)
            for file in files if file.split(os.sep)[-1] in doc_ids
        ]
        assert len(text_files) == len(xml_files), "directories contain varying file numbers"
        # align by sorting
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
        # create empty data frame with colums for all item elements
        items = pd.DataFrame(columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd"])
        for xml_file, text_file in self.annotations:
            try:
                # concatenate data frame so far with new one for current file
                items = pd.concat(
                    [items, self.create_items_for_file(xml_file, text_file)],
                     ignore_index=True
                )
            except AttributeError:
                print(f"File {xml_file} skipped due to issue with annotation")
        # clean up final data frame by dropping np.nan
        items.dropna(inplace=True)
        # ...transforming index columns to integer
        items = items.astype({"sentexprStart": "int",
                              "sentexprEnd": "int",
                              "targetStart": "int",
                              "targetEnd": "int"})
        # ...and reindexing
        return items.reset_index()

    def create_items_for_file(self, xml_path, text_path):
        """Create a data frame with items for a single file.

        Returns:
            pd.DataFrame: columns=[sentence, start index of sentiment
                expression, end index of sentiment expression, start index of
                target phrase, end index of target phrase, expressed sentiment,
                intensity of sentiment] for a single file
        """
        # read annotations
        xml_file = minidom.parse(xml_path)
        anns = xml_file.getElementsByTagName('Annotation')
        ann_list = [[i[1] for i in ann.attributes.items()] for ann in anns]
        # create data frame with all annotations
        df = pd.DataFrame(ann_list, columns=["Id", "Type", "StartNode", "EndNode"])
        df[["StartNode", "EndNode"]] = df[["StartNode", "EndNode"]].astype(int)
        # add text for each annotation
        with open(text_path, encoding="utf-8") as txt_file:
            text = txt_file.read()
            df.loc[:,"Text"] = df.apply(
                lambda x: text[int(x["StartNode"]):int(x["EndNode"])], axis=1
            )
        # split df
        sent_df = self._get_sent_df(df)
        sentexpr_df = self._get_sentexpr_df(df, anns)
        target_df = self._get_target_df(df, anns)
        # create item df
        sentexpr_df = sentexpr_df.apply(
            lambda x: self._add_linked_information(x, sent_df, target_df),
            axis=1,
            result_type="expand"
        )
        # return empty df with right columns to avoid concatenation problems
        if not sentexpr_df.empty:
            sentexpr_df.columns = ["sentence", "sentexprStart", "sentexprEnd",
                                   "targetStart", "targetEnd"]
            return sentexpr_df
        return pd.DataFrame(columns=["sentence", "sentexprStart", "sentexprEnd",
                                     "targetStart", "targetEnd"])

    def _add_linked_information(self, attitude, sent_df, target_df):
        """Collect the linked information for each sentiment expression.

        This linked information is the information that is added to the items.

        Args:
            attitude (pd.Series): row from sentexpr_df
            sent_df (pd.DataFrame): DataFrame containing sentence objects of
                document
            target_df (pd.DataFrame): DataFrame containing target objects of
                document

        Returns:
            list: [sentence as text, start index of sentiment expression, end
                index of sentiment expression, start index of target phrase,
                end index of target phrase, expressed sentiment, intensity of
                sentiment]; or: list containing np.nan for all of these if
                target is not in same sentence
        """
        # find right sentence
        sent = sent_df[(sent_df["StartNode"] <= attitude["StartNode"]) & \
                       (sent_df["EndNode"] >= attitude["EndNode"])].iloc[0]
        # get target
        target = target_df.loc[attitude["TargetLink"]]
        target_start = target["StartNode"] - sent["StartNode"]
        target_end = target["EndNode"] - sent["StartNode"]
        # filter and make sure that returned instance is of right type
        if isinstance(target_start, pd.Series):
            target_start = target_start[0]
            target_end = target_end[0]
        # target not within sentence
        if (target_start < 0) or (sent["EndNode"] - sent["StartNode"] < target_end):
            return [np.nan]*5
        return [
            sent["Text"],
            attitude["StartNode"] - sent["StartNode"],
            attitude["EndNode"] - sent["StartNode"],
            target_start,
            target_end
        ]

    @staticmethod
    def _get_sent_df(df):
        """Create a subdataframe for sentences."""
        return df[df["Type"] == "sentence"]

    def _get_sentexpr_df(self, df, anns):
        """Create a subdataframe for sentiment expressions with features."""
        # create targetframe df to collect target links
        targetframe_df = self._get_targetframe_df(df, anns)
        # slice df and annotations to get senti expressions
        sentexpr_df, ann_features = self._get_annos_per_type(
            df, anns, df["Type"] == "attitude"
        )
        idx, link = [], []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # add features to df - maybe this can be done with apply - i'll check
            if feature_dict.get("attitude-type") in self.SENTEXPR:
                idx.append(True)
                link.append(feature_dict["targetFrame-link"])
            elif "attitude-type" in feature_dict:
                idx.append(False)
        # add to df
        sentexpr_df = sentexpr_df[idx]
        # add target links
        links = [targetframe_df.loc[i]["TargetLink"]
                 if i in targetframe_df.index else None for i in link]
        sentexpr_df.loc[:,"TargetLink"] = links
        return sentexpr_df[[i is not None for i in sentexpr_df["TargetLink"]]]

    def _get_target_df(self, df, anns):
        """Create a subdataframe for target annotations with features."""
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

    def _get_targetframe_df(self, df, anns):
        """Create a subdataframe for target frames with links."""
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
        """Extract features as a dict from annotation features."""
        # collect features
        feature_dict = {}
        for feature in feature_obj:
            name = feature.getElementsByTagName('Name')[0].firstChild.data
            value = feature.getElementsByTagName('Value')[0].firstChild.data
            feature_dict[name] = value
        return feature_dict

    @staticmethod
    def _get_annos_per_type(df, anns, condition):
        """Get annotations of given type as annotation list and df."""
        label_df = df[condition].copy()
        anns = [anns[i] for i in np.where(condition)[0]]
        ann_features = [ann.getElementsByTagName('Feature') for ann in anns]
        return label_df, ann_features


if __name__ == "__main__":
    anno_dir_path = "mpqa_corpus/gate_anns"
    text_dir_path = "mpqa_corpus/docs"
    corpus_reader = CorpusReader(anno_dir_path, text_dir_path)
    corpus_reader.items.to_pickle("./test_files/items.pkl")

    with open("./test_files/items", "w") as f_out:
        for idx in range(0, len(corpus_reader.items)):
            i = corpus_reader.items.iloc[idx]
            print(idx, file=f_out)
            print(i.sentence, file=f_out)
            f_out.write("{} {} {} {}\nSENTI: {}\nTARGET: {}\n\n".format(
                    i.sentexprStart, 
                    i.sentexprEnd, 
                    i.targetStart, 
                    i.targetEnd,
                    i.sentence[i.sentexprStart:i.sentexprEnd],
                    i.sentence[i.targetStart:i.targetEnd]
                )
            )
