import os
from xml.dom import minidom

import numpy as np
import pandas as pd


class CorpusReader():
    
    SENTEXPR = {"sentiment-neg", "sentiment-pos"}
    
    def __init__(self):
        pass

    def xml2tsv(self, xml_path, txt_path):
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
        # add text
        with open(txt_path, encoding="utf-8") as txt_file:
            text = txt_file.read()
            df.loc[:,"Text"] = df.apply(
                lambda x: text[int(x["StartNode"]):int(x["EndNode"])], axis=1
            )
            df["Text"].str.replace("\n", "")
        # split df
        sent_df = self.get_sent_df(df)
        sentexpr_df =  self.get_sentexpr_df(df, anns)
        target_df = self.get_target_df(df, anns)
        targetframe_df = self.get_targetframe_df(df, anns)

        # well i guess we shall return a tuple collection here.
        # sentence link still missing - i shall do that before monday
        return

    @staticmethod
    def get_sent_df(df):
        """Create a sub dataframe for sentences."""
        return df[df["Type"] == "sentence"]

    def get_sentexpr_df(self, df, anns):
        """Create a sub dataframe for sentiment expressions."""
        # create targetframe df to collect target links
        targetframe_df = self.get_targetframe_df(df, anns)
        # slice df and annotations to get senti expressions
        sentexpr_df = df[df["Type"] == "attitude"]
        anns = [anns[i] for i in np.where(df["Type"] == "attitude")[0]]
        # get features
        ann_features = [ann.getElementsByTagName('Feature') for ann in anns]
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
        target_df = df[df["Type"].str.endswith("Target")].copy()
        anns = [anns[i] for i in np.where(df["Type"].str.endswith("Target"))[0]]
        ann_features = [ann.getElementsByTagName('Feature') for ann in anns]
        idx = []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # collect idxs
            idx.append(feature_dict["id"])
        target_df.loc[:,"idx"] = idx
        target_df.set_index("idx", inplace=True)
        return target_df

    def get_targetframe_df(self, df, anns):
        targetframe_df = df[df["Type"] == "targetFrame"].copy()
        anns = [anns[i] for i in np.where(df["Type"] == "targetFrame")[0]]
        ann_features = [ann.getElementsByTagName('Feature') for ann in anns]
        link, idx = [], []
        for feature_obj in ann_features:
            feature_dict = self._get_feature_dict(feature_obj)
            # collect links
            if feature_dict["sTarget-link"] != "none":
                link.append(feature_dict["sTarget-link"])
            else:
                link.append(feature_dict["newETarget-link"].split(",")[0])
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


if __name__ == "__main__":
    corpus_reader = CorpusReader()
    corpus_reader.xml2tsv("mpqa_corpus/gate_anns/20011125/21.01.04-6923/gateman.mpqa.3.0.xml",
                          "mpqa_corpus/docs/20011125/21.01.04-6923")
