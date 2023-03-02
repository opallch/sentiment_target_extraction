# -*- coding: utf-8 -*-
"""
Script for creating training data
"""
import pandas as pd
from benepar import Parser
from tqdm import tqdm

from features.constituency_features import ConstituencyParseFeatures
from features.dependency_features import DependencyParseFeatures
from features.feature_utils import char_span_to_token_span, parse_sent, get_candidates

# cancel an annoying UserWarning
import warnings
warnings.filterwarnings("ignore")


def create_training_data(df, filename=""):
    """Create training instances from the data."""
    parser = Parser("benepar_en3")
    cpf = ConstituencyParseFeatures()
    dpf = DependencyParseFeatures()
    X, y = [], []
    for row_tuple in tqdm(df.iterrows()):
        row = row_tuple[1].copy()
        for k, v in char_span_to_token_span(row).items():
            row[k] = v
        tree = parse_sent(row["sentence"], parser)
        tree.add_spans()
        candidates = get_candidates(row["sentence"], tree)
        for c in candidates:
            instance = row.copy()
            if (c.span_start() == row["targetStart"]) and (c.span_end() == row["targetEnd"]):
                y.append(1)
            else:
                y.append(0)
            instance["targetStart"] = c.span_start()
            instance["targetEnd"] = c.span_end()
            c_f = cpf.get_features(instance, tree)
            d_f = dpf.get_features(row_tuple[1])
            X.append(c_f + d_f)
    out_df = pd.DataFrame(X)
    out_df["gold"] = y
    if filename:
        out_df.to_csv(filename, encoding="utf-8")
    return out_df


if __name__ == "__main__":
    gold_examples = pd.read_csv("test_files/test_doc.csv", sep="\t", encoding="utf-8")
    out_df = create_training_data(gold_examples, "test_files/tr_data_cd.csv")
    print(out_df)
