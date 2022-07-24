# doc: spacy.Doc
# no weight on the edge
import spacy
import pandas as pd
from spacy import displacy

def lowest_common_ancestor(node1, node2, root):
    """returns the lowest ancestor between 2 nodes."""
    if root is None:
        return None
    
    if node1 == root or node2 == root:
        return root

    if list(root.children) is None or len(list(root.children)) == 0:
        return None

    else:
        result = []
        for child in list(root.children):
            result.append(
                lowest_common_ancestor(node1, node2, child)
                )
        
        result = [elem for elem in result if elem is not None]
        #print(result)

        if len(result) == 2:
            return root 
        
        elif len(result) == 1:
            return result[0]
        
        else:
            return None

def distance_btw_3_pts(node1, node2, ancestor):
    """
    It returns the distance between node 1 and node 2 via their common ancestor.
    If ancestor is None, it indicates that node1 and node2 are in two different sentences
    which, we assume, are connected together with a parent node.
    For calculating the distance in this case, please refer to `_distance_btw_child_ancestor()` 
    and the `README` in `/features/`.
    """
    return  _distance_btw_child_ancestor(node1, ancestor) + \
            _distance_btw_child_ancestor(node2, ancestor)
            

def _distance_btw_child_ancestor(child, ancestor):
    """
    A help function of `distance_btw_3_pts()` which returns the distance between a child and its ancestor.
    If ancestor is None, the distance is the distance between the child and its root +1.
    The reason for adding one is that we assume the corresponding sentence is connected to an external node
    which binds another sentence.
    For more information, please refer to `_distance_btw_child_ancestor()` 
    and the `README` in `/features/`.
    """
    try:
        if child == ancestor:
            return 0
        
        else:
            current_child = child
            current_parent = current_child.head
            steps = 1

            if ancestor is None: # go along way to the root + 1
                while current_child.dep_ != 'ROOT':
                    steps += 1
                    current_child = current_parent
                    current_parent = current_parent.head
                steps += 1
                    
            else:
                while current_parent != ancestor:
                    steps += 1
                    current_child = current_parent
                    current_parent = current_parent.head
                
            return steps
    
    except AttributeError:
        return -1

if __name__ == "__main__":
    items_df = pd.read_pickle("../test_files/items.pkl")
    item = items_df.iloc[1100]
    nlp = spacy.load('en_core_web_sm')
    sentence = item.sentence
    sent_doc = nlp(sentence)
    displacy.serve(sent_doc, style='dep')
    root = next(sent_doc.sents).root
    lca = lowest_common_ancestor(sent_doc[9], sent_doc[15], root)
    print(lca) 
    print(
        distance_btw_3_pts(sent_doc[9], sent_doc[15], lca)
    )