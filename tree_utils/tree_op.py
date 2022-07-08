# doc: spacy.Doc
# no weight on the edge
import spacy
from spacy import displacy

def lowest_common_ancestor(node1, node2, root):
    """returns the index of the lowest ancestor between 2 nodes."""
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
        print(result)

        if len(result) == 2:
            return root 
        
        elif len(result) == 1:
            return result[0]
        
        else:
            return None

def distance_btw_3_pts(node1, node2, ancestor):
    """returns the distance between node 1 and node 2 via their common ancestor."""
    return distance_btw_child_ancestor(node1, ancestor) + \
            distance_btw_child_ancestor(node2, ancestor)

def distance_btw_child_ancestor(child, ancestor):
    """returns the distance between a child and its ancestor."""
    current_child = child
    current_parent = current_child.head
    steps = 1
        
    while current_parent != ancestor:
        step += 1
        current_child = current_parent
        current_parent = current_parent.head
        
    return steps

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm')
    sentence = "The United States has been preparing annual reports on human rights in 190 countries for 25 years while ignoring the real situation at home."
    sent_doc = nlp(sentence)
    #displacy.serve(sent_doc, style='dep')
    root = next(sent_doc.sents).root
    lca = lowest_common_ancestor(sent_doc[17], sent_doc[21], root)
    print(
        distance_btw_3_pts(sent_doc[17], sent_doc[21], lca)
    )