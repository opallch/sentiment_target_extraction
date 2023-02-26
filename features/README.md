# Feature Engineering
Currently, some features from the constituency parse and the dependency parse are implemented. Categorical features are implemented with One-Hot-Encoding.

## Constituency Features
1. Tree label of the target <!--POS Tag right?-->
2. Label of the lowest common ancestor of the target and the sentiment expression <!--POS Tag right?-->

## Dependency Features
All the dependency features are based on the head of the sentiment expression and that of the target, including:
1. Relation between the two heads
2. POS-tag of the sentiment head
3. POS-tag of the target head
4. Distance between the two heads
SpaCy is used for the dependency features extraction. 