# Feature Engineering
Currently, some features from the constituency parse and the dependency parse are implemented.

## Constituency Features


## Dependency Features
All the dependency features are based on the head of the sentiment expression and that of the target, including:
1. Relation between the two heads
2. POS-tag of the sentiment head
3. POS-tag of the target head
4. Distance between the two heads
SpaCy is used for the feature extraction. 1-3 are implemented with One-Hot-Encoding due to their categorical nature.