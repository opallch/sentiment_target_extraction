from os.path import basename
from xml.dom import minidom

class CorpusReader():
    
    def __init__(self):
        pass

    def xml2tsv(self, xml_path):
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
        xml_file = minidom.parse(xml_path) # throws FileNotFoundError
        output_file = basename(xml_path).replace(".xml", ".tsv")
        anns = xml_file.getElementsByTagName('Annotation')
        i = 1 # for annotation index
        
        with open(output_file, 'w') as f_out: # better: os.path?
            for ann in anns:
                start_node = ann.attributes['StartNode'].value
                end_node = ann.attributes['EndNode'].value
                type =  ann.attributes['Type'].value
                features = []

                # loop through the features
                for feature in ann.getElementsByTagName('Feature'):                
                    name = feature.getElementsByTagName('Name')[0].firstChild.data
                    value = feature.getElementsByTagName('Value')[0].firstChild.data
                    features.append(f'{name}="{value}"')

                # write into output file
                features_str = '\t'.join(features)
                line = f"{i}\t{start_node},{end_node}\t{type}\t{features_str}\n"
                f_out.write(line)

                i += 1

def main():
    corpus_reader = CorpusReader()
    corpus_reader.xml2tsv("./test_files/test.xml")

if __name__ == "__main__":
    main()