import sys
from lxml import etree

def remove_text_elements(input_file, output_file):
    # Parse the SVG file
    tree = etree.parse(input_file)
    root = tree.getroot()

    # Find and remove all <text> elements
    for text_element in root.xpath('.//*[local-name()="text"]'):
        parent = text_element.getparent()
        parent.remove(text_element)

    # Write the modified SVG to the output file
    tree.write(output_file, pretty_print=True, xml_declaration=True, encoding='UTF-8')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python remove_svg_text.py input.svg output.svg')
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_text_elements(input_file, output_file)
