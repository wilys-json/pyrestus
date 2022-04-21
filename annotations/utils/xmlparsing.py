from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List

__all__ = [
    'get_tag',
    'get_frame_size'
]

def get_tag(xml_file: Path, tag: str, how: str = 'all'):
    """
    Parse XML tags.
    """
    assert how in ['once', 'all'], \
        f"Tag Extraction can only be `once` or `all`. Got {how}."

    tree = ET.parse(xml_file)
    root = tree.getroot()

    if how == 'once':
        return root.find(tag)

    return root.findall(tag)


def get_frame_size(xml_file: List[Path],
                   image_tag: str = './image',
                   width_tag: str = 'width',
                   height_tag: str = 'height',
                   overlay: bool = True):
    """
    Helper function to get frame size.
    """

    frame_size = []

    for file in xml_file:
        img_tag = get_tag(file, image_tag, how='once')
        frame_size += [(tuple(map(int, (img_tag.attrib[width_tag],
                                        img_tag.attrib[height_tag]))))]

    if overlay:
        assert len(set(frame_size)) == 1,\
            "Frame Size for all image must be the same for overlaying."
        return frame_size[0]

    return frame_size
