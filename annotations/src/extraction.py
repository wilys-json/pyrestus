import sys
sys.path.insert(0, '..')
import cv2
import xml
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Tuple, Union, List
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from itertools import product
from warnings import warn
from ..utils import get_tag, get_frame_size

# Global Parameters
VIDEO_FORMATS = {'avi', 'mp4'}
IMAGE_FORMATS = {'png', 'jpg'}

@dataclass
class BoundingBox:
    """
    Helper class to encapsulate bounding box parameters.
    """
    color:Tuple[Tuple[int]]=((255,0,255), (255,255,0))
    radius:int=2
    thickness:int=1
    shift=2
    box_size=127

    def __post_init__(self):
        self.factor = (1 << self.shift)


@dataclass
class IOParameters:
    """
    Helper class to encapsulate i/o parameters.
    """
    xml_file:List[Path]
    frame_size:Tuple[int]
    image_file_dir:str=''
    output_dir:str=''
    output_format:Tuple[str]=('avi', 'png')
    output_suffix:str='annotated'
    codec:str='MJPG'
    fps:int=15

    def __post_init__(self):

        self.overlay = (len(self.xml_file) > 1)
        self.xml_file = [Path(str(file)) for file in self.xml_file]
        self.output_writers = []

        output_dir = self.xml_file[0].parent / self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize image and video writing functions/class
        for output in self.output_format:

            # Image writing function
            if output in IMAGE_FORMATS:
                self.output_writers += [
                    lambda name, image : (
                        cv2.imwrite(str(output_dir/f'{name}-{self.output_suffix}.{output}'), image))
                ]

            # Video writer
            elif output in VIDEO_FORMATS:

                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.output_writers += [
                        cv2.VideoWriter(str(output_dir/f'{self.output_suffix}.{output}'),
                                        fourcc,
                                        self.fps,
                                        self.frame_size)
                ]


@dataclass
class XMLParameters:
    """
    Helper class to encapsulate xml parsing parameters.
    """
    image_tag:str='image'
    image_attrib:str='name'
    point_tag:str='points'
    point_attrib:str='points'
    match_attrib:str='name'
    _dir_begin_at:int=0

    def __post_init__(self):
        self.image_tag = './' + self.image_tag
        self.point_tag = './' + self.point_tag
        self.get_image_attrib = lambda tag : Path(*Path(tag.attrib[self.image_attrib]).parts[self._dir_begin_at:])
        self.get_point_attrib = lambda tag : tag.attrib[self.point_attrib]


class AnnotationManager:


    def __init__(self):
        pass

    @staticmethod
    def get_annotation_points(xml_file:Path,
                              xmlparams:XMLParameters,
                              normalized:bool=False,
                              include_img_index=False):

        points = []
        frame_size = get_frame_size([xml_file])

        for tag in get_tag(xml_file, xmlparams.image_tag):

            # Find points
            point_tag =  tag.findall(xmlparams.point_tag)

            if point_tag:

                coordinate = tuple(map(float, xmlparams.get_point_attrib(point_tag[0]).split(',')))

                if normalized:
                    coordinate = tuple(map(lambda x, y : (x/y), coordinate, frame_size))

                if include_img_index:

                    img_attrib = str(xmlparams.get_image_attrib(tag))
                    coordinate = (img_attrib,) + coordinate

                points += [coordinate]

        if include_img_index:

            output_df = pd.DataFrame(points, columns=['image_path', 'x', 'y']).set_index('image_path')
            return output_df

        return np.array(points)

    @staticmethod
    def get_sequence(ioparams:IOParameters,
                     xmlparams:XMLParameters,
                     start:Union[int, None]=None,
                     end:Union[int, None]=None,
                     step:Union[int, None]=None):


        for file in ioparams.xml_file:

            sequences = [list(get_tag(file, xmlparams.image_tag))[start:end:step] for file in ioparams.xml_file]

        if not ioparams.overlay:
            return sequences

        match_sequences = deepcopy(sequences)

        # Get sequence +/- attributes
        if xmlparams.match_attrib:
            match_sequences = [[tag.attrib[xmlparams.match_attrib] for tag in sequence] for sequence in sequences]

        # Sanity Check : same image paths
        seq_cache = []
        for i, sequence in enumerate(match_sequences):
            if i > 0 :
                assert seq_cache == sequence, \
                "Input sequence contains different image"
            seq_cache = sequence

        return sequences


@dataclass
class AnnotationRenderer(AnnotationManager):

    ioparams:IOParameters
    xmlparams:XMLParameters
    bounding_box:BoundingBox
    start:Union[int, None]=None
    end:Union[int, None]=None
    step:Union[int, None]=None

    def __post_init__(self):

        self.writers = self.ioparams.output_writers

    @staticmethod
    def draw(img:np.ndarray,
            points_tag,
            xmlparams:XMLParameters,
            bounding_box:BoundingBox,
            index:int):

        """
        Draw Annotations in `img`.
        """

        # Center point
        x, y = tuple(map(float, xmlparams.get_point_attrib(points_tag[0]).split(',')))
        x = int(x * bounding_box.factor)
        y = int(y * bounding_box.factor)

        # Unpack parameters
        dist = bounding_box.box_size // 2
        radius = bounding_box.radius * bounding_box.factor
        color = bounding_box.color[index]
        thickness = bounding_box.thickness
        shift = bounding_box.shift

        # Draw circle and bounding box
        pt1, pt2 = (x-dist, y-dist), (x+dist, y+dist)
        cv2.circle(img, (x, y), radius, color, thickness, shift=shift)
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness, shift=shift)

        return img

    def run(self):

        """
        Render Annotations in `ioparams.output_format`.
        """

        # Check sequence
        sequences = self.get_sequence(self.ioparams, self.xmlparams, self.start, self.end, self.step)

        # Create empty image list
        image_files = [str(Path(self.ioparams.image_file_dir) / self.xmlparams.get_image_attrib(image))
                       for image in sequences[0]]

        # Iterate over images
        for i, image_file in enumerate(tqdm(image_files)):

            # Iterate over annotations
            for j, sequence in enumerate(sequences):

                assert image_file == (str(Path(self.ioparams.image_file_dir) / self.xmlparams.get_image_attrib(sequence[i])))

                if j == 0:
                    assert Path(image_file).exists()
                    img = cv2.imread(image_file).astype(np.float32)

                # Look for points in i-th element in j-th sequence
                points = sequence[i].findall(self.xmlparams.point_tag)

                if points:
                    img = self.draw(img, points, self.xmlparams, self.bounding_box, j)
                    img = img.astype(np.uint8)

            for writer in self.writers:
                try:
                    writer.write(img)
                except AttributeError:
                    writer(f"{str(Path(image_file).parent / Path(image_file).stem).replace('/', '-')}-{str(i).zfill(5)}", img)

        for writer in self.writers:
            try:
                writer.release()
            except AttributeError:
                pass
