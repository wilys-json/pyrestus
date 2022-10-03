#################################################################################
# MIT License                                                                   #
#                                                                               #
# Copyright (c) 2021 Wilson Lam                                                 #
#                                                                               #
# Permission is hereby granted, free of charge, to any person obtaining a copy  #
# of this software and associated documentation files (the "Software"), to deal`#
# in the Software without restriction, including without limitation the rights  #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     #
# copies of the Software, and to permit persons to whom the Software is         #
# furnished to do so, subject to the following conditions:                      #
#                                                                               #
# The above copyright notice and this permission notice shall be included in all#
# copies or substantial portions of the Software.                               #
#                                                                               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
# SOFTWARE.                                                                     #
#################################################################################
from abc import ABC, abstractmethod
from ..utils import get_tag, get_frame_size
from ...usv import read_DICOM_dir
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Tuple, Union, List
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import xml
import cv2

# Global Parameters
VIDEO_FORMATS = {'avi', 'mp4'}
IMAGE_FORMATS = {'png', 'jpg'}


def retrieve_template(data_dir: Union[Path, str],
                      img_dir: Union[Path, str],
                      img_frame: str,
                      x: Union[float, str],
                      y: Union[float, str],
                      img_format:str ='.png',
                      **kwargs) -> np.ndarray:

    """
    Return an image template for SiamFC training.
    Image will be of center point (x,y).
    """

    img_pth = Path(str(data_dir)) / img_dir / f'{img_frame}{img_format}'
    y1, x1, y2, x2 = cropping_dim(x,y,**kwargs)

    img = cv2.imread(str(img_pth))

    if img is not None:
        return cv2.imread(str(img_pth))[x1:x2, y1:y2, :]

    return None


def cropping_dim(x: Union[str, float],
                 y: Union[str, float],
                 bbox_size: int) -> Tuple[int]:

    """
    Return tuple of (left, top, right, bottom) pixel address
    w.r.t. a defined bounding box.

    Bounding Box defined in **kwargs
    `box_size` = dimension of the bounding box, assumed to be a square
    `shift` = bit shift, pass into OpenCV functions
    """

    bounding_box = BoundingBox(box_size=bbox_size)
    x, y = RendererBase._floats2ints((float(x), float(y)),
                                      bounding_box.factor)
    x1, y1 = map(lambda i : (i // bounding_box.factor) \
                            - (bounding_box.box_size //2), (x,y))

    return (y1, x1, y1 + bounding_box.box_size, x1 + bounding_box.box_size)


class Color:
    """
    Color palette.
    """
    def __class_getitem__(cls, idx: int):
        COLORS = (
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (0, 255, 0)
        )
        idx = idx % len(COLORS)  # Comment out to disable cycle indexing
        return COLORS[idx]


class BoundingBox:
    """
    Helper class to encapsulate bounding box parameters.
    """
    radius: int = 2
    thickness: int = 1
    shift = 2
    box_size = 127

    def __init__(self, radius:int=2,
                       thickness:int=1,
                       shift:int=2,
                       box_size:int=127):

        self.radius=radius
        self.thickness=thickness
        self.shift=shift
        self.box_size=box_size
        self.factor=1 << shift


@dataclass
class IOParameters:
    """
    Helper class to encapsulate i/o parameters.
    """
    xml_file: List[Path] = None
    frame_size: Tuple[int] = None
    image_file_dir: str = ''
    output_dir: str = ''
    output_format: Tuple[str] = ('avi', 'png')
    output_suffix: str = 'annotated'
    codec: str = 'MJPG'
    fps: int = 15

    def __post_init__(self):
        if not self.xml_file:
            self.output_writers = []
            return
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
                    lambda name, image: (
                        cv2.imwrite(str(output_dir / f'{name}-{self.output_suffix}.{output}'), image))
                ]

            # Video writer
            elif output in VIDEO_FORMATS:

                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.output_writers += [
                    cv2.VideoWriter(str(output_dir / f'{self.output_suffix}.{output}'),
                                    fourcc,
                                    self.fps,
                                    self.frame_size)
                ]


@dataclass
class XMLParameters:
    """
    Helper class to encapsulate xml parsing parameters.
    """
    structure_tag: str = 'track'
    structure_attrib: str = 'label'
    target_structure: str = 'HB'
    image_tag: str = 'image'
    image_attrib: str = 'name'
    annotation_tag: str = 'points'
    annotation_attrib: str = 'points'
    match_attrib: str = 'name'
    coordinate_sep: str = ','
    points_sep: str = ';'
    source: str = ''
    find_strcuture: bool = False

    _avi_integrated: bool = False
    _dir_begin_at: int = 0

    def __post_init__(self):
        self.image_tag = './' + self.image_tag
        self.annotation_tag = './' + self.annotation_tag
        self.get_image_attrib = lambda tag: Path(
            *Path(tag.attrib[self.image_attrib]).parts[self._dir_begin_at:])
        self.get_annotation_attrib = lambda tag: tag.attrib[self.annotation_attrib]


class AnnotationManager:

    """
    Helper class to extract parameters of interst from XML tags.
    """

    @staticmethod
    def get_coordinate(xmlparams: XMLParameters,
                       coordinate: Union[ET.Element, str]):
        sep = xmlparams.coordinate_sep
        if isinstance(coordinate, ET.Element):
            coordinate = xmlparams.get_annotation_attrib(coordinate)
        coordinate = coordinate.split(sep)

        return tuple(map(float, coordinate))

    @staticmethod
    def get_coordinates(xmlparams: XMLParameters,
                        coordinates: List[str]):

        sep = xmlparams.points_sep
        coordinates = xmlparams.get_annotation_attrib(coordinates).split(sep)
        return [AnnotationManager.get_coordinate(xmlparams, coord)
                for coord in coordinates]

    @staticmethod
    def get_annotation_points(xml_file: Path,
                              xmlparams: XMLParameters,
                              normalized: bool = False,
                              include_img_index=False):

        points = []
        frame_size = get_frame_size([xml_file])

        for tag in get_tag(xml_file, xmlparams.image_tag):

            # Find points
            point_tag = tag.findall(xmlparams.annotation_tag)

            if point_tag:

                coordinate = AnnotationManager.get_coordinate(xmlparams,
                                                              point_tag[0])

                if normalized:
                    coordinate = tuple(
                        map(lambda x, y: (x / y), coordinate, frame_size))

                if include_img_index:

                    img_attrib = str(xmlparams.get_image_attrib(tag))
                    coordinate = (img_attrib,) + coordinate

                points += [coordinate]

        if include_img_index:

            output_df = pd.DataFrame(
                points, columns=['image_path', 'x', 'y']).set_index('image_path')
            return output_df

        return np.array(points)

    @staticmethod
    def get_sequence(ioparams: IOParameters,
                     xmlparams: XMLParameters,
                     start: Union[int, None] = None,
                     end: Union[int, None] = None,
                     step: Union[int, None] = None):

        """
        Retrieve image file sequence. Applicable for inter-rater reliability.
        """

        for file in ioparams.xml_file:

            sequences = [list(get_tag(file, xmlparams.image_tag))[
                start:end:step] for file in ioparams.xml_file]

        if not ioparams.overlay:
            return sequences

        match_sequences = deepcopy(sequences)

        # Get sequence +/- attributes
        if xmlparams.match_attrib:
            match_sequences = [[tag.attrib[xmlparams.match_attrib]
                                for tag in sequence] for sequence in sequences]

        # Sanity Check : same image paths
        seq_cache = []
        for i, sequence in enumerate(match_sequences):
            if i > 0:
                assert seq_cache == sequence, \
                    "Input sequence contains different image"
            seq_cache = sequence

        return sequences


class RendererInterface(ABC):

    @abstractmethod
    def draw(self,
             img: np.ndarray,
             annotation_tag,
             index: int,
             **kwargs):
        pass


@dataclass
class RendererBase:

    ioparams: IOParameters = None
    xmlparams: XMLParameters = None
    start: Union[int, None] = None
    end: Union[int, None] = None
    step: Union[int, None] = None

    def __post_init__(self):
        if self.ioparams.output_writers:
            self.writers = self.ioparams.output_writers

    @staticmethod
    def _floats2ints(coordinates: Tuple[float], factor: int):
        x, y = coordinates
        return int(x * factor), int(y * factor)


@dataclass
class PointRenderer(AnnotationManager, RendererInterface, RendererBase):

    bounding_box: BoundingBox = BoundingBox()

    def draw(self,
             img: np.ndarray,
             annotation_tag,
             index: int,
             **kwargs):

        # Center point
        if isinstance(annotation_tag, str):
            x, y = self.get_coordinate(self.xmlparams, annotation_tag[0])
        else:
            x, y = annotation_tag  # expects list / numpy array
        x, y = self._floats2ints((x, y), self.bounding_box.factor)

        # Unpack parameters
        dist = self.bounding_box.box_size // 2
        radius = self.bounding_box.radius * self.bounding_box.factor
        thickness = self.bounding_box.thickness
        shift = self.bounding_box.shift
        color = kwargs.pop('color', Color[index])

        # Draw circle and bounding box
        pt1, pt2 = (x - dist, y - dist), (x + dist, y + dist)
        cv2.circle(img, (x, y), radius, color, thickness, shift=shift)
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness, shift=shift)

        return img


@dataclass
class LineRenderer(AnnotationManager, RendererInterface, RendererBase):

    def _get_contour(self, contour_tag: ET.Element):
        contour = self.get_coordinates(self.xmlparams, contour_tag)
        contour = np.array([self._floats2ints(xy, 1) for xy in contour])
        return contour

    def draw(self,
             img: np.ndarray,
             contour_tag: List[ET.Element],
             index: int,
             **kwargs):

        contour = self._get_contour(contour_tag[0])
        color = kwargs.pop('color', Color[index])
        cv2.polylines(img, [contour], False, color=color, **kwargs)

        return img


@dataclass
class ContourRenderer(LineRenderer):

    def draw(self,
             img: np.ndarray,
             contour_tag: List[ET.Element],
             index: int,
             **kwargs):

        contour = self._get_contour(contour_tag[0])
        color = kwargs.pop('color', Color[index])
        cv2.polylines(img, [contour], True, color=color, **kwargs)

        return img


class RendererFactory:

    @classmethod
    def create(cls, option, **kwargs):
        renderers = dict(
            point=PointRenderer,
            line=LineRenderer,
            contour=ContourRenderer
        )
        return renderers[option](**kwargs)

class VideoExtractor:

    @classmethod
    def extract(cls, video_dir, target_dir, video_format='.avi'):
        file_paths = read_DICOM_dir(video_dir)
        file_paths = {path.name : path for path in file_paths
                                        if path.suffix == video_format}

class AnnotationRenderer:

    def __init__(self, option, **kwargs):
        self.renderer = RendererFactory.create(option, **kwargs)

    def _xml_img_rendering(self, **kwargs):
        ioparams = self.renderer.ioparams
        xmlparams = self.renderer.xmlparams
        start = self.renderer.start
        end = self.renderer.end
        step = self.renderer.step
        writers = self.renderer.writers

        # Check sequence
        sequences = self.renderer.get_sequence(
            ioparams, xmlparams, start, end, step)

        # Create empty image list
        image_files = [str(Path(ioparams.image_file_dir) /
                           xmlparams.get_image_attrib(image))
                           for image in sequences[0]]

        # Iterate over images
        for i, image_file in enumerate(tqdm(image_files)):

            # Iterate over annotations
            for j, sequence in enumerate(sequences):

                assert image_file == (
                    str(Path(ioparams.image_file_dir) /
                    xmlparams.get_image_attrib(sequence[i])))

                if j == 0:
                    assert Path(image_file).exists()
                    img = cv2.imread(image_file).astype(np.float32)

                # Look for points in i-th element in j-th sequence
                annotations = sequence[i].findall(xmlparams.annotation_tag)

                if annotations:
                    img = self.renderer.draw(img, annotations, j, **kwargs)
                    img = img.astype(np.uint8)

            for writer in writers:
                try:
                    writer.write(img)
                except AttributeError:
                    writer(
                        f"{str(Path(image_file).stem).replace('/', '-')}-{str(i).zfill(5)}", img)

        for writer in writers:
            try:
                writer.release()
            except AttributeError:
                pass

    def _avi_integrated_rendering(self, **kargs):

        ioparams = self.renderer.ioparams
        xmlparams = self.renderer.xmlparams
        start = self.renderer.start
        end = self.renderer.end
        step = self.renderer.step
        writers = self.renderer.writers
        source = ioparams.source



    def run(self, **kwargs):
        """
        Render Annotations in `ioparams.output_format`.
        """

        xmlparams = self.renderer.xmlparams
        if not xmlparams._avi_integrated:
            self._xml_img_rendering(**kwargs)

        else:
            pass
