import pydicom
import numpy as np
import pandas as pd
import time
import math
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from dateutil import parser
from pathlib import Path
from pydicom import FileDataset
from dataclasses import dataclass
from typing import Tuple, Union, List
from warnings import warn
from IPython import display
from pydicom.errors import InvalidDicomError

np.random.seed(0)

US_SEQUENCE = lambda x : getattr(x, 'SequenceOfUltrasoundRegions')

US_ATTRIBUTES = (
 lambda x: (
     setattr(x, 'fps',
             1000 / float(getattr(x,'FrameTime')))
 ),
  lambda x: (
     setattr(x, 'frame_delay',
            float(getattr(x, 'FrameTime')))
 ),
 lambda x: (
        setattr(x, 'dob', parser.parse('19000101'))
        if not getattr(x, 'PatientBirthDate')
        else setattr(x, 'dob',
             parser.parse(getattr(x, 'PatientBirthDate')))
 ),
 lambda x: (
     setattr(x, 'pid',
             getattr(x, 'PatientID'))
 ),
 lambda x: (
     setattr(x, 'name',
             getattr(x, 'PatientName'))
 ),
 lambda x: (
     setattr(x, 'procedure',
             getattr(x, 'PerformedProcedureStepDescription'))
 ),
 lambda x: (
     setattr(x, 'start_time',
            parser.parse(getattr(x, 'ContentDate')
                        +getattr(x, 'ContentTime')))
 ),
 lambda x: (
     setattr(x, 'color_space',
            getattr(x, 'PhotometricInterpretation'))
 ),
 lambda x: (
     setattr(x, 'delta_x',
            getattr(US_SEQUENCE(x)[0], 'PhysicalDeltaX'))
 ),
 lambda x: (
     setattr(x, 'delta_y',
            getattr(US_SEQUENCE(x)[0], 'PhysicalDeltaY'))
 ),
 lambda x: (
     setattr(x, 'unit_x',
             getattr(US_SEQUENCE(x)[0], 'PhysicalUnitsXDirection'))
 ),
 lambda x: (
     setattr(x, 'unit_y',
            getattr(US_SEQUENCE(x)[0], 'PhysicalUnitsYDirection'))
 ),
 lambda x: (
     setattr(x, '_roi_coordinates',
        (
            getattr(US_SEQUENCE(x)[0], 'RegionLocationMinX0'),
            getattr(US_SEQUENCE(x)[0], 'RegionLocationMinY0'),
            getattr(US_SEQUENCE(x)[0], 'RegionLocationMaxX1'),
            getattr(US_SEQUENCE(x)[0], 'RegionLocationMaxY1')
        )
    )
 ),
)

COLOR_CONVERSION = {
    'RGB' : (lambda x : x),
    'YBR_FULL_422' : (
        lambda x: cv2.cvtColor(x, cv2.COLOR_YUV2RGB)
    )
}


@dataclass(init=False)
class UltrasoundVideo(FileDataset):

    color_space:str='RGB'
    delta_x:float=0.0
    delta_y:float=0.0
    unit_x:int=3
    unit_y:int=3
    dob:datetime
    name:str=''
    pid:str=''
    start_time:datetime=datetime(1900,1,1)
    procedure:str=''
    fps:float=0.0
    transducer:str='linear'
    data:np.ndarray=np.array([], dtype='uint8')
    _roi_coordinates:Tuple[int]=(0,0,0,0)
    _smoothing_kernel:Tuple[int]=(3,3)


    def __init__(self, file: Union[str, Path], **kwargs):

        try:
            # Initialize FileDataset
            file_dataset = pydicom.dcmread(str(file))
            super().__dict__.update(file_dataset.__dict__)

            # Set attributes w.r.t. Metadata
            for setter in US_ATTRIBUTES:
                try:
                    setter(self)
                except AttributeError:
                    warn(f"Unable to find attribute for {setter}")

            # Handle non-video input
            if len(self.pixel_array.shape) != 4:
                self.data = self.pixel_array
            else: self._preprocess()

        # if `file` is in non-DICOM video format
        except InvalidDicomError:
            _data = []
            # read Video file
            cap = cv2.VideoCapture(str(file))
            # stream Video into `_data`
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _data += [frame]
            if not _data:
                warn(f"Invalid video format. {__name__} data is empty.")
            else:
                self.data = np.array(_data)

            # Override metadata
            for attr, value in kwargs.items():
                setattr(self, attr, value)


    def _has_data(self):
        return self.data.size > 0


    def _color_conversion(self, image:np.ndarray)->np.ndarray:
        """
        Convert to RGB image w.r.t. defined color space.
        """
        return COLOR_CONVERSION[self.color_space](image)


    def _preprocess(self):
        """
        Preprocess ultrasound video images.
        """
        X0, Y0, X1, Y1 = self._roi_coordinates
        frames = self.pixel_array.shape[0]
        channels = self.pixel_array.shape[3]
        self.data = np.empty(shape=(frames, Y1-Y0,
                                    X1-X0, channels), dtype=np.uint8)

        # Iterate through all images
        for i in range(self.pixel_array.shape[0]):
            self.data[i] = self._color_conversion(self.pixel_array[i][Y0:Y1, X0:X1])

        # self.data = np.array(images)


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        if self._has_data():
            for i in range(len(self.data)):
                yield self.data[i]


    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except KeyError:
            if self._has_data():
                return self.data[idx]

    @property
    def frame_size(self):
        return self.data.shape[1:3]

    @property
    def channels(self):
        return self.data.shape[-1]

    def show(self):
        """
        Play Ultrasound Video.
        """
        if self._has_data():
            frame_delay = int(self.get('frame_delay', 1))
            for i, image in enumerate(self.data):
                cv2.imshow(f'{self.name} at {self.start_time}', image)
                key = cv2.waitKey(frame_delay)
                if (i == len(self.data) - 1) or key == 27:
                    cv2.destroyAllWindows()
                    break

            cv2.waitKey(1)
            cv2.destroyAllWindows()


    # def _find_boundaries(self)->Tuple[int]:
    #     """
    #     Find universal left & right boundaries of ROI
    #     by randomly sampling log(n) number of frames,
    #     where n = total number of frames.
    #     """
    #
    #     points = []
    #     length = self.pixel_array.shape[0]
    #     sample_length =  math.ceil(math.log(length))
    #
    #     # Random sampling of frames
    #     sample_segments = np.random.randint([length] * sample_length)
    #
    #     # Find right- & left-most point of ROI
    #     for i in sample_segments:
    #         image = self._color_conversion(self.pixel_array[i])
    #         image = self._get_roi(image)
    #         points += detect_lines(image=image,
    #                                kernel=self._smoothing_kernel,
    #                                transducer=self.transducer)
    #
    #     # Remove None from set
    #     points = np.array(list(set(points).difference({None})))
    #
    #     # Left-most & Right-most points on Y axis
    #     return points.min(), points.max()
