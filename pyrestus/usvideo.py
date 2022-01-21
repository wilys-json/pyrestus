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

import pydicom
import numpy as np
import pandas as pd
import time
import datetime
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
import pyximport
pyximport.install()
from .utils import convert_color, _format_filename, create_video_writer
from joblib import Parallel, delayed

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
        lambda x: cv2.cvtColor(x, cv2.COLOR_YUV2BGR)
    )
}

@dataclass
class converter:

    """
    WORK IN PROGRESS.
    TODO: refactor this class & UltrasoundVideo.



    from joblib import cpu_count


    %%timeit
    dicoms = [file for file in dicom.iterdir()] # dicom : DICOM directory
    dicoms.sort()
    batch_size = ceil(len(dicoms) / cpu_count())
    batches = [dicoms[i:i+batch_size]for i in range(0, len(dicoms), batch_size)]
    n_jobs = len(batches)
    converter(n_jobs, 'threading').DICOMs2AVIs(batches)

    >> for 56 files:
    >> 41.3 s ± 907 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    """

    n_jobs:int
    backend:str

    def __post_init__(self):
        self.parallel = Parallel(n_jobs=self.n_jobs,
                                 backend=self.backend)

    @staticmethod
    def convertAVI_file(file):
        out = str(file).split('/')[-1]
        try:
            filedataset = pydicom.dcmread(file)
            try:
                sequence = filedataset.SequenceOfUltrasoundRegions[0]
            except AttributeError:
                return
            X0, Y0, X1, Y1 = (sequence.RegionLocationMinX0,
                              sequence.RegionLocationMinY0,
                              sequence.RegionLocationMaxX1,
                              sequence.RegionLocationMaxY1)
            data = filedataset.pixel_array
            if len(data.shape) != 4 : return
            frame_size = data[:, Y0:Y1, X0:X1, :].shape[1:3][::-1]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = (1000 / filedataset.FrameTime)
            writer = cv2.VideoWriter(f'{out}.avi', fourcc, fps, frame_size)
            for frame in data:
                frame = cv2.cvtColor(frame[Y0:Y1, X0:X1, :], cv2.COLOR_YUV2RGB)
                writer.write(frame)
            writer.release()
        except InvalidDicomError:
            return

    @staticmethod
    def convertAVI_batch(batch:list):
        [converter.convertAVI_file(file) for i, file in enumerate(batch)]

    def DICOMs2AVIs(self, batches):
        self.parallel(delayed(converter.convertAVI_batch)(batch) for batch in batches)


@dataclass(init=False)
class UltrasoundVideo(FileDataset):

    color_space:str='RGB'
    delta_x:float=0.0
    delta_y:float=0.0
    unit_x:int=3
    unit_y:int=3
    dob:datetime=datetime(1900,1,1)
    name:str=''
    pid:str=''
    start_time:datetime=datetime(1900,1,1)
    procedure:str=''
    fps:float=0.0
    transducer:str='linear'
    data:np.ndarray=np.array([], dtype='uint8')
    _roi_coordinates:Tuple[int]=(0,0,0,0)
    _smoothing_kernel:Tuple[int]=(3,3)
    _use_cython:bool = False
    _is_video:bool = False


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

            # Override metadata
            for attr, value in kwargs.items():
                setattr(self, attr, value)

            # Handle non-video input
            if len(self.pixel_array.shape) != 4:
                self.data = self.pixel_array
            else:
                self._preprocess(**kwargs)
                self._is_video = True

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
                self._is_video = True

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


    def _preprocess(self, **kwargs):
        """
        Preprocess ultrasound video images.
        """
        # Retrieve keypoints
        X0, Y0, X1, Y1 = self._roi_coordinates
        roi_dimensions = self.pixel_array[:, Y0:Y1, X0:X1, :].shape
        video_writer = create_video_writer(format=kwargs.get('video_format'),
                                            frame_size=roi_dimensions[1:3][::-1],
                                            fps=self.fps,
                                            name=self.name,
                                            pid=self.pid,
                                            start_time=self.start_time,
                                            codec=kwargs.get('codec'))

        if not self._use_cython:
            # Slice ROI
            self.data = np.empty(roi_dimensions, dtype=np.uint8)
            func = COLOR_CONVERSION[self.color_space]

            # Iterate through all images
            for i, frame in enumerate(self.pixel_array):
                self.data[i] = func(frame[Y0:Y1, X0:X1, :])
                if video_writer is not None:
                    video_writer.write(self.data[i])

            if video_writer is not None:
                video_writer.release()
        # Use Cython code; timeit diff is not significant
        else:
            self.data = convert_color(np.asarray(self.pixel_array,
                                                 dtype=np.uint8,
                                                 order='c'),
                                      self._roi_coordinates,
                                      self.color_space)


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

    def show(self, **kwargs):
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

    def saveimg(self, folder:str=''):
        save_dir = (Path.cwd() if not folder or not Path(folder).is_dir()
                    else Path(folder))
        save_dir_suffix, file_suffix = _format_filename(format='png',
                                        name=self.name,
                                        pid=self.pid,
                                        start_time=self.start_time).split('.')
        save_dir = save_dir / save_dir_suffix
        save_dir.mkdir(parents=True, exist_ok=True)
        pad = len(str(len(self.data)))
        for i, image in enumerate(self.data):
            cv2.imwrite(f'{save_dir}/{str(i).zfill(pad)}.{file_suffix}', image)
