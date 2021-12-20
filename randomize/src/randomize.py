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

import os
import pandas as pd
import numpy as np
import glob
import cv2
from tqdm import tqdm
from copy import deepcopy

np.random.seed(1337)

IMAGE_FORMATS = {'png', 'jpg', 'jpeg', 'bmp'}
VIDEO_FORMATS = {'mp4', 'mov', 'avi'}
HIDDEN_DATA = '.randomized'
OUTPUT_DIR = 'outputs'


def listdir(path:str)->list:
   """
   List files in a directory.
   """
   if os.path.exists(path):
       return glob.glob(os.path.join(path, "*"))


def shuffle(file_list:list, split_factor:float=0.3)->list:
   """
   Shuffle the input list and repeat `split_factor` percentage of list.
   """
   assert (0 < split_factor <= 1), \
    "Splitting Factor (f) can only be 0 < f <= 1."
   list1 = deepcopy(file_list)
   list2 = deepcopy(file_list)
   np.random.shuffle(list1)
   np.random.shuffle(list2)
   split = int(len(list2) * split_factor * -1)
   if split == 0: return list1
   return list1 + list2[split:]


def generate_video(video_file:str , idx:int, folder:str='output'):
   """
   Generate a video from `video_file` with `idx`
   in the lower left corner in `folder`.
   """
   cap = cv2.VideoCapture(video_file)
   frame_width = int(cap.get(3))
   frame_height = int(cap.get(4))
   padding = 50
   org = (padding, frame_height - padding)
   font = cv2.FONT_HERSHEY_SIMPLEX
   fontScale = 1
   color = (255, 255, 255)
   thickness = 2


   out = cv2.VideoWriter('{}/{}.avi'.format(folder,
        str(idx)), cv2.VideoWriter_fourcc('M','J','P','G'),
        cap.get(cv2.CAP_PROP_FPS), (frame_width,frame_height))
   while cap.isOpened():
       ret, frame = cap.read()
       if ret:
           cv2.putText(frame, str(idx), org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
           out.write(frame)
       else:
           break

   out.release()
   cap.release()


def generate_image(image_file:str , idx:int , folder:str='output'):
    """
    Generate an image from `image_file` with `idx`
    in the lower left corner to `folder`.
    """
    img = cv2.imread(image_file)
    h, w, c = img.shape
    padding = 50
    org = (padding, h - padding)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(img, str(idx), org, font,
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite('{}/{}.png'.format(folder, str(idx)), img)


def generate_files(timestamp, shuffled_list, extension, folder='output'):
   if not os.path.exists(folder):
       os.mkdir(folder)
   file_df = pd.DataFrame(shuffled_list)
   file_df = file_df.reset_index()

   file_df.to_csv(os.path.join(OUTPUT_DIR,
                              '{}_{}.csv'.format(HIDDEN_DATA, timestamp)),
                 index=False, header=False)

   tqdm.pandas()

   """
   # DEPRECATED: enumerating the rows.

   for i, (index, file) in enumerate(file_df.values):
       if extension.intersection(VIDEO_FORMATS):
           generate_video(file, index, folder)
       else:
           generate_image(file, index, folder)
       print("generating {} / {} files...".format(i+1, len(file_df)))
   """
   if extension.intersection(VIDEO_FORMATS):
       (file_df.reset_index()
               .progress_apply(lambda row: generate_video(row[0],
                                           row['index'],
                                           folder), axis=1))
   else:
      (file_df.reset_index()
              .progress_apply(lambda row: generate_image(row[0],
                                          row['index'],
                                          folder), axis=1))


def randomize(timestamp, files_path, split_factor=0.3, output_folder='output'):

   assert os.path.exists(files_path), "File path not found."

   file_list = listdir(files_path)
   extensions = set([(file.split('.')[-1]).lower() for file in file_list])
   invalid_formats = list(
                    extensions.difference(VIDEO_FORMATS.union(IMAGE_FORMATS))
                    )

   if invalid_formats:
       raise ValueError('Invalid file formats: {}'.format(invalid_formats))
   if len(extensions.intersection(VIDEO_FORMATS)) > 1:
       raise ValueError('Multiple file type found. Please unify file formats.')

   print("Randomizing {} files".format(len(file_list)))
   file_list = shuffle(file_list, split_factor)

   print("Randomization done.\n {} files will be generated: ".format(len(file_list)))
   generate_files(timestamp, file_list, extensions, folder=output_folder)

   print("Randomized files generated. Please checkout files in {} .".format(output_folder))
