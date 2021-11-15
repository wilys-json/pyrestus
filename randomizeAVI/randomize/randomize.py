import os
import pandas as pd
import numpy as np
import glob
import argparse
import sys
import cv2
from copy import deepcopy

def listdir(path):
   if os.path.exists(path):
       return glob.glob(os.path.join(path, "*"))


def shuffle(file_list, split_factor=0.3):
   assert (0 < split_factor <= 1), "Splitting Factor (f) can only be 0 < f <= 1."
   list1 = deepcopy(file_list)
   list2 = deepcopy(file_list)
   np.random.shuffle(list1)
   np.random.shuffle(list2)
   return list1 + list2[int(len(list2) * split_factor * -1):]


def generate_file(video_file, idx, folder='output'):
   cap = cv2.VideoCapture(video_file)
   frame_width = int(cap.get(3))
   frame_height = int(cap.get(4))
   padding = 50
   org = (padding, frame_height - padding)
   font = cv2.FONT_HERSHEY_SIMPLEX
   fontScale = 1
   color = (255, 255, 255)
   thickness = 2


   out = cv2.VideoWriter('{}/{}.avi'.format(folder, str(idx)),cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (frame_width,frame_height))
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


def generate_files(shuffled_list, folder='output'):
   if not os.path.exists(folder):
       os.mkdir(folder)
   file_df = pd.DataFrame(shuffled_list)
   file_df = file_df.reset_index()

   for i, (index, file) in enumerate(file_df.values):
       generate_file(file, index, folder)
       print("generating {} / {} files...".format(i+1, len(file_df)))

   file_df.to_csv('.Randomized.csv', index=False, header=False)


def randomize(files_path, split_factor=0.3, output_folder='output'):
   file_list = listdir(files_path)
   print("Randomizing {} files".format(len(file_list)))
   file_list = shuffle(file_list, split_factor)
   print("Randomization done.\n {} files will be generated: ".format(len(file_list)))
   generate_files(file_list, folder=output_folder)
   print("Randomized files generated. Please checkout files in {}/ .".format(output_folder))
