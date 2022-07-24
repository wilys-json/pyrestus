from pathlib import Path
import cv2
import numpy as np
import argparse
import json
from PIL import Image
from tqdm_batch import batch_process
from joblib import cpu_count
from typing import Tuple
import sys
sys.path.insert(0, '..')
from usv import read_DICOM_dir


class _Config:

    def __init__(self, input_file):
        with open(input_file, 'r') as json_file:
            json_dict = json.load(json_file)
            for attr, val in self._unpack_dict(json_dict).items():
                self.__dict__[attr] = val

    def _unpack_dict(self, dict_obj):
        output_dict = {}
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                output_dict.update(self._unpack_dict(value))
            else:
                output_dict.update({key:value})
        return output_dict

    @property
    def items(self):
        return self.__dict__


def getFrames(file: Path,
              output_dir: Path,
              keep_parents: int,
              cropping: Tuple[int],
              **kwargs) -> None:
    """
    generate PNG frames from `file` to `output_dir`
    """
    cap = cv2.VideoCapture(str(file))
    output_dir = output_dir / \
        Path(*file.parts[-(keep_parents + 1):-1]) / file.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    if cap.isOpened():
        i = kwargs.get('start_idx', 0)
        padding = kwargs.get('padding', False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_file = str(output_dir / f'{str(i).zfill(3)}.png')
            y1, y2, x1, x2 = cropping
            cropped = cv2.cvtColor(
                frame[y1:y2, x1:x2, :], cv2.COLOR_BGR2GRAY)
            if padding:
                frame = np.zeros((frame.shape[:-1]), dtype=np.int32)
                frame[y1:y2, x1:x2] = cropped
            else:
                frame = cropped
            cv2.imwrite(output_file, frame)
            i += 1
        cap.release()
        cv2.destroyAllWindows()


def getVideos(file: Path,
              output_dir: Path,
              keep_parents: int,
              cropping: Tuple[int],
              **kwargs) -> None:
    """
    generate cropped video from `file` to `output_dir`
    """

    video_format = kwargs.get('video_format', 'avi')
    codec = kwargs.get('codec', 'MJPG')
    fps = kwargs.get('fps', 15)

    cap = cv2.VideoCapture(str(file))
    output_dir = output_dir / Path(*file.parts[-(keep_parents + 1):-1]) / file.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f'{file.stem.replace(".", "-")}.{video_format}'

    fourcc = cv2.VideoWriter_fourcc(*codec)
    frame_size = (cropping[3] - cropping[2], cropping[1] - cropping[0])
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, frame_size)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            y1, y2, x1, x2 = cropping
            frame = frame[y1:y2, x1:x2]
            writer.write(frame)
        cap.release()
        cv2.destroyAllWindows()


def main():
    """
    Generate cropped videos / frames from video file.

    `python3 frame_generation [input_dir] [output_dir] [-v][-f]`

Worker 1: 100%|███████████████████████████████████| 2/2 [00:07<00:00,  3.58s/it]
Worker 2: 100%|███████████████████████████████████| 2/2 [00:07<00:00,  3.58s/it]
Worker 3: 100%|███████████████████████████████████| 2/2 [00:07<00:00,  3.58s/it]
Worker 4: 100%|███████████████████████████████████| 2/2 [00:07<00:00,  3.58s/it]
Worker 5: 100%|███████████████████████████████████| 2/2 [00:07<00:00,  3.58s/it]


    Default config:

    {
      "processing":
        {
          "cropping" : [175, 850, 223, 1062],
          "video_format" : "avi",
          "codec" : "MJPG",
          "fps" : 15
        },
      "io":
        {
          "keep_parents" : 2,
          "start_idx" : 0
        }
    }

    `cropping` is a variable tailored to specific cropping dimensions.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default='',
                        help='Path to input directory.')
    parser.add_argument('-o', '--output-dir', type=str, default='../outputs',
                        help='Path to output directory. Default: `../outputs`')
    parser.add_argument('-c', '--config', type=str, default='../frame_generation_config.json',
                        help='Path to config file in .json format.')
    parser.add_argument('-v', '--generate-videos', action='store_const',
                        dest='task', const='videos')
    parser.add_argument('-f', '--generate-frames', action='store_const',
                        dest='task', const='frames')

    args = parser.parse_args()

    tasks = {
        "videos" : getVideos,
        "frames" : getFrames
    }

    assert Path(args.input_dir).is_dir(), \
        f"`{args.input_dir}` is not a valid directory."

    assert Path(args.config).exists(), \
        f"Config file `{args.config}` not found."

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    input_dir = read_DICOM_dir(args.input_dir)
    config = _Config(args.config)
    task = tasks.get(args.task, getFrames)  # Default: generate frames

    batch_process(input_dir,
                  task,
                  n_workers=cpu_count(),
                  sep_progress=True,
                  output_dir=output_dir,
                  **config.items)


if __name__ == '__main__':

    main()
