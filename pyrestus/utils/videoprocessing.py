import cv2
from pathlib import Path
from datetime import datetime
from typing import Tuple

__all__ = ['create_video_writer', '_format_filename']

def _format_filename(format:str, **kwargs)->str:

    if not format:
        return ''

    name = '' if not kwargs.get('name') else kwargs.get('name')
    pid = '' if not kwargs.get('pid') else f'({kwargs.get("pid")})'
    time = ('' if not kwargs.get('start_time')
               else "_".join(f'{kwargs.get("start_time")}'.split(" ")))
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    filename = f'{name}{pid}{time}' if all([name, pid, time]) else timestamp
    filename += f'.{format.split(".")[-1]}'

    return filename


def create_video_writer(format:str,
                         frame_size:Tuple[int],
                         fps:float,
                         **kwargs)->cv2.VideoWriter:

    filename = _format_filename(format, **kwargs)


    if not filename : return None
    output_dir = kwargs.get('output_dir')
    if output_dir:
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)
        filename = output_dir / filename

    codec = 'MJPG' if not kwargs.get('codec') else kwargs.get('codec')
    fourcc = cv2.VideoWriter_fourcc(*codec)

    return cv2.VideoWriter(str(filename), fourcc, fps, frame_size)
