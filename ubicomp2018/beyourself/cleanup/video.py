import subprocess
import logging
from ..core.util import subtract_relative_time


logger = logging.getLogger(__name__)


def cut_video(inpath, start, end, outpath):
    '''
    Split video using ffmpeg from start to end time (relative time)
    
    Parameters:

        inpath: path to the original video
        start, end: string, format as HH:MM:SS.xxx
        outpath: path to the output video

    NOTE: in ffmpeg, -t is the DURATION, not end time
    '''

    cmd = 'ffmpeg -hide_banner -ss {} -i {} -c copy -t {} {}'.format(start, inpath, \
        subtract_relative_time(start, end), outpath)
    logger.info(cmd)
    
    subprocess.call(cmd, shell=True)


def convert(inpath, outpath):

    cmd = 'ffmpeg -hide_banner -i {} -s 640x360 -vcodec h264 -acodec aac -strict -2 {}'.format(inpath, outpath)

    logger.info(cmd)

    subprocess.call(cmd, shell=True)
