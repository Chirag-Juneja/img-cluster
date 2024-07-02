import cv2 as cv
from pinaka.utils import logger
from pinaka.video.frame_selector import FrameSelector


class Panorama:
    def __init__(self, dim=512):
        self.dim = dim
        self.fs = FrameSelector()
        self.stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)

    def convert(self, video_path):
        frames, selected_frames = self.fs.select(video_path)
        logger.debug(f"Total Frames:{len(frames)}")
        logger.debug(f"Selected Frames:{len(selected_frames)}")

        selected_frames = [cv.resize(f, (self.dim, self.dim)) for f in selected_frames]

        status, panorama = self.stitcher.stitch(selected_frames)

        return panorama
