

from enum import Enum

from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class Image:
    wide_field: bool
    treated: bool|None
    channel: "Channel"
    image: npt.NDArray
    file_name: str


class Channel(Enum):
    TRANS = 0  # transmission (grayscale)
    DAPI = 1  # blue
    GFP = 2  # green
