import torchvision.transforms.functional as TF
import random
from typing import Sequence
class RotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        im0, im1 = x['image_0'], x['image_1']
        angle = random.choice(self.angles)
        im0 = TF.rotate(im0, angle)
        im1 = TF.rotate(im1, angle)
        return {'image_0': im0, 'image_1': im1}