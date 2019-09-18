"""
Object Detection visualization kit for Surromind Robotics
2019/09 Sangbum Kim <sbkim@surromind.ai>
"""

from typing import Union, List
import numpy as np
from PIL import Image, ImageFont, ImageDraw


# Predefined color maps
COLORS = {
  # Ref: https://material.io/design/color/#tools-for-picking-colors
  'material': ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4',
               '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722'],
}


def draw_detections(image: Union[Image.Image, np.ndarray, list],
                    boxes: list,
                    class_ids: List[int],
                    class_names: List[str],
                    scores: Union[List[int], List[float], None] = None,
                    color_map: Union[list, str] = 'material',
                    color_by_class: bool = True,
                    font_path: str = 'D2Coding.ttf') -> Image.Image:
  """
  Draw the bounding boxes to each detection and label them
  :param image: input image
  :param boxes: list of bounding boxes for each detection represented in length 4 list
  :param class_ids: list of class ids for each detection
  :param class_names: list of class names for dataset
  :param scores: list of confidence score for each detection. Pass None to draw without scores.
  :param color_map: name of built-in color map to use, or custom list of colors
  :param color_by_class: whether to use same color for same class objects. Otherwise, color is chosen instance-wise.
  :param font_path: path to font file to use, or simply font name in case of installed font
  :return: PIL Image object
  """

  # boxes, class_ids, scores (if present) should have same length
  assert len(boxes) == len(class_ids)
  if scores is not None:
    assert len(boxes) == len(scores)

  if isinstance(image, list):
    image = np.asarray(image)
  if isinstance(image, np.ndarray):
    image = Image.fromarray(image)
  image = image.convert('RGB')
  draw = ImageDraw.Draw(image, 'RGB')

  if len(boxes) == 0:
    return image

  colors = COLORS[color_map] if isinstance(color_map, str) else color_map
  font = ImageFont.truetype(font_path, size=16)

  for idx in range(len(boxes)-1, -1, -1):
    box, cls, score = boxes[idx], class_ids[idx], scores[idx] if scores is not None else None
    x1, y1, x2, y2 = box
    color = colors[(cls if color_by_class else idx) % len(colors)]

    # draw bounding box
    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=4)

    name = class_names[cls]
    text = '{}: {:.2f}'.format(name, score) if score else name
    w, h = font.getsize(text)

    # draw label with background box
    draw.rectangle(((x1, y1), (x1 + w, y1 - h)), color)
    draw.text((x1, y1 - h), text, font=font, fill='#FFFFFF')

  return image
