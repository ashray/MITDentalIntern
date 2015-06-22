"""
::

  Morph from source to destination face or
  Morph through all images in a folder

  Usage:
    morpher.py (--src=<src_path> --dest=<dest_path> | --images=<folder>)
              [--width=<width>] [--height=<height>]
              [--num=<num_frames>] [--fps=<frames_per_second>]
              [--out_frames=<folder>] [--out_video=<filename>]
              [--alpha] [--plot]

  Options:
    -h, --help              Show this screen.
    --src=<src_imgpath>     Filepath to source image (.jpg, .jpeg, .png)
    --dest=<dest_path>      Filepath to destination image (.jpg, .jpeg, .png)
    --images=<folder>       Folderpath to images
    --width=<width>         Custom width of the images/video [default: 500]
    --height=<height>       Custom height of the images/video [default: 600]
    --num=<num_frames>      Number of morph frames [default: 20]
    --fps=<fps>             Number frames per second for the video [default: 10]
    --out_frames=<folder>   Folder path to save all image frames
    --out_video=<filename>  Filename to save a video
    --alpha                 Flag to save transparent background [default: False]
    --plot                  Flag to plot images [default: False]
    --version               Show version.
"""

from docopt import docopt
import scipy.ndimage
# import numpy as np
import os
import locator
import aligner
# import warper
# import blender
# import plotter
import videoer
import cv2

def verify_args(args):
  if args['--images'] is None:
    valid = os.path.isfile(args['--src']) & os.path.isfile(args['--dest'])
    if not valid:
      print('--src=%s or --dest=%s are not valid images' % (
        args['--src'], args['--dest']))
      exit(1)
  else:
    valid = os.path.isdir(args['--images'])
    if not valid:
      print('--images=%s is not a valid directory' % args['--images'])
      exit(1)

def load_image_points(path, size):
  print 'in load image points'
  img = scipy.ndimage.imread(path)[..., :3]
  points = locator.face_points(path)
  if len(points) == 0:
    print 'No face in %s' % path
    return None, None
  else:
    print 'Face landmark points : ',points
    for i in range (0,15):
      cv2.line(img, (points[i][0],points[i][1]), (points[i][0],points[i][1]), (0,255,0),5)

    cv2.line(img, (points[18][0],points[18][1]), (points[18][0],points[18][1]), (255,224,0),5)
    cv2.line(img, (points[21][0],points[21][1]), (points[21][0],points[21][1]), (255,224,0),5)
    cv2.line(img, (points[22][0],points[22][1]), (points[22][0],points[22][1]), (255,224,0),5)
    cv2.line(img, (points[25][0],points[25][1]), (points[25][0],points[25][1]), (255,224,0),5)

    cv2.line(img, (points[30][0],points[30][1]), (points[30][0],points[30][1]), (255,0,0),5)
    cv2.line(img, (points[40][0],points[40][1]), (points[40][0],points[40][1]), (255,0,0),5)

    cv2.line(img, (points[54][0],points[54][1]), (points[54][0],points[54][1]), (0,255,0),5)
    cv2.line(img, (points[56][0],points[56][1]), (points[56][0],points[56][1]), (0,255,0),5)
    cv2.line(img, (points[58][0],points[58][1]), (points[58][0],points[58][1]), (0,255,0),5)

    cv2.line(img, (points[59][0],points[59][1]), (points[59][0],points[59][1]), (0,0,255),5)
    cv2.line(img, (points[65][0],points[65][1]), (points[65][0],points[65][1]), (0,0,255),5)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    return aligner.resize_align(img, points, size)

# def load_valid_image_points_32(imgpaths, size):
#   print 'load valid image points called'
#   print 'dsfknsdfsdfsdnfs df'
#   for path in imgpaths:
#     img, points = load_image_points(path, size)
#     if img is not None:
#       print path, 'is none'
#       yield (img, points)

def list_imgpaths(images_folder=None, src_image=None, dest_image=None):
  if images_folder is None:
    yield src_image
    yield dest_image
  else:
    for fname in os.listdir(images_folder):
      if (fname.lower().endswith('.jpg') or
         fname.lower().endswith('.png') or
         fname.lower().endswith('.jpeg')):
        yield os.path.join(images_folder, fname)


def morpher(imgpaths, width=500, height=600, num_frames=20, fps=10,
            out_frames=None, out_video=None, alpha=False, plot=False):
  """
  Create a morph sequence from multiple images in imgpaths

  :param imgpaths: array or generator of image paths
  """
  print 'in morpher'

  video = videoer.Video(out_video, fps, width, height)
  print 'calling load valid image points'

  # images_points_gen = load_valid_image_points_32(imgpaths, (height, width))
  print 'load valid image points called'
  print 'dsfknsdfsdfsdnfs df'
  for path in imgpaths:
    print 'path',path
    img, points = load_image_points(path, (height, width))

  video.end()

if __name__ == "__main__":
  args = docopt(__doc__, version='Face Morpher 1.0')
  verify_args(args)
  morpher(list_imgpaths(args['--images'], args['--src'], args['--dest']),
          int(args['--width']), int(args['--height']),
          int(args['--num']), int(args['--fps']),
          args['--out_frames'], args['--out_video'],
          args['--alpha'], args['--plot'])

