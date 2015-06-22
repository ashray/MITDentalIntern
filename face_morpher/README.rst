Face Morpher
============

| Warp, average and morph human faces!
| Scripts will automatically detect frontal faces and skip images if
  none is detected.

Built with Python 2.7, OpenCV, Numpy, Scipy, Stasm.

Requirements
--------------
-  Install `OpenCV`_: `Mac installation steps`_
-  ``pip install -r requirements.txt``

Either use as:

-  `Local command-line utility`_
-  `pip library`_

.. _`Local command-line utility`:

Use as local command-line utility
---------------------------------
::

    $ git clone https://github.com/alyssaq/face_morpher

Morphing Faces
--------------

Morph from a source to destination image:

::

    python facemorpher/morpher.py --src=<src_imgpath> --dest=<dest_imgpath>

Morph through a series of images in a folder:

::

    python facemorpher/morpher.py --images=<folder>

All options listed in ``morpher.py`` (pasted below):

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

Averaging Faces
---------------

Average faces from all images in a folder:

::

    python facemorpher/averager.py --images=<images_folder>

All options listed in ``averager.py`` (pasted below):

::

    Face averager

    Usage:
      averager.py --images=<images_folder> [--blur] [--alpha] [--plot]
                [--width=<width>] [--height=<height>] [--out=<filename>]

    Options:
      -h, --help         Show this screen.
      --images=<folder>  Folder to images (.jpg, .jpeg, .png)
      --blur             Flag to blur edges of image [default: False]
      --alpha            Flag to save with transparent background [default: False]
      --width=<width>    Custom width of the images/video [default: 500]
      --height=<height>  Custom height of the images/video [default: 600]
      --out=<filename>   Filename to save the average face [default: result.png]
      --plot             Flag to display the average face [default: False]
      --version          Show version.

Steps (facemorpher folder)
--------------------------

1. Locator
^^^^^^^^^^

-  Locates face points (using `stasm`_)
-  For a different locator, return an array of (x, y) control face
   points

2. Aligner
^^^^^^^^^^

-  Align faces by resizing, centering and cropping to given size

3. Warper
^^^^^^^^^

-  Given 2 images and its face points, warp one image to the other
-  Triangulates face points
-  Affine transforms each triangle with bilinear interpolation

4a. Morpher
^^^^^^^^^^^

-  Morph between 2 or more images

4b. Averager
^^^^^^^^^^^^

-  Average faces from 2 or more images

Blender
^^^^^^^

Optional blending of warped image:

-  Weighted average
-  Alpha feathering
-  Poisson blend

Examples - `Being John Malkovich`_
----------------------------------

Create a morphing video between the 2 images:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| ``> python facemorpher/morpher.py --src=alyssa.jpg --dest=john_malkovich.jpg``
| ``--out_video=out.avi``

(out.avi played and recorded as gif)

.. figure:: https://raw.github.com/alyssaq/face_morpher/master/examples/being_john_malvokich.gif
   :alt: gif

Save the frames to a folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| ``> python facemorpher/morpher.py --src=alyssa.jpg --dest=john_malkovich.jpg``
| ``--out_frames=out_folder --num=30``

Plot the frames:
^^^^^^^^^^^^^^^^

| ``> python facemorpher/morpher.py --src=alyssa.jpg --dest=john_malkovich.jpg``
| ``--num=12 --plot``

.. figure:: https://raw.github.com/alyssaq/face_morpher/master/examples/plot.png
   :alt: plot

Average all face images in a folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

85 images used

| ``> python facemorpher/averager.py --images=images --blur --alpha``
| ``--width=220 --height=250``

.. figure:: https://raw.github.com/alyssaq/face_morpher/master/examples/average_faces.png
   :alt: average\_faces

.. _`pip library`:

Use as library
---------------------------------
::

    $ pip install facemorpher

Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Additional options are exactly the same as the command line

::

    import facemorpher

    # Get a list of image paths in a folder
    imgpaths = facemorpher.list_imgpaths('imagefolder')

    # To morph, supply an array of face images:
    facemorpher.morpher(imgpaths, plot=True)

    # To average, supply an array of face images:
    facemorpher.averager(['image1.png', 'image2.png'], plot=True)


Details
------------
-  Data for the haar face classifiers are in the ``facemorpher/data``
   folder
-  Stasm binary in ``facemorpher/bin/stasm_util``. You can build a new
   stasm binary with the `Stasm 4 build scripts`_.

Documentation
-------------

http://alyssaq.github.io/face_morpher

Build & publish Docs
^^^^^^^^^^^^^^^^^^^^

::

    ./scripts/publish_ghpages.sh

License
-------
`MIT`_

.. _Being John Malkovich: http://www.rottentomatoes.com/m/being_john_malkovich
.. _Mac installation steps: http://scriptogr.am/alyssa/post/installing-opencv-on-mac-osx-with-homebrew
.. _MIT: http://alyssaq.github.io/mit-license/
.. _OpenCV: http://opencv.org
.. _Stasm 4 build scripts: https://github.com/alyssaq/stasm_build
.. _stasm: http://www.milbo.users.sonic.net/stasm
