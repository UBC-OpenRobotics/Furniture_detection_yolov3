from setuptools import setup

setup(
  name='furnituredetection',
  version='0.1.0',
  author='',
  packages=['furnituredetection'],
  scripts=[],
  description='Using yolov3 to detect furnitures',
  long_description=open('README.md').read(),
  install_requires=[
      "opencv-python",
  ],
)