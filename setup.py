 from setuptools import setup

 setup(
   name='bagdetection',
   version='0.1.0',
   author='',
   packages=['bagdetection'],
   scripts=[],
   description='Using yolov5 to detect bags',
   long_description=open('README.txt').read(),
   install_requires=[
       "tenserflow",
       "pytest",
   ],
)