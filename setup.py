from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='discopy-rknaebel',
      version='0.1.0',
      description='Shallow Discourse Parser',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/rknaebel/discopy',
      author='Rene Knaebel',
      author_email='rknaebel@uni-potsdam.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'nltk',
          'joblib',
          'sklearn',
          'sklearn-crfsuite'
          'ujson'
      ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'discopy=main:main',
              'discopy-parse=parse'
          ],
      }
      )
