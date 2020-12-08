from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='discopy-rknaebel',
      version='1.0.0',
      description='Shallow Discourse Parser',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/rknaebel/discopy',
      author='Rene Knaebel',
      author_email='rene.knaebel@uni-potsdam.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'cython',
          'nltk',
          'joblib',
          'sklearn',
          'sklearn-crfsuite',
          'ujson',
          'spacy',
          'supar',
      ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'discopy-train=cli.train:main',
              'discopy-test=cli.test:main',
              'discopy-parse=cli.parse:main',
          ],
      }
      )
