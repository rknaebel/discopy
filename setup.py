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
          'numpy>=1.18.0',
          'cython',
          'nltk>=3.4',
          'joblib',
          'sklearn',
          'sklearn-crfsuite',
          'ujson>=2.0.0',
          'tensorflow>=2.1.0'
          'torch>=1.4.0'
          'spacy>=2.3.5',
          'supar>=1.0.0',
          'transformers>=3.5.0'
      ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'discopy-train=cli.train:main',
              'discopy-test=cli.test:main',
              'discopy-parse=cli.parse:main',
          ],
      },
      python_requires='>=3.7',
      )
