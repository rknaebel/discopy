from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='discopy-rknaebel',
      version='1.1.0',
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
          'nltk>=3.4',
          'joblib',
          'sklearn',
          'sklearn-crfsuite',
          'tensorflow>=2.1.0',
          'transformers==4.30.0',
          'fastapi==0.67.0',
          'discopy-data-rknaebel',
      ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'discopy-train=cli.train:main',
              'discopy-eval=cli.eval:main',
              'discopy-parse=cli.parse:main',
              'discopy-predict=cli.predict:main',
              'discopy-nn-train=cli.bert.train:main',
              'discopy-nn-parse=cli.bert.parse:main',
              'discopy-nn-predict=cli.bert.predict:main',
          ],
      },
      python_requires='>=3.7',
      )
