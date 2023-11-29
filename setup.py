from setuptools import find_packages, setup
import numpy


# python setup.py install
setup(name='gojo',
      version='0.0.10',
      license='MIT',
      description='Package with diverse Machine Learning and Deep Learning pipelines.',
      author='Fernando García Gutiérrez',
      author_email='fegarc05@ucm.es',
      url='https://github.com/FernandoGaGu/GoJo-ML',
      #install_requires=[],
      keywords=['Machine Learning', 'Deep Learning'],
      packages=find_packages(),
      include_dirs=[numpy.get_include()],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'],
      )
