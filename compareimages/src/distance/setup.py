from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

def configuration(parent_package=None, top_path=''):
    config = dict(name='distance')
    extensions = []

    extensions += [Extension("distance", ["distance.pyx"],
                   include_dirs=[numpy.get_include()])]

    config.update(ext_modules=cythonize(extensions))
    return config


if __name__ == '__main__':
    from distutils.core import setup
    setup(**configuration(top_path=''))
