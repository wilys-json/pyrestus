import numpy
from Cython.Build import cythonize
from pathlib import Path

# def prebuild():
#     from distutils.core import setup, Extension
#     config = dict(name='distance')
#     extensions = []
#
#     extensions += [Extension("distance", [str(Path(__file__).parent / "*.pyx")],
#                    include_dirs=[numpy.get_include()])]
#
#     config.update(ext_modules=cythonize(extensions, quiet=True))
#     setup(**config)


def configuration(parent_package='',top_path=None):
      from numpy.distutils.misc_util import Configuration, get_info
      cythonize([str(Path(__file__).parent / "*.pyx")], quiet=True)
      config = Configuration('', parent_package, top_path)
      src = ['*.c']
      inc_dir = [numpy.get_include()]
      config.add_extension('distance',sources=src, include_dirs=inc_dir)
      config.set_options(quiet=True)
      return config



if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
