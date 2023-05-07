import os

from setuptools import find_packages, setup


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()


setup(
    name="edit3d",
    version="0.0.1",
    description="A tool for generating and updating 3D shapes from sketches.",
    long_description=(read('README.md')),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
    license="MIT",
    keywords="edit3d improvedEdit3d",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    scripts=['edit3d/reconstruct_from_rgb.py', 'edit3d/reconstruct_from_sketch.py', 'edit3d/edit_via_scribble.py', 'edit3d/edit_via_sketch.py', 'edit3d/train.py']
)
