#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

from setuptools import setup, find_packages, Extension
import numpy as np

entry_points = []
entry_points.append("PairwiseAligner = praline.component:PairwiseAligner")
entry_points.append("DummyMasterSlaveAligner = praline.component:DummyMasterSlaveAligner")
entry_points.append("GlobalMasterSlaveAligner = praline.component:GlobalMasterSlaveAligner")
entry_points.append("LocalMasterSlaveAligner = praline.component:LocalMasterSlaveAligner")
entry_points.append("AdHocMultipleSequenceAligner = praline.component:AdHocMultipleSequenceAligner")
entry_points.append("TreeMultipleSequenceAligner = praline.component:TreeMultipleSequenceAligner")
entry_points.append("ProfileBuilder = praline.component:ProfileBuilder")
entry_points.append("GuideTreeBuilder = praline.component:GuideTreeBuilder")
entry_points.append("BlastPlusSequenceFinder = praline.component:BlastPlusSequenceFinder")
entry_points.append("PsiBlastPlusSequenceFinder = praline.component:PsiBlastPlusSequenceFinder")
entry_points.append("RawPairwiseAligner = praline.component:RawPairwiseAligner")
entry_points.append("PralineMultipleSequenceAlignmentWorkflow = praline.component:PralineMultipleSequenceAlignmentWorkflow")

console_scripts = []
console_scripts.append("praline = praline.cmd:main")
console_scripts.append("pralined = praline.daemon:main")

ext_calign = Extension("praline.util.cext", ["praline/util/cext.c"],
                       include_dirs=[np.get_include()],
                       extra_compile_args=["-std=c99", "-ffast-math", "-O3"])

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="praline-aln",
    version="1.1.0",
    description="PRALINE sequence alignment toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Maurits Dijkstra",
    author_email="mauritsdijkstra@gmail.com",
    url="https://github.com/ibivu/PRALINE/",
    license="GPLv2",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6"
    ],

    ext_modules=[ext_calign],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.2.*,!=3.3.*,!=3.4.*',
    install_requires=["numpy>=1.6.1", "falcon>=0.3.0",
                      "itsdangerous>=0.24","six>=1.10.0"],
    packages=["praline", "praline.core", "praline.container",
              "praline.component", "praline.util"],
    package_data={
          "praline": ["matrices/*"]
    },
    entry_points={
      "praline.type": entry_points,
      "console_scripts": console_scripts
    }
)
