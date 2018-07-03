#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

from setuptools import setup, find_packages, Extension
import numpy as np

entry_points = []
entry_points.append('PairwiseAligner = praline.component:PairwiseAligner')
entry_points.append('DummyMasterSlaveAligner = praline.component:DummyMasterSlaveAligner')
entry_points.append('GlobalMasterSlaveAligner = praline.component:GlobalMasterSlaveAligner')
entry_points.append('LocalMasterSlaveAligner = praline.component:LocalMasterSlaveAligner')
entry_points.append('AdHocMultipleSequenceAligner = praline.component:AdHocMultipleSequenceAligner')
entry_points.append('TreeMultipleSequenceAligner = praline.component:TreeMultipleSequenceAligner')
entry_points.append('ProfileBuilder = praline.component:ProfileBuilder')
entry_points.append('GuideTreeBuilder = praline.component:GuideTreeBuilder')
entry_points.append('BlastPlusSequenceFinder = praline.component:BlastPlusSequenceFinder')
entry_points.append('PsiBlastPlusSequenceFinder = praline.component:PsiBlastPlusSequenceFinder')
entry_points.append('LegacyPairwiseAligner = praline.component:LegacyPairwiseAligner')
entry_points.append('RawPairwiseAligner = praline.component:RawPairwiseAligner')
entry_points.append('PralineMultipleSequenceAlignmentWorkflow = praline.component:PralineMultipleSequenceAlignmentWorkflow')

console_scripts = []
console_scripts.append('praline = praline.cmd:main')
console_scripts.append('pralined = praline.daemon:main')

ext_calign = Extension("praline.util.cext", ["praline/util/cext.c"],
                       include_dirs=[np.get_include()],
                       extra_compile_args=["-std=c99", "-ffast-math", "-O3"])

setup(name='PRALINE',
      version = '1.1',
      description = 'PRALINE sequence alignment toolkit',
      author = 'Maurits Dijkstra',
      author_email = 'mauritsdijkstra@gmail.com',
      url = 'http://www.few.vu.nl/',
      license = "GPL",

      ext_modules = [ext_calign],
      install_requires = ['numpy>=1.6.1', 'falcon>=0.3.0',
                          'itsdangerous>=0.24','six>=1.10.0'],
      packages = ['praline', 'praline.core', 'praline.container',
                  'praline.component', 'praline.util'],
      package_data = {
              'praline': ['matrices/*']
          },
      entry_points = {
          'praline.type': entry_points,
          'console_scripts': console_scripts
      }

    )
