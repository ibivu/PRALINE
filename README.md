# PRALINE

Reimplementation of the PRALINE multiple sequence alignment program.

# Features

* Progressive multiple sequence alignment
* Tree generation through hierarchical clustering and on-the-fly (PRALINE-style)
* Profile-profile pairwise alignment
* Affine and linear gap penalties supported
* Semi-global, global alignment for the MSA merge step
* Local, global and semi-global preprofile generation
* Improved preprofile sampling through PSI-BLAST search (requires local installation of NCBI BLAST+)
* Nucleotide and amino acid alphabets supported out of the box
* Easy extensible to arbitrary alphabets (for example secondary sequence)
* Packaged with common substitution matrices (BLOSUM), can be provided to the program.

# Coming soon

* HMM-like gap penalties per position
* Heuristics to speed up tree building for large numbers of sequences
* Better multithreading for improved scaling on many-core systems

# Installing PRALINE

## Requirements

* Python 2.7 / Python 3.6 (earlier 3.x versions may also work, but have not been tested)
* A C compiler (C99 support required)
* NCBI BLAST+ (optional, for homology searching using PSI-BLAST)

## Instructions

You can install PRALINE by cloning this repository and running (in a shell):

`python setup.py install`

A version will be uploaded to PyPI in the near future, which should make PRALINE
installable with:

`pip install praline`
