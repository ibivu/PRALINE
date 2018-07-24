from __future__ import division, absolute_import, print_function

import sys
import argparse
import os
import uuid
import shutil
import tarfile
import os.path
import atexit

import six.moves.urllib.parse

from praline import load_score_matrix, load_sequence_fasta, open_builtin
from praline import write_alignment_clustal, write_alignment_fasta
from praline.core import *
from praline.component import PralineMultipleSequenceAlignmentWorkflow
from praline.component import PairwiseAligner
from praline.container import ALPHABET_AA, TRACK_ID_INPUT
from praline.util import run, write_log_structure

ROOT_TAG = "__ROOT_TAG__"

def main():
    # Parse arguments.
    args = parse_args()
    verbose = not args.quiet or args.verbose

    # Setup the execution manager.
    index = TypeIndex()
    index.autoregister()
    if args.remote:
        if args.remote_secret is None:
            secret = "__MUCH_SECRITY__"
        else:
            with open(args.remote_secret, 'r') as f:
                secret = f.readline()

        manager = RemoteManager(index, args.remote_host, args.remote_port,
                                secret)
    elif args.num_threads > 1:
        manager = ParallelExecutionManager(index, args.num_threads - 1)
    else:
        manager = Manager(index)

    # Register manager cleanup code at exit.
    atexit.register(_atexit_close_manager, manager=manager)

    # Load inputs and other data.
    with open_resource(args.score_matrix, "matrices") as f:
        score_matrix = load_score_matrix(f, alphabet=ALPHABET_AA)
    seqs = load_sequence_fasta(args.input, ALPHABET_AA)
    gap_series = [-float(x) for x in args.gap_penalties.split(",")]

    # Setup environment.
    keys = {}
    keys['gap_series'] = gap_series
    keys['db_name'] = args.psi_blast_db
    keys['num_seqs'] = args.psi_blast_num
    keys['max_evalue'] = args.psi_blast_evalue
    keys['profile_evalue'] = args.psi_blast_inclusion
    keys['num_iterations'] = args.psi_blast_iters
    keys['score_threshold'] = args.preprofile_score
    keys['linkage_method'] = args.tree_linkage
    keys['waterman_eggert_iterations'] = args.num_preprofile_alignments
    keys['aligner'] = PairwiseAligner.tid
    keys['debug'] = args.debug
    if args.merge_semiglobal_auto:
        keys['merge_mode'] = 'semiglobal_auto'
    elif args.merge_semiglobal:
        keys['merge_mode'] = 'semiglobal'
    else:
        keys['merge_mode'] = 'global'

    if args.dist_semiglobal_auto:
        keys['dist_mode'] = 'semiglobal_auto'
    elif args.dist_semiglobal:
        keys['dist_mode'] = 'semiglobal'
    else:
        keys['dist_mode'] = 'global'

    if args.pregen_tree:
        keys['msa_mode'] = 'tree'
    else:
        keys['msa_mode'] = 'ad_hoc'

    if args.preprofile_global:
        keys['preprofile_mode'] = 'global'
    elif args.preprofile_local:
        keys['preprofile_mode'] = 'local'
    else:
        keys['preprofile_mode'] = 'dummy'

    if args.psi_blast:
        keys['run_psi_blast'] = True

    if args.no_accelerate:
        keys['accelerate'] = False
    else:
        keys['accelerate'] = True

    try:
        keys['blast_plus_root'] = os.environ['BLAST_PLUS_ROOT']
    except KeyError:
        pass

    env = Environment(keys=keys)

    # Initialize root node for output
    root_node = TaskNode(ROOT_TAG)

    # Run the PRALINE MSA workflow
    component = PralineMultipleSequenceAlignmentWorkflow
    execution = Execution(manager, ROOT_TAG)
    task = execution.add_task(component)
    task.inputs(sequences=seqs, score_matrix=score_matrix)
    task.environment(env)

    outputs = run(execution, verbose, root_node)[0]
    alignment = outputs['alignment']

    # Write alignment to output file.
    outfmt = args.output_format
    if outfmt == 'fasta':
        write_alignment_fasta(args.output, alignment, TRACK_ID_INPUT)
    elif outfmt == "clustal":
        write_alignment_clustal(args.output, alignment, TRACK_ID_INPUT,
                                score_matrix)
    else:
        raise DataError("unknown output format: '{0}'".format(outfmt))

    if verbose:
        sys.stdout.write('\n')

    # Collect log bundles
    if args.debug > 0:
        write_log_structure(root_node)


def _atexit_close_manager(manager):
    manager.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file in FASTA format")
    parser.add_argument("output", help="output alignment")
    parser.add_argument("-g", "--gap-penalties",
                        help="comma separated list of gap penaties",
                        default="11,1", dest="gap_penalties")
    parser.add_argument("-m", "--score-matrix",
                        help="score matrix to use for alignment",
                        default="blosum62", dest="score_matrix")
    parser.add_argument("-t", "--threads", help="number of threads to use",
                        default=1,
                        dest="num_threads", type=int)
    parser.add_argument("-s", "--preprofile-score", default=None,
                        dest="preprofile_score", type=float,
                        help="exclude preprofile alignments by score")
    parser.add_argument("-f", "--output-format", default="fasta",
                        dest="output_format",
                        help="write the alignment in the specified format")
    parser.add_argument("--tree-linkage", default="average",
                        dest="tree_linkage",
                        help="use this linkage method when building tree")
    parser.add_argument("--no-accelerate", default=False,
                        dest="no_accelerate", action="store_true",
                        help="disable acceleration by C python extension")
    parser.add_argument("--preprofile-alignments", default=2,
                        dest="num_preprofile_alignments", type=int,
                        help="local preprofile alignments per sequence")
    parser.add_argument("--debug", "-d", action="count", dest="debug",
                        default=0, help="enable debugging output")


    group = parser.add_argument_group("PSI-BLAST")
    group.add_argument("--psi-blast",
                       help="use PSI-BLAST homologs in preprofile building",
                       dest="psi_blast", action="store_true", default=False)
    group.add_argument("--psi-blast-evalue",
                       help="e-value threshold for preprofile inclusion",
                       dest="psi_blast_evalue", type=float,
                       default=1.0)
    group.add_argument("--psi-blast-num",
                       help="top number of PSI-BLAST hits to use",
                       dest="psi_blast_num", type=int, default=10)
    group.add_argument("--psi-blast-db",
                       help="PSI-BLAST database name",
                       dest="psi_blast_db", default="nr")
    group.add_argument("--psi-blast-iters",
                       help="number of PSI-BLAST iterations to run",
                       dest="psi_blast_iters", type=int, default=3)
    group.add_argument("--psi-blast-inclusion",
                       help="e-value threshold for PSI-BLAST inclusion",
                       dest="psi_blast_inclusion", type=float,
                       default=0.002)

    group = parser.add_argument_group("Remote execution")
    group.add_argument("--remote",
                       help="execute this job on a pralined instance",
                       dest="remote", action="store_true", default=False)
    group.add_argument("--remote-host",
                       help="hostname of pralined instance",
                       dest="remote_host", type=str, default="127.0.0.1")
    group.add_argument("--remote-port",
                       help="port of pralined instance",
                       dest="remote_port", type=int, default=9000)
    parser.add_argument("--remote-secret", dest="remote_secret", default=None,
                        help="file with secret value shared with prtalined "
                             "instance to sign and validate messages with")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preprofile-none", help="build no preprofiles",
                       action="store_true", dest="preprofile_none",
                       default=False)
    group.add_argument("--preprofile-local", help="build local preprofiles",
                       action="store_true", dest="preprofile_local",
                       default=False)
    group.add_argument("--preprofile-global", help="build global preprofiles",
                       action="store_true", dest="preprofile_global",
                       default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--msa-tree', dest="pregen_tree",
                        action="store_true", default=False,
                        help="do pre-generated tree mutliple alignment")
    group.add_argument('--msa-adhoc',  dest="adhoc_tree",
                        action="store_true", default=True,
                        help="do ad-hoc tree multiple alignment")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--merge-global', dest="merge_global",
                        action="store_true", default=True,
                        help="merge with a global alignment")
    group.add_argument('--merge-semiglobal',  dest="merge_semiglobal",
                        action="store_true", default=False,
                        help="merge with a semiglobal alignment")
    group.add_argument('--merge-semiglobal-auto',  dest="merge_semiglobal_auto",
                        action="store_true", default=False,
                        help="merge with free end/start gaps for the shortest "
                             "sequence")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dist-global', dest="dist_global",
                        action="store_true", default=True,
                        help="dist with a global alignment")
    group.add_argument('--dist-semiglobal',  dest="dist_semiglobal",
                        action="store_true", default=False,
                        help="dist with a semiglobal alignment")
    group.add_argument('--dist-semiglobal-auto',  dest="dist_semiglobal_auto",
                        action="store_true", default=False,
                        help="dist with free end/start gaps for the shortest "
                             "sequence")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", dest="verbose", help="be verbose",
                        action="store_true", default=False)
    group.add_argument("-q", "--quiet", dest="quiet", help="be quiet",
                       action="store_true", default=True)

    return parser.parse_args()


def open_resource(filename, prefix):
    try:
        return open(filename)
    except IOError:
        return open_builtin(os.path.join(prefix, filename))
