"""Support code for command-line tools running PRALINE workflows.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""

from __future__ import division, absolute_import, print_function

import sys
import os
import uuid
import tarfile

import six
import six.moves.urllib.parse


_LINE_FMT = "{0}{1:<{width}.{width}} {2:<5.5}   [{3:<35}] {4:>6.1%}\n"


def run(execution, verbose, root_node):
    initial = True
    for message in execution.run():
        if verbose:
            if not initial:
                fmt = "{0}\r\x1b[A" * (root_node.num_leaves())
                sys.stdout.write(fmt.format(" " * 80))
            else:
                initial = False

        root_node.update(message)

        if verbose:
            sys.stdout.write(serialize(root_node))


    return execution.outputs


def write_log_structure(node, path=None):
    if path is None:
        path = "Debug#{0}".format(uuid.uuid4().hex)
    else:
        path = os.path.join(path, node.tag)

    os.mkdir(path)

    if node.log_url is not None:
        unpack_log_bundle(node.log_url, path)

    for child in node.children.values():
        write_log_structure(child, path=path)


def unpack_log_bundle(url, path):
    u = six.moves.urllib.parse.urlparse(url)
    if u.scheme == 'file':
        if not tarfile.is_tarfile(u.path):
            return
        tf = tarfile.open(u.path)
        tf.extractall(path)
        tf.close()

        os.remove(u.path)


def format_line(node, level):
    components = node.tag.split('#')
    name = components[0]
    if len(components) > 1:
        u = components[1]
    else:
        u = ""

    indent = level * "  "
    hashes = "#" * int(round(35.0 * node.progress))

    return _LINE_FMT.format(indent, name, u, hashes, node.progress,
                      width=20-(level*2))


def serialize(node, level=-1):
    if level >= 0:
        s = format_line(node, level=level)
    else:
        s = ""
    for child in six.itervalues(node.children):
        if not child.complete:
            s += serialize(child, level=level + 1)
    return s
