"""Logging support for components. This module allows components to log
data that is non-essential in the sense that the alignment process does
not require it to be able to continue.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

import six.moves.urllib.parse
import six.moves.urllib.request, six.moves.urllib.parse, six.moves.urllib.error

import tempfile
import os.path
import shutil
import uuid


from .exception import *

ROOT_LOG_NAME = "component.log"

class LogBundle(object):
    """A log bundle is essentially a packaged directory of log files.
    Client components can request both open file objects or pathnames.
    The latter is convenient when you want to capture the output of
    an external program to a file. When the logging has completed
    the log bundle can be packaged, which transforms it into a single
    archive file. This archive file can then be presented to the system
    in the form of a URL. If instructed to do so by the user, the system
    will download and unpack the log bundle for the user to view.

    """
    def __init__(self):
        self._gone = False
        self._basedir = tempfile.mkdtemp()
        self._files = []
        self._message_files = {}

    def _check_gone(self):
        """Helper method to check whether this log bundle has been
        deleted. Because we can't force an object to be deleted we need
        some way to prevent badly written code from performing operations
        on a log bundle which is no longer associated with an on-disk
        structure.

        """
        if self._gone:
            s = "attempted to perform operations on a LogBundle " \
                "on which delete() has been called"
            raise LogError(s)

    def path(self, filename):
        """Return a path within the bundle. This is mainly useful for
        capturing the output of external programs that can't be
        redirected to write to a file object.

        :param filename: the filename to construct a bundle path from
        :returns: a bundle path pointing to the file with the given filename

        """
        self._check_gone()

        return os.path.join(self._basedir, filename)

    def file(self, filename, mode='w'):
        """Returns an open file object pointing to the path given by a
        filename inside the bundle. This can be used to write logging
        information directly from Python code (generally from within a
        component) or by redirecting the standard output of an external
        program.

        :param filename: filename of the file to open within the bundle
        :param mode: which mode the file should be opened in
        :returns: a file object pointing to the given file in the bundle

        """
        self._check_gone()

        f = open(self.path(filename), mode)
        self._files.append(f)

        return f

    def message(self, filename, message):
        """Writes a log message to the file at the given path inside
        the log bundle. The file is only opened once and then kept open
        until the log bundle is closed or otherwise destroyed.

        :param filename: name of the file to which the message should
                         be written
        :param message: the message to write, a newline will automatically
                        be appended

        """
        self._check_gone()

        if not filename in self._message_files:
            self._message_files[filename] = self.file(filename, mode='a')
            self._files.append(self._message_files[filename])

        f = self._message_files[filename]

        f.write("{0}\n".format(message))

    def close(self):
        """Closes the bundle. This calls close() on all the files which
        have been opened for logging. This is mainly useful if you want to
        close any files that may still be open for logging without having
        to worry about the lifecycle of each and every one.

        """
        self._check_gone()

        for f in self._files:
            f.close()

        self._files = []
        self._message_files = {}

    def delete(self):
        """Deletes the on-disk directory structure associated with this
        bundle. First calls close() to close any bundle files that may still
        be open. After calling this method no further operations may be
        performed on this bundle. Doing so will cause an exception to be
        raised.

        """
        if not self._gone:
            self.close()
            shutil.rmtree(self._basedir)
            self._gone = True

    def flush(self):
        """Flushes all the open files to disk. This is called by archive() to
        write any unwritten data before the archive is created.

        """
        self._check_gone()

        for f in self._files:
            f.flush()

    def archive(self):
        """Packages the bundle as an archive ready to send to a client.
        Note that this does not delete the bundle data. It it thus possible
        to archive a bundle many times.

        :returns: the absolute filename of a temporary file containing the
            bundle archive
        """

        self._check_gone()

        self.flush()

        tmp_dir = tempfile.gettempdir()
        u = uuid.uuid4().hex
        tmp_name = os.path.join(tmp_dir, u)

        return shutil.make_archive(tmp_name, 'gztar', self._basedir)

def path_to_url(path):
    return six.moves.urllib.parse.urljoin(
      'file:', six.moves.urllib.request.pathname2url(path))
