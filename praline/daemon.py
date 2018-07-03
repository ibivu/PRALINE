from __future__ import division, absolute_import, print_function

from wsgiref.simple_server import make_server
import argparse
import json
import threading
from uuid import uuid4 as uuid
import sys

try:
    import six.moves.cPickle as pickle
except ImportError:
    import pickle
import six.moves.queue as queue
import falcon
import itsdangerous

from praline.core import *

def _generate_job_id():
    """Generate a random job ID.

    :returns: a string containing a random job id
    """
    return uuid().hex

class RequirePickle(object):
    """WSGI middleware requiring posted data to be of the pickle MIME type.

    """
    def process_request(self, req, resp):
        if not req.client_accepts('application/python-pickle'):
            raise falcon.HTTPNotAcceptable(
                'This API only supports responses encoded as pickle.')

        if req.method in ('POST', 'PUT'):
            if 'application/python-pickle' not in req.content_type:
                raise falcon.HTTPUnsupportedMediaType(
                    'This API only supports requests encoded as pickle.')

class SignedPickleTranslator(object):
    """WSGI middleware translating request and responses from/to itsdangerous-signed
    pickled format.

    """
    def __init__(self, secret):
        self._serializer = itsdangerous.Serializer(secret, serializer=pickle)

    def process_request(self, req, resp):
        # req.stream corresponds to the WSGI wsgi.input environ variable,
        # and allows you to read bytes from the request body.
        #
        # See also: PEP 3333
        if req.content_length in (None, 0):
            # Nothing to do
            return

        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body',
                                        'A valid pickle document is required.')

        try:
            req.context['args'] = self._serializer.loads(body)

        except (pickle.PickleError, IndexError):
            raise falcon.HTTPError(falcon.HTTP_753,
                                   'Malformed pickle',
                                   'Could not decode the request body. The '
                                   'pickled string was incorrect.')
        except itsdangerous.BadSignature:
            raise falcon.HTTPError(falcon.HTTP_403, "Invalid signature",
                                   "The signature used to sign the request "
                                   "args was not correct.")

    def process_response(self, req, resp, resource):
        if 'result' not in req.context:
            return

        resp.content_type = "application/python-pickle"
        resp.body = self._serializer.dumps(req.context['result'])

class RootResource(object):
    """Root API resource class. Does nothing but scare people away.

    """
    def on_get(self, req, resp):
        msg = {"msg": "Nothing to see here, move along please!"}

        req.context['result'] = msg


class JobListResource(object):
    """Job list API resource class. Allows a client to list all jobs and post
    new jobs.

    :param job_list: JobList instance from which to pull job information
    :param job_queue: Queue instance used to submit jobs to the worker thread
    """
    def __init__(self, job_list, job_queue):
        self._job_list = job_list
        self._job_queue = job_queue

    def on_get(self, req, resp):
        rows = []
        with self._job_list.lock:
            for row in self._job_list.list_jobs():
                job_id, state, data = row

                rows.append((job_id, state))

        req.context['result'] = rows
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        tid, tag, parent_tag, inputs, env = req.context["args"]

        job_id = _generate_job_id()
        state = "waiting"
        data = None

        with self._job_list.lock:
            self._job_list.set_job(job_id, state, data)

        self._job_queue.put((job_id, tid, tag, parent_tag, inputs, env))

        req.context['result'] = {'job_id': job_id, 'msg': 'Job created!'}
        resp.status = falcon.HTTP_201


class JobResource(object):
    """Job detail API resource class. Allows a client to request details about
    individual jobs such as the state, the messages it produced and any other
    information.

    :param job_list: JobList instance from which to pull job information
    """
    def __init__(self, job_list):
        self._job_list = job_list

    def on_get(self, req, resp, job_id, start_at):
        with self._job_list.lock:
            job_id, state, data = self._job_list.get_job(job_id)

        start_at = int(start_at)

        with self._job_list.message_lock:
            messages, next_start_at = self._job_list.get_messages(job_id,
                                                                  start_at)



        result = {'job_id': job_id, 'state': state, 'messages': messages,
                  'next_start_at': next_start_at, data: 'data'}
        req.context['result'] = result
        resp.status = falcon.HTTP_200


class JobRunnerThread(threading.Thread):
    """A Thread subclass which runs jobs submitted through the API. Note that
    this is not for parallelism, since the GIL prevents this, but to prevent the
    execution of jobs from blocking API requests.

    :param job_queue: Queue instance used to submit jobs to the worker thread
    :param job_list: JobList instance from which to pull job information
    """
    def __init__(self, job_queue, job_list):
        super(JobRunnerThread, self).__init__()

        self._job_queue = job_queue
        self._job_list = job_list

        self.daemon = True

    def run(self):
        index = TypeIndex()
        index.autoregister()
        manager = Manager(index)

        while True:
            job = self._job_queue.get()
            job_id, tid, tag, parent_tag, inputs, env = job

            try:
                component = index.resolve(tid)

                with self._job_list.lock:
                    self._job_list.set_job(job_id, 'running', None)

                request = tid, inputs, tag, env
                for msg in manager.execute_one(request, parent_tag):
                    with self._job_list.message_lock:
                        self._job_list.append_message(job_id, msg)

                with self._job_list.lock:
                    self._job_list.set_job(job_id, 'complete', None)
            except Exception as e:
                with self._job_list.lock:
                    self._job_list.set_job(job_id, 'error', str(e))


class JobList(object):
    """Simple class that stores the shared job list. Also includes various
    locks to synchronize access to the data between the API and runner threads.

    """
    def __init__(self):
        self._data = {}
        self._messages = {}

        self.lock = threading.Lock()
        self.message_lock = threading.Lock()

    def get_job(self, job_id):
        return self._data[job_id]

    def set_job(self, job_id, state, data):
        item = (job_id, state, data)

        self._data[job_id] = item

    def clear_job(self, job_id):
        del self._data[job_id]

    def clear_all_jobs(self):
        self._data.clear()

    def list_jobs(self):
        return list(self._data.values())

    def append_message(self, job_id, msg):
        try:
            self._messages[job_id].append(msg)
        except KeyError:
            self._messages[job_id] = [msg]

    def get_messages(self, job_id, start_at=0):
        try:
            next_start_at = len(self._messages[job_id])
            messages = self._messages[job_id][start_at:]

            return (messages, next_start_at)
        except KeyError:
            return ([], 0)


def main():
    args = parse_args()

    if args.secret is None:
        warn_msg = "WARNING: no secret provided! using insecure default!"
        print(warn_msg, file=sys.stderr)

        secret = "__MUCH_SECRITY__"
    else:
        with open(args.secret, 'r') as f:
            secret = f.readline()

    job_list = JobList()
    job_queue = queue.Queue()

    runner_thread = JobRunnerThread(job_queue, job_list)
    runner_thread.start()

    api = falcon.API(middleware=[
        RequirePickle(),
        SignedPickleTranslator(secret),
    ])
    api.add_route('/', RootResource())
    api.add_route('/jobs', JobListResource(job_list, job_queue))
    api.add_route('/jobs/{job_id}/{start_at}', JobResource(job_list))

    httpd = make_server(args.interface, args.port, api)
    httpd.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface",
                        help="bind to the interface given by IP",
                        default="127.0.0.1", dest="interface")
    parser.add_argument("-p", "--port", help="bind to the following PORT",
                        default=9000, dest="port", type=int)
    parser.add_argument("--secret", dest="secret", default=None,
                        help="file containing the secret to validate API calls "
                             "made by a client")

    return parser.parse_args()
