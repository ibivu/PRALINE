"""Manager-related classes and support methods.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""
from __future__ import division, absolute_import, print_function

from uuid import uuid4 as uuid
from multiprocessing import Queue, Process, cpu_count, BoundedSemaphore, Lock
from multiprocessing import Manager as MultiprocessingManager
from threading import Thread
import time
import pkg_resources

from six.moves.queue import Empty, Queue as TQueue
import six
from six.moves import range, zip
try:
    import six.moves.cPickle as pickle
except ImportError:
    import pickle
import six.moves.urllib.request, six.moves.urllib.error, six.moves.urllib.parse
import itsdangerous

from .exception import *
from .component import Component, Message, Container, T, ErrorMessage
from .component import MESSAGE_KIND_COMPLETE, MESSAGE_KIND_ERROR
from .component import MESSAGE_KIND_BEGIN, CompleteMessage
from .component import ALLOWED_PRIMITIVE_TYPES, BeginMessage

ENTRY_POINT_GROUP = 'praline.type'

class TypeIndex(object):
    """A TypeIndex object manages the mapping between type id strings
    and the classes associated with these. Whenever a component, manager or
    other object wants to look up a class it can do so though its type index.

    Mappings between type ids and types can be registered manually, but
    autoregistration through setuptools entry points is also a possibility.
    This mechanism also allows for third party packages to register their
    components into the PRALINE base system just by being installed by
    the user.

    """
    def __init__(self):
        self._types = {}


    def register(self, component_class):
        """Manually register a component class in this index. This is
        generally not recommended unless developing components or when
        interactively writing a small component to do some work.

        :param component_class: the component class to register

        """
        self._types[component_class.tid] = component_class

    def unregister(self, component_class):
        """Unregister a registered component class.

        :param component_class: the component class to unregister

        """
        try:
            del self._types[component_class.tid]
        except KeyError:
            s = "component with type id '{0}' not registered"
            s = s.format(tid)
            raise ComponentError(s)

    def autoregister(self):
        """Automatically register any component classes that have been
        registered to the PRALINE setuptools entry point. This allows
        registering all the built-in components with just one method call.
        Another benefit is that it allows third-party packages to
        automatically register their own components into the PRALINE system
        on installation by the user.

        """
        entrypoints = pkg_resources.iter_entry_points(group=ENTRY_POINT_GROUP)
        for entrypoint in entrypoints:
            # Grab the function that is the actual plugin.
            component_class = entrypoint.load() # Call the plugin
            self.register(component_class)

    def resolve(self, tid):
        """Resolve a type id string to a component class.

        :param tid: the type id to resolve into a class
        :returns: the class which this type id maps to

        """
        try:
            return self._types[tid]
        except KeyError:
            s = "component with type id '{0}' not registered"
            s = s.format(tid)
            raise ComponentError(s)


def _require_open_manager(func):
    """Decorator that checks whether a manager is open before calling through
    to the wrapped method.

    """
    def inner(self, *args, **kwargs):
        if not self.open:
            raise PralineError("manager has been closed")

        return func(self, *args, **kwargs)

    return inner

class Manager(object):
    """The Manager is the class responsible for coordinating component
    execution in the PRALINE system. The manager both instantiates components
    and translates execution requests into method calls to component. In
    addition to this role it also provides components with a type index and
    allows them to schedule subtasks for execution within the current context.

    This class is the most basic implementation, supporting only serial
    execution. Subclasses may offer additional functionality such as
    (but not limited to) parallel execution or transparent serialization /
    unserialization for execution on remote hosts.

    :param index: the type index that will be used to look up type ids
    """

    def __init__(self, index):
        self.index = index
        self.open = True

    @_require_open_manager
    def execute_one(self, request, parent_tag):
        """Execute a single execution request using this manager. Will yield
        any messages produced during execution.

        :param request: a tuple containing the type id, inputs, tag and
            environment
        :param parent_tag: a unique tag identifying the parents' execution
            context

        """
        if not self.open:
            raise PralineError("manager has been closed")

        tid, inputs, tag, env = request
        for message in self._invoke(tid, inputs, tag, env,
                                    parent_tag = parent_tag):
            yield message

    @_require_open_manager
    def execute_many(self, requests, parent_tag):
        """Execute multiple execution requests using this manager. For this
        manager, this is equivalent to calling the exection method for a
        single request in a loop. Will yield any messages produced during
        execution.

        :param requests: a list of tuples containing the type id, inputs, tag
            and environment
        :param parent_tag: a unique tag identifying the parents' execution
            context

        """
        for request in requests:
            tid, inputs, tag, env = request
            for message in self._invoke(tid, inputs, tag, env,
                                        parent_tag = parent_tag):
                yield message

    def _invoke(self, tid, inputs, tag, environment, submanager=None,
                parent_tag=None):
        """Helper method to do the heavy lifting of instantiating a
        component, checking the inputs, calling the execution
        method and finally checking and returning the outputs. Will
        yield any messages produced during execution.

        :param tid: the type id of the component to instantiate
        :param inputs: the inputs to pass to the component
        :param tag: a unique tag identifying this execution context
        :param environment: the environment for this component execution
        :param submanager: optional second manager for this component
            execution to schedule its subtasks on
        :param parent_tag: tag uniquely identifying the parent execution
            context

        """
        if submanager is None:
            submanager = self

        component_class = self.index.resolve(tid)
        component = component_class(submanager, environment, tag)

        for key, value in six.iteritems(component.options):
            _conforms_signature(value, environment[key])

        for name, port in six.iteritems(component.inputs):
            inputs[name] = inputs.get(name, None)

            if inputs[name] is None and not port.optional:
                s = "input '{0}' is not optional but was not supplied"
                s = s.format(name)
                raise DataError(s)
            elif not inputs[name] is None:
                _conforms_signature(port.signature, inputs[name])

        begin_message = BeginMessage(parent_tag)
        begin_message.tag = tag
        yield begin_message

        for message in component.execute(**inputs):
            if not isinstance(message, Message):
                s = "component messages should be subclasses of Message"
                raise TypeError(s)

            if message.tag is None:
                message.tag = tag

            if message.kind == MESSAGE_KIND_COMPLETE and message.tag == tag:
                for name, port in six.iteritems(component.outputs):
                    message.outputs[name] = message.outputs.get(name, None)
                    if message.outputs[name] is None and not port.optional:
                        s = "output '{0}' is not optional but was not supplied"
                        s = s.format(name)
                        raise DataError(s)
                    elif not message.outputs[name] is None:
                        signature = port.signature
                        output = message.outputs[name]
                        _conforms_signature(signature, output)

            yield message

    @_require_open_manager
    def close(self):
        """Close this manager, cleaning up any resources it might have
        allocated. After this method has been called any use of the manager
        will generate an exception.

        """
        self.open = False

WORKER_TYPE_THREAD = 0
WORKER_TYPE_PROCESS = 1

WORKER_BUFFER_INTERVAL = 0.25

def _worker(manager, batch_queue, out_queue, run_once = False):
    while True:
        batch, parent_tag = batch_queue.get()

        if batch is None:
            # Termination signal, time to quit.
            break

        last_flush = time.time()
        buf = []

        try:
            for tid, inputs, tag, env in batch:
                for msg in manager._invoke(tid, inputs, tag, env, manager,
                                               parent_tag):
                    buf.append(msg)

                    cur_time = time.time()
                    if (cur_time - last_flush) > WORKER_BUFFER_INTERVAL:
                        out_queue.put(buf)
                        buf = []
                        last_flush = cur_time

            # Flush any remaining messages to the queue on completion.
            out_queue.put(buf)
        except Exception as e:
            msg = ErrorMessage(str(e))
            msg.tag = tag
            out_queue.put([msg])
            break

        if run_once:
            break

class ParallelExecutionManager(Manager):
    """A basic extension of the standard manager which implements parallel
    execution using the Python multiprocessing API. Because this API
    runs tasks in separate worker processes this allows PRALINE to run on
    multiple cores and/or CPU's concurrently, bypassing the problems caused
    by the Python global interpreter lock.

    :param index: the type index that will be used to look up type ids
    :param concurrent_tasks: number of concurrent workers to use, if not
        provided the manager will try to guess based on the number of CPU's.

    """
    def __init__(self, index, concurrent_tasks=None):
        super(ParallelExecutionManager, self).__init__(index)

        if concurrent_tasks is None:
            concurrent_tasks = cpu_count()

        self.batch_queues = [Queue() for i in range(concurrent_tasks)]
        self.out_queues = [Queue() for i in range(concurrent_tasks)]
        self.locks = [Lock() for i in range(concurrent_tasks)]
        self.processes = []
        for i in range(concurrent_tasks):
            args = (self, self.batch_queues[i], self.out_queues[i])
            process = Process(target=_worker, args=args)
            process.daemon = True
            self.processes.append(process)
            process.start()


    def _schedule_requests(self, requests, parent_tag, running_map, job_map,
                           batch_size=10):
        # Oportunistically see whether we can schedule stuff in a worker
        # process. Try to grab any of the worker locks (non-blockingly)
        # and then dispatch a job to that worker if we succeed.
        workers_acquired = 0

        for i, lock in enumerate(self.locks):
            if not requests:
                # Our requests are exhausted, time to stop.
                break

            if lock.acquire(False):
                workers_acquired += 1

                # Generate a random ID for the batch. A UUID is a bit overkill
                # here (and slow), but hey, convenience!
                batch_id = uuid().hex
                batch = []
                try:
                    for n in range(batch_size):
                        batch.append(requests.pop())
                except IndexError as e:
                    pass

                # Update our running map with the lock so we can release it
                # later.
                out_queue = self.out_queues[i]
                running_map[batch_id] = (WORKER_TYPE_PROCESS, lock, out_queue)
                batch_tags = set(tag for tid, inputs, tag, env in batch)
                for batch_tag in batch_tags:
                    job_map[batch_tag] = (batch_id, batch_tags)

                # Push the job to the worker.
                batch_queue = self.batch_queues[i]
                batch_queue.put((batch, parent_tag))

        # Check if we already have a thread worker running.
        thread_worker_running = False
        for type_, lock, queue in six.itervalues(running_map):
            if type_ == WORKER_TYPE_THREAD:
                thread_worker_running = True

        # If we weren't able to grab any free worker processes and still
        # have requests pending then we need to run a thread worker. This is
        # because we might have worker processes which are waiting on the
        # output of a subtask. In such a case we will deadlock.
        if not thread_worker_running and requests and not workers_acquired:
            batch_queue = TQueue()
            out_queue = TQueue()

            # Generate a random ID for the batch. A UUID is a bit overkill
            # here (and slow), but hey, convenience!
            batch_id = uuid().hex
            batch = []
            try:
                for n in range(batch_size):
                    batch.append(requests.pop())
            except IndexError as e:
                pass

            running_map[batch_id] = (WORKER_TYPE_THREAD, None, out_queue)
            batch_tags = set(tag for tid, inputs, tag, env in batch)
            for batch_tag in batch_tags:
                job_map[batch_tag] = (batch_id, batch_tags)

            args = (self, batch_queue, out_queue, True)
            thread_worker = Thread(target=_worker, args=args)
            thread_worker.daemon = True
            batch_queue.put((batch, parent_tag))
            thread_worker.start()

    def _multiplex_queues(self, queues):
        got_buffer = False
        for queue in queues:
            try:
                buffer = queue.get(block=False)
                got_buffer = True
                for msg in buffer:
                    yield msg
            except Empty:
                pass

        if not got_buffer:
            # Block for a short while so this does not turn into a busy-wait
            # loop.
            time.sleep(0.1)


    @_require_open_manager
    def execute_many(self, requests, parent_tag):
        """The implementation of this method will spawn a fixed number of
        worker processes. All the execution requests are then processed
        in parallel using the worker processes, the amount of concurrent
        tasks at any time not exceeding the limit set when instantiating
        the manager.

        :param requests: a list of tuples containing the type id, inputs, tag
            and environment
        :param parent_tag: a unique tag identifying the parents' execution
            context

        """
        if len(requests) == 1:
            # We only have one job to run. In such a case it's a better idea to
            # run the job in the current thread as a regular Manager would, as
            # we don't need the parallelism anyway.
            for msg in self.execute_one(requests[0], parent_tag):
                yield msg
            return

        pending_requests = list(requests)
        running_map = {}
        job_map = {}
        while pending_requests or running_map:
            if pending_requests:
                self._schedule_requests(pending_requests, parent_tag,
                                        running_map, job_map)
                queues = [q for t, l, q in six.itervalues(running_map)]

            while True:
                should_reschedule = False

                for msg in self._multiplex_queues(queues):
                    if msg.kind == MESSAGE_KIND_COMPLETE and msg.tag in job_map:
                        batch_id, batch_tags = job_map[msg.tag]
                        batch_tags.remove(msg.tag)
                        del job_map[msg.tag]

                        if not batch_tags:
                            # If the completed job was ran in a worker process
                            # then we need to release its lock so other
                            # processes kan acquire it.
                            type_, lock, queue = running_map[batch_id]
                            if type_ == WORKER_TYPE_PROCESS:
                                lock.release()

                            #print "batch {0} done, completed in {1}".format(batch_id, type_)

                            # Remove the request from the running map, so
                            # _schedule_requests() will try to fill the opened
                            # slot.
                            del running_map[batch_id]

                            # Force a reschedule after all messages have been
                            # processed to see if we can schedule new tasks.
                            should_reschedule = True

                    yield msg

                if should_reschedule:
                    break

    @_require_open_manager
    def close(self):
        """Close this manager, sending termination messages to all of the
        running workers. After this method has been called any use of the
        manager will generate an exception.

        """
        super(ParallelExecutionManager, self).close()

        for batch_queue in self.batch_queues:
            # Push a termination message to all the workers. This will cause
            # them to break out of their main loop and end their processes.
            batch_queue.put((None, None))


class RemoteManager(Manager):
    """The remote manager transparently executes jobs on remote nodes running
    an instance of the PRALINE daemon. The interface used to communicate is
    based on HMAC-signed pickled Python objects sent and received over HTTP.
    Note that this means that a user can **execute arbitary code** on your
    worker nodes if the secret key is compromised. In setups where security is
    an issue you should always firewall your worker nodes from the internet and
    make sure that the secret key is not added to your SCM repository.

    :param index: the type index that will be used to look up type ids
    :param host: the hostname of the machine running the pralined instance
    :param port: the port on which the pralined instance is running remotely

    """
    def __init__(self, index, host, port, secret):
        super(RemoteManager, self).__init__(index)

        self.host = host
        self.port = port

        self._serializer = itsdangerous.Serializer(secret, serializer=pickle)

    def execute_one(self, request, parent_tag):
        """Execute a single execution request using this manager. Will yield
        any messages produced during execution.

        :param request: a tuple containing the type id, inputs, tag and
            environment
        :param parent_tag: a unique tag identifying the parents' execution
            context

        """
        tid, inputs, tag, env = request

        data = self._serializer.dumps((tid, tag, parent_tag, inputs, env))
        headers = {}
        headers['Content-Type'] = 'application/python-pickle'
        headers['Accept'] = 'application/python-pickle'
        url = "http://{0}:{1}/jobs/".format(self.host, self.port)
        req = six.moves.urllib.request.Request(url, data, headers)
        res = self._serializer.loads(six.moves.urllib.request.urlopen(req).read())
        job_id = res["job_id"]

        start_at = 0
        while True:
            res = self._poll_status(job_id, start_at)

            start_at = res["next_start_at"]

            for msg in res["messages"]:
                yield msg

            if res['state'] == 'complete':
                break
            elif res['state'] == 'error':
                raise RemoteError(res['data'])

            time.sleep(1)

    def execute_many(self, requests, parent_tag):
        """Execute multiple execution requests using this manager. For this
        manager, this is equivalent to calling the exection method for a
        single request in a loop. Will yield any messages produced during
        execution.

        :param requests: a list of tuples containing the type id, inputs, tag
            and environment
        :param parent_tag: a unique tag identifying the parents' execution
            context

        """
        for request in requests:
            for msg in self.execute_one(request, parent_tag):
                yield msg

    def _poll_status(self, job_id, start_at):
        fmt = "http://{0}:{1}/jobs/{2}/{3}"
        url = fmt.format(self.host, self.port, job_id, start_at)
        headers = {}
        headers['Accept'] = 'application/python-pickle'

        req = six.moves.urllib.request.Request(url, None, headers)
        res = six.moves.urllib.request.urlopen(req).read()

        return pickle.loads(res)

def _conforms_signature(sig, data):
    """A helper method to check whether input or output data for a
    component conforms to the input or output signature specified by the
    component. If this is found not to be the case an exception is raised.

    :param sig: the signature to check against
    :param data: the input or output data to check against the signature

    """
    data_type=type(data)
    sig_type=type(sig)

    if data_type == list:
        if sig_type != list:
            s = "signature specifies list while data has non-list"
            raise DataError(s)
        for item in data:
            _conforms_signature(sig[0], item)
    elif data_type == tuple:
        if sig_type != tuple:
            s = "signature specifies tuple while data has non-tuple"
            raise DataError(s)
        if len(sig)!=len(data):
            s = "signature tuple of len {0}, data tuple of len {1}"
            s = s.format(len(sig), len(data))
            raise DataError(s)
        for data_item, sig_item in zip(data, sig):
            _conforms_signature(sig_item, data_item)
    elif data is None:
        if sig_type != T or not sig.nullable:
            s = "data is null but signature specifies not nullable"
            raise DataError(s)
    elif sig_type == T:
        _conforms_signature(sig.tid, data)
    else:
        if not (isinstance(data, Container) or data_type in ALLOWED_PRIMITIVE_TYPES):
            s = "data object is not of a container or an allowed primitive type"
            raise DataError(s)
        if not (sig_type in {str, T} or sig in ALLOWED_PRIMITIVE_TYPES):
            s = "data object {0}, signature specifies list/tuple"
            s = s.format(data.tid)
            raise DataError(s)
        if sig in ALLOWED_PRIMITIVE_TYPES and data_type != sig:
            s = "signature specifies primitive {0} but data is {1}"
            s = s.format(sig, data_type)
            raise DataError(s)
        if sig_type == str and data.tid != sig:
            s = "data container {0} unequal to signature container {1}"
            s = s.format(data.tid, sig)
            raise DataError(s)
