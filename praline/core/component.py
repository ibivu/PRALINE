"""Core component-related classes and methods.

.. moduleauthor:: Maurits Dijkstra <mauritsdijkstra@gmail.com>

"""

from __future__ import division, absolute_import, print_function

import six

from .exception import *


MESSAGE_KIND_BEGIN = "begin"
MESSAGE_KIND_PROGRESS = "progress"
MESSAGE_KIND_COMPLETE = "complete"
MESSAGE_KIND_ERROR = "error"
MESSAGE_KIND_LOG = "log"

ALLOWED_PRIMITIVE_TYPES = {int, float, str, bool, six.text_type}

class Component(object):
    """Component superclass. You should never instantiate this directly.

    :param manager: a manager instance provided to the component to run
        subcomponents
    :param environment: the configuration environment for this component
        execution
    :param tag: a string which uniquely identifies this component within
        an execution tree

    """

    tid = "praline.component.Component"
    inputs = {}
    outputs = {}

    def __init__(self, manager, environment, tag):
        for port in six.itervalues(self.inputs):
            _valid_signature(port.signature)
        for port in six.itervalues(self.outputs):
            _valid_signature(port.signature)
        for signature in six.itervalues(self.options):
            _valid_signature(signature)

        self.manager = manager
        self.environment = environment
        self.tag = tag

    def execute(self, **kwargs):
        """This method is called by the manager when the component is
        executed. Typically, a subclass would override this method to
        provide the component behavior.

        """
        s = "please override execute() in your Component subclass"
        raise NotImplementedError(s)


class Container(object):
    """Container supertype. All the non-primitive datatypes that are
    passed around as inputs to or outputs from components should subclass
    this class. This class provides a common interface for things like
    flattening and unflattening the object for serialization.

    """
    tid = "praline.container.Container"

    def flatten(self, os=None):
        """Flatten this container object. This means that any references
        that this object holds to other objects should be added to a dict
        and returned.

        :param os: if supplied, a dictionary with a partially completed set
            of objects
        :returns: a dictionary containing the flattened represenation of this
            object and any objects it contains references to

        """
        s = "please provide a flatten() implementation in your Container" \
            "subclass"
        raise NotImplementedError(s)

    @classmethod
    def unflatten(self, d):
        """The inverse operation of flatten. This is provided a dictionary
        of object representations and is expected to return a recreation
        of the object.

        :params d: a dictionary with flattened information to recreate
            this object
        :returns: an unflattened recreation of the object

        """
        s = "please provide an unflatten() implementation in your Container" \
            "subclass"
        raise NotImplementedError(s)

class Environment(Container):
    """An Environment container object. This object can contain key-value
    pairs that specify options. The environment contains functionality
    to automatically inherit option values. Options are inherited in the
    following order, with sources at the bottom having the highest
    precedence.
    * Defaults specified by the current component.
    * The options specified in the parent environment.
    * The options specified directly when creating this environment.

    The support includes recursively inheriting from any subenvironments that
    may be present in this enviroment. As an example, say there is an
    environment which has a subenvironment assigned to key *x*. This
    subenvironment in turn has key *y*. Furthermore, when creating
    this environment a parent environment is supplied for inheritance which
    also has key *x*, but with a subenvironment containing key *z*. If the
    resulting inheritance is collapsed into a new environment, then it will
    contain a key *x* with a subenvironment. This subenvironment, however,
    will contain both the key *y* that was directly supplied, as well as the
    key *z* that was recursively inherited from the parents' copy of *x*.

    :param keys: a dict of key-value pairs for this environment
    :param component: a component to inherit default option values from
    :param parent: a parent environment to inherit option values from
    """

    tid = "praline.container.Environment"

    def __init__(self, keys=None, component=None, parent=None):
        sources = []
        if component:
            sources.append(component.defaults)
        if parent:
            sources.append(parent.keys)
        if keys:
            sources.append(keys)
        self.keys = self._inherit_keys(sources)

    def collapse(self, component, env):
        """Collapse the environments' inheritance chain to create a new
        environment.

        :params component: a component to inherit default values from
        :params env: an environment serving as the child environment, this
            environment servering as the parent
        :returns: a new environment containing the collapsed keys according
            to the inheritance chain

        """
        k = env.keys
        return Environment(keys=k, component=component, parent=self)

    def _inherit_keys(self, sources):
        """Helper method to do the actual recursive inheritance.
        :params sources: a list of dicts with options to inherit, with the
            position in the list specifying their inheritance precedence,
            dicts at the end having the highest precedence
        :returns: a dict with the collapsed options

        """
        d = {}

        for source in sources:
            for key, value in six.iteritems(source):
                d[key] = value

        for key, value in six.iteritems(d):
            if isinstance(value, Environment):
                envsources = []
                for source in sources:
                    if key in source and isinstance(source[key], Environment):
                        envsources.append(source[key].keys)
                keys = self._inherit_keys(envsources)
                d[key] = Environment(keys)

        return d

    def __getitem__(self, key):
        return self.keys[key]

    def flatten(self, os=None):
        if os is None:
            os = {}

        d = {}
        d['items'] = []
        for key, value in six.iteritems(self.keys):
            d['items'].append((key, value))
            if isinstance(value, Container):
                if not value in os:
                    os[value] = value.flatten()

        os[self] = d

        return os

    @classmethod
    def unflatten(cls, d):
        keys = {}
        for key, value in d['items']:
            keys[key] = value

        return cls(keys)


class Port(object):
    """A small wrapper object specifying a component input/output port. This
    stores nothing more than a type signature and whether the input or
    output is optional or not.

    :param signature: the type signature for this I/O port
    :param optional: whether the I/O port is optional or not
    """

    def __init__(self, signature, optional=False):
        self.signature = signature
        self.optional = optional


class Message(object):
    """Message base class. Messages are how components signal the manager
    (and user code) that a particular event has occurred. Messages also
    store data, such as output data when a component has completed
    running. Messages are always 'tagged', which means that they contain
    a *tag* string property which uniquely identifies the component execution
    context which generated them.

    This class should not be instantiated directly, but rather
    should be used as a superclass if you want to implement new message
    types.

    :param kind: a string describing what kind of message this is

    """

    def __init__(self, kind):
        self.kind = kind
        self.tag = None

    def __repr__(self):
        fmt = "<{0} tag='{1}'>"

        return fmt.format(type(self).__name__, self.tag)


class BeginMessage(Message):
    """Message used to tell the system that a component has started
    executing. The tag of the parent component can be obtained through
    this message, which allows the system to construct an execution graph.

    :param parent_tag: a string uniquely identifying the execution context
        of the parent component

    """
    def __init__(self, parent_tag=None):
        super(BeginMessage, self).__init__(MESSAGE_KIND_BEGIN)

        self.parent_tag = parent_tag

    def __repr__(self):
        fmt = "<BeginMessage tag='{0}' parent_tag='{1}'>"

        return fmt.format(self.tag, self.parent_tag)

class ProgressMessage(Message):
    """Progress message used to signify progress of the execution of a
    component. The progress itself is a float between 0.0 and 1.0.

    :param progress: a float containing the progress

    """
    def __init__(self, progress):
        super(ProgressMessage, self).__init__(MESSAGE_KIND_PROGRESS)

        if progress > 1.0 or progress < 0.0:
            s="progress should be a float value between 0.0 and 1.0"
            raise MessageError(s)

        self.progress = progress

    def __repr__(self):
        fmt = "<ProgressMessage tag='{0}' progress={1:.2%}>"

        return fmt.format(self.tag, self.progress)


class CompleteMessage(Message):
    """A message used to signal to the system that the component execution
    has successfully concluded. This message also contains the outputs that
    were generated by the component execution.

    :param outputs: a dict containing component execution outputs
    """

    def __init__(self, outputs):
        super(CompleteMessage, self).__init__(MESSAGE_KIND_COMPLETE)

        self.outputs = outputs

class ErrorMessage(Message):
    """A message used to signal an error condition. If this is sent it will
    usually result in the system throwing an exception with the error string
    supplied in this message as explanation.

    :param error: a string describing the error that has occurred

    """
    def __init__(self, error):
        super(ErrorMessage, self).__init__(MESSAGE_KIND_ERROR)

        self.error = error

class LogMessage(Message):
    """A message used to signal that there is a log bundle available for
    consumption by the system. Log bundles are archive files of
    non-essential output generated by component execution. These files
    may be useful when an in-depth look at system behavior is required.

    This message has an *url* property containing an URL that points to the
    location of the log bundle.

    :param url: a string containing the URL where the log bundle can be found

    """
    def __init__(self, url):
        super(LogMessage, self).__init__(MESSAGE_KIND_LOG)

        self.url = url

class T(object):
    """A simple wrapper class to specify nullable objects in type signatures.
    This is nothing more than a type id and a boolean flag that specifies
    whether this object can be null (None) or not.

    :param tid: string containing the type id of the container type
    :param nullable: bool describing whether this object reference can be
        null/None or not

    """

    def __init__(self, tid, nullable=False):
        self.tid = tid
        self.nullable = nullable

def _valid_signature(sig):
    """Helper method to check whether a type signature is well-formed. This
    doesn't check whether a set of inputs or outputs actually conforms
    to a type signature, just whether it obeys the syntactic rules we have
    defined. If the signature is found to be invalid, a fatal exception will
    be thrown, so use this method with care.

    :param sig: the type signature to verify the validity of
    """
    sig_type = type(sig)
    if sig_type == list:
        if len(sig) != 1 :
            s = "lists in signatures must contain exactly one item"
            raise SignatureError(s)
        _valid_signature(sig[0])
    elif sig_type == tuple:
        if len(sig) == 0:
            s = "tuples in signatures must be non-empty"
            raise SignatureError(s)
        for item in sig:
            _valid_signature(item)
    elif sig_type == str:
        return
    elif sig_type == T:
        _valid_signature(sig.tid)
    elif sig in ALLOWED_PRIMITIVE_TYPES:
        return
    else:
        s = "type invalid for signature: {0}".format(sig_type)
        raise SignatureError(s)
