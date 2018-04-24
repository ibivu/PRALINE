class PralineError(Exception):
    pass


class AlphabetError(PralineError):
    pass


class SequenceError(PralineError):
    pass


class SignatureError(PralineError):
	pass


class ComponentError(PralineError):
	pass


class MessageError(PralineError):
	pass


class DataError(PralineError):
	pass


class EnvironmentError(PralineError):
	pass


class RemoteError(PralineError):
	pass


class LogError(PralineError):
	pass