"""Install in-memory TEST-ONLY stand-ins for absent optional packages.

The full project requires its declared dependencies at runtime. These shims
allow unit tests to verify control flow in the sandbox without package access.
"""
import sys
import types

try:
    from diskcache import Cache
except ImportError:
    m = types.ModuleType('diskcache')
    class Cache:
        def __init__(self, path):
            self.store = {}
        def get(self, key):
            return self.store.get(key)
        def set(self, key, value, expire=None):
            self.store[key] = value
    m.Cache = Cache
    sys.modules['diskcache'] = m

try:
    import tenacity
except ImportError:
    m = types.ModuleType('tenacity')
    m.retry = lambda *a, **kw: (lambda f: f)
    m.stop_after_attempt = lambda n: None
    m.wait_exponential = lambda *a, **kw: None
    sys.modules['tenacity'] = m

try:
    from openai import OpenAI
except ImportError:
    m = types.ModuleType('openai')
    class OpenAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('Install openai to run the real model; tests must mock calls')
    m.OpenAI = OpenAI
    sys.modules['openai'] = m
