import time
from pinaka.utils import logger


def benchmark(func):
    def wrapped(*args, **kwargs):
        begin = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        logger.debug("Runtime:"+func.__name__+":"+str(end-begin)+" sec")

        return result
    return wrapped
