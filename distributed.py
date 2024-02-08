from abc import ABC, abstractmethod
from evaluations import EvaluationMethod
import multiprocessing as mp
import queue


class DistributedMethod(ABC):
    @abstractmethod
    def __init__(self, eval_method: EvaluationMethod):
        pass
    @abstractmethod
    def add_task(self, dna, policy_network_class):
        pass
    @abstractmethod
    def get_task_result(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def worker(dna, policy_network_class, evaluation_method):
  worker.queue.put(evaluation_method.eval(dna, policy_network_class))

def worker_initializer(queue):
    worker.queue = queue

class LocalSynchronous(DistributedMethod):
    def __init__(self, evaluation_method: EvaluationMethod):
        self.evaluation_method = evaluation_method
        self.queue = queue.Queue()
        worker_initializer(self.queue)

    def add_task(self, dna, policy_network_class):
        worker(dna, policy_network_class, self.evaluation_method)

    def get_task_result(self):
        return self.queue.get()


DistributedMethod.register(LocalSynchronous)
 
class LocalMultithreaded(DistributedMethod):
    ''' 
    Uses python multiprocessing to add tasks to a local pool

    NOTE: need to create like so:
    with LocalMultithreaded() as x:
        ...
    so that python garbage collector can close correctly
    '''
    def __init__(self, pool_size, evaluation_method: EvaluationMethod):
        self.queue = mp.Queue()
        self.evaluation_method = evaluation_method
        self.pool = mp.Pool(pool_size, initializer=worker_initializer, initargs=(self.queue,))

    def add_task(self, dna, policy_network_class):

        self.pool.apply_async(worker, (dna, policy_network_class, self.evaluation_method))

    def get_task_result(self):
        return self.queue.get()

    # TODO understand this better, is it correct?
    def __enter__(self):
        self.pool.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.__exit__(exc_type, exc_val, exc_tb)
        

DistributedMethod.register(LocalMultithreaded)



class DistributedRedis(DistributedMethod):

    def __init__(self, pool_size, evaluation_method: EvaluationMethod):
        self.queue = mp.Queue()
        self.evaluation_method = evaluation_method
        self.pool = mp.Pool(pool_size, initializer=worker_initializer, initargs=(self.queue,))

    def add_task(self, dna, policy_network_class):

        self.pool.apply_async(worker, (dna, policy_network_class, self.evaluation_method))

    def get_task_result(self):
        return self.queue.get()

    def retry_connect(self, redis_cfg, tries=300, base_delay=4.):
        for i in range(tries):
            try:
                r = redis.StrictRedis(**redis_cfg)
                r.ping()
                return r
            except redis.ConnectionError as e:
                if i == tries - 1:
                    raise
                else:
                    delay = base_delay * (1 + (os.getpid() % 10) / 9)
                    logger.warning('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.format(
                        redis_cfg, delay, i + 2, tries, e))
                    time.sleep(delay)

    def retry_get(pipe, key, tries=300, base_delay=4.):
        for i in range(tries):
            # Try to (m)get
            if isinstance(key, (list, tuple)):
                vals = pipe.mget(key)
                if all(v is not None for v in vals):
                    return vals
            else:
                val = pipe.get(key)
                if val is not None:
                    return val
            # Sleep and retry if any key wasn't available
            if i != tries - 1:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
                time.sleep(delay)
        raise RuntimeError('{} not set'.format(key))

'''
Simplest framework:
Each worker system only has to 
1. query the master for a new task
2. publish results of task to the master

'''
