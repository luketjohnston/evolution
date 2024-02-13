from abc import ABC, abstractmethod
import pickle
from codes import BasicDNA
from evaluations import EvaluationMethod
import multiprocessing as mp
import queue
import pika


class DistributedMethod(ABC):
    @abstractmethod
    def __init__(self, eval_method: EvaluationMethod):
        pass
    @abstractmethod
    def add_task(self, dna):
        pass
    @abstractmethod
    def get_task_results(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def worker(dna, evaluation_method):
  worker.queue.put(evaluation_method.eval(dna))

def worker_initializer(queue):
    worker.queue = queue

class LocalSynchronous(DistributedMethod):
    def __init__(self, evaluation_method: EvaluationMethod):
        self.evaluation_method = evaluation_method
        self.queue = queue.Queue()
        worker_initializer(self.queue)

    def add_task(self, dna):
        worker(dna, self.evaluation_method)

    def get_task_results(self):
        while True:
            yield self.queue.get()


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

    def add_task(self, dna):

        self.pool.apply_async(worker, (dna, self.evaluation_method))

    def get_task_results(self):
        while True:
            yield self.queue.get()

    # TODO understand this better, is it correct?
    def __enter__(self):
        self.pool.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.__exit__(exc_type, exc_val, exc_tb)
        

DistributedMethod.register(LocalMultithreaded)

class DistributedRabbitMQ(DistributedMethod):
    def __init__(self, evaluation_method: EvaluationMethod, is_master=True):
        # don't do anything with the evaluation method - will have to setup workers elsewhere
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='task_queue', durable=True) # TODO do we want durable?
        self.channel.queue_declare(queue='results_queue', durable=True)
        self.results_queue = queue.Queue()
        self.is_master = is_master
        self.eval_method = evaluation_method

    def add_task(self, dna):
        assert self.is_master
        self.channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=dna.serialize(), 
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent
            ))
    def get_task_results(self):
        assert self.is_master
        for method, properties, body in self.channel.consume('results_queue'):
            result = pickle.loads(body)
            self.channel.basic_ack(delivery_tag=method.delivery_tag)
            yield result

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def start_worker(self):
        assert not self.is_master

        # TODO import this from constants file somewhere? Or move into 
        # eval method?

        def callback(ch, method, properties, body):
            dna = BasicDNA.deserialize(body) # TODO pass dna type as arg somewhere?
            evaluation = self.eval_method.eval(dna)
            # send evaluation result back to master
            self.channel.basic_publish(
                exchange='',
                routing_key='results_queue',
                body=pickle.dumps(evaluation),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ))
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='task_queue', on_message_callback=callback)
        self.channel.start_consuming()

