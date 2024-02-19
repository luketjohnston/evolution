from abc import ABC, abstractmethod
import functools
import datetime
import time
import pickle
from threading import Thread
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
        print("Connecting to rabbitmq...", flush=True)
        while True: # Have to retry until rabbitmq service is up, TODO clean up
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host='rabbitmq', port=5672)) # rabbitmq to match docker service name??
                break
            except:
                time.sleep(1)
        print("Succesfully connected to rabbitmq", flush=True)
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
        ''' we have to do the work in a separate thread (so that pika can continue sending
        heartbeats to rabbitmq). Then we have to use add_callback_threadsafe to communicate 
        from this other thread when the job finishes.
        see https://stackoverflow.com/questions/52973253/rabbitmq-pika-exceptions-connectionclosed-1-error104-connection-reset-by '''

        def send_result(channel, delivery_tag, evaluation):
            print("Sending result", datetime.datetime.now(), flush=True)
            channel.basic_publish(
                exchange='',
                routing_key='results_queue',
                body=pickle.dumps(evaluation),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent
                ))
            print("Acking", datetime.datetime.now(), flush=True)
            channel.basic_ack(delivery_tag=delivery_tag)

        # We need to do the work in separate thread, otherwise pika won't be able
        # to send its heartbeats often enough and rabbitmq will time it out.
        def do_work_and_ack(ch, method, properties, body):
            print("doing work", datetime.datetime.now(), flush=True)
            dna = BasicDNA.deserialize(body) # TODO pass dna type as arg somewhere?
            evaluation = self.eval_method.eval(dna)
            delivery_tag = method.delivery_tag
            print("Done, adding callback", datetime.datetime.now(), flush=True)
            self.connection.add_callback_threadsafe(functools.partial(send_result, ch, delivery_tag, evaluation))

        def callback(ch, method, properties, body):
            print(f"Starting thread for {body}: ", datetime.datetime.now(), flush=True)
            t = Thread(target=do_work_and_ack, args=(ch, method, properties, body))
            t.start()



        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='task_queue', on_message_callback=callback)
        print("Starting consuming...", flush=True)
        self.channel.start_consuming()

