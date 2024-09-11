

import time
import os
import psutil
import logging

#%%

def timer(func):

    def wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        logging.info(f"Execution time: {round(execution_time/60, 1)} minutes ({round(execution_time/3600, 1)} hours)")
        # return the result of the decorated function execution
        return result
    # return reference to the wrapper function
    return wrapper


def exception_handler(func):

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle the exception
            print(f"An exception occurred: {str(e)}")

    return wrapper


def retry(max_attempts, delay=1):
    def decorator(func):

        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)

            print(f"Function failed after {max_attempts} attempts")

        return wrapper
    return decorator



# inner psutil function
def process_memory():
	process = psutil.Process(os.getpid())
	mem_info = process.memory_info()
	return mem_info.rss

# decorator function for memory consumption
def ram_consumption(func):
	def wrapper(*args, **kwargs):

		mem_before = process_memory()
		result = func(*args, **kwargs)
		mem_after = process_memory()
		logging.info("{}:consumed memory: {:,}".format(
			func.__name__,
			mem_before, mem_after, mem_after - mem_before))

		return result
	return wrapper


#%% test decorators on a class


# @timer
# @retry(2, delay = 1)
# class prova():
#     def __init__(self, a: int):
#         self.a = a

#     @timer
#     def do_stuff(self):
        
#         time.sleep(2)
#         print('i wait 2 seconds and your number is ', self.a)
#         blabla # this will raise an exception
#         # this will be re executed failing again

# prova(3)
# prova(3).do_stuff()

