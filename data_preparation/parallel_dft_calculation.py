import datetime
import os
import typing
import logging
import time

from multiprocessing import Pool
from utils import batch, ORCA_PATH

"""
Script Executes parallel DFT Calculations using ORCA. Number of Workers, as well as ORCA path and path
to a directory, containing folders with .inp files can be configured.
"""

# Specify a number of parallel worker processes - optimal for local machine on M1 chip - is 4
WORKERS = 4
# Specify the directory with compound folders containing .inp files
INPUT_FOLDER = "/Input_folder_path"
# Specify Calculation type OPT or NMR
CALC_TYPE = "NMR"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


# Definition of calculation job
def orca_job(input_data):
    num_tasks = len(input_data[1])
    total_time = 0
    for task in input_data[1]:
        try:
            task_index = input_data[1].index(task) + 1
            logging.info(
                f"Starting computation for {task} - {task_index} out of {num_tasks} on worker {input_data[0]}"
            )
            job_start = time.time()
            os.system(
                f"{ORCA_PATH}/orca {INPUT_FOLDER}/{task}/{task}_{CALC_TYPE}.inp > {INPUT_FOLDER}/{task}/{task}_{CALC_TYPE}.out"
            )
            job_end = time.time()
            task_time = job_end - job_start
            total_time += task_time
            average_time_per_task = total_time / task_index
            logging.info(
                f"{task} computed in {round(task_time, 2)} seconds on worker {input_data[0]} - "
                f"Estimated Time Remaining "
                f"{datetime.timedelta(seconds=round(average_time_per_task*(num_tasks-(task_index-1))))}"
            )
        except:
            logging.info(f"Calculations for {task} on worker {input_data[0]} failed")
    logging.info(
        f"Computation of {num_tasks} tasks on worker {input_data[0]} finished in"
        f" {datetime.timedelta(seconds=round(total_time))}"
    )


# Pool definition
def pool_handler(job_input: typing.List[typing.Tuple[int, typing.List[str]]]):
    p = Pool(WORKERS)
    p.map(orca_job, job_input)


# Pool Entrypoint
if __name__ == "__main__":
    logging.info(f"Preparing Computation using {WORKERS} workers")
    # Creating full list of compounds
    input_dir = os.listdir(INPUT_FOLDER)
    compound_folders = []
    for folder in input_dir:
        # Check if folder contains corresponding input file
        if os.path.isdir(f"{INPUT_FOLDER}/{folder}") and os.path.isfile(
            f"{INPUT_FOLDER}/{folder}/{folder}_{CALC_TYPE}.inp"
        ):
            compound_folders.append(folder)

    # Splitting list of compounds for each worker process
    tasks = []
    n_elements = len(compound_folders) // WORKERS
    for worker_tasks in batch(compound_folders, n_elements):
        tasks.append(worker_tasks)

    # Starting Calculation
    start = time.time()
    pool_handler(tasks)
    end = time.time()
    logging.info(
        f"Parallel Computation using {WORKERS} worker processes for {len(compound_folders)} compounds"
        f" Completed in {datetime.timedelta(seconds=round(end - start))}"
    )
