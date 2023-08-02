import subprocess
import time
import os

# User-defined parallelism
parallelism = int(input("Enter the level of parallelism: "))

# File to read commands from
cmd_file = "commands.txt"

# List to store subprocesses
processes = []

# Function to check running processes
def check_processes(processes, parallelism):
    # If the number of active processes is greater than or equal to the parallelism limit
    while len(processes) >= parallelism:
        for process in processes:
            # If the process has finished execution
            if process.poll() is not None:
                processes.remove(process)
        time.sleep(10)  # Avoid tight loop

# Read and execute commands
with open(cmd_file, 'r') as file:
    for line in file:
        cmd = line.strip()

        # Start a subprocess to execute the command
        processes.append(subprocess.Popen(cmd, shell=True))

        # Check the number of active processes
        check_processes(processes, parallelism)

# Check remaining active processes
check_processes(processes, 0)