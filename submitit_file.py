import submitit


def add(a, b):
    return a + b


# the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
# The specified folder is used to dump job information, logs and result when finished
# %j is replaced by the job id at runtime
log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder=log_folder)
# The AutoExecutor provides a simple abstraction over SLURM to simplify switching between local and slurm jobs (or other clusters if plugins are available).
# specify sbatch parameters (here it will timeout after 4min, and run on dev)
# This is where you would specify `gpus_per_node=1` for instance
# Cluster specific options must be appended by the cluster name:
# Eg.: slurm partition can be specified using `slurm_partition` argument. It
# will be ignored on other clusters:
executor.update_parameters(timeout_min=4, slurm_partition="newgen")
# The submission interface is identical to concurrent.futures.Executor
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = (
    job.result()
)  # waits for the submitted function to complete and returns its output
print(output)
# if ever the job failed, job.result() will raise an error with the corresponding trace
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
