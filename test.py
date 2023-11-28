from ray.job_submission import JobSubmissionClient

# If using a remote cluster, replace 127.0.0.1 with the head node's IP address.
client = JobSubmissionClient("http://192.168.88.240:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python3 /home/user/Ray_test/task.py"
    # Path to the local directory that contains the script.py file
)
print(job_id)