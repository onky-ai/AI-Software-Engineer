import docker
import os
from docker.errors import DockerException, ImageNotFound, APIError
from docker.types import Mount

# --- Configuration ---
DOCKER_IMAGE = "python-runner-env:latest" # Image built from Dockerfile.python_runner
CONTAINER_WORKDIR = "/app" # Must match WORKDIR in Dockerfile.python_runner

def run_in_docker(command: list[str], host_project_dir: str, timeout_seconds: int = 60) -> tuple[int, str, str]:
    """
    Runs a command inside a Docker container with the project directory mounted.

    Args:
        command: The command and arguments to run (e.g., ["pytest", "-v"]).
        host_project_dir: The absolute path to the project directory on the host.
        timeout_seconds: Max execution time for the container.

    Returns:
        A tuple containing: (exit_code, stdout_str, stderr_str)
    """
    client = None
    container = None
    stdout_str = ""
    stderr_str = ""
    exit_code = -1 # Default to error state

    if not os.path.isdir(host_project_dir):
        return -1, "", f"Error: Host project directory '{host_project_dir}' does not exist."

    try:
        client = docker.from_env()

        # Ensure the image exists
        try:
            client.images.get(DOCKER_IMAGE)
        except ImageNotFound:
            return -1, "", f"Error: Docker image '{DOCKER_IMAGE}' not found. Please build it first."

        # Define the mount: Mount host directory to container's workdir
        mount = Mount(target=CONTAINER_WORKDIR, source=host_project_dir, type='bind', read_only=False)

        print(f"Attempting to run in Docker: {' '.join(command)}")
        print(f"  Image: {DOCKER_IMAGE}")
        print(f"  Mounting: {host_project_dir} -> {CONTAINER_WORKDIR}")

        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=command,
            mounts=[mount],
            working_dir=CONTAINER_WORKDIR,
            detach=True, # Run in background initially
            # --- Security Enhancements ---
            network_disabled=True, # Disable networking unless specifically needed!
            mem_limit="256m",      # Limit memory
            cpu_quota=50000,       # Limit CPU (e.g., 50% of one core)
            # remove=True          # Auto-remove (handled below with timeout)
            # user=1000             # Run as non-root user (if image supports it)
        )

        # Wait for container to finish with a timeout
        result = container.wait(timeout=timeout_seconds)
        exit_code = result.get('StatusCode', -1)

        # Get logs *after* container finishes
        stdout_bytes = container.logs(stdout=True, stderr=False)
        stderr_bytes = container.logs(stdout=False, stderr=True)
        stdout_str = stdout_bytes.decode('utf-8', errors='replace')
        stderr_str = stderr_bytes.decode('utf-8', errors='replace')

        print(f"Docker run finished. Exit Code: {exit_code}")
        # Optional: print concise logs here if needed
        # print(f"  Stdout: {stdout_str[:200]}{'...' if len(stdout_str)>200 else ''}")
        # print(f"  Stderr: {stderr_str[:200]}{'...' if len(stderr_str)>200 else ''}")


    except docker.errors.ContainerError as e:
        print(f"Docker ContainerError: {e}")
        stderr_str += f"\nDocker ContainerError: {e}"
        exit_code = e.exit_status
        if container:
            # Try to get logs even on error
            try:
                stdout_str += container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                stderr_str += container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
            except Exception:
                 pass # Ignore log errors if container failed badly
    except docker.errors.APIError as e:
        print(f"Docker APIError: {e}")
        stderr_str += f"\nDocker APIError: {e}"
        exit_code = -1
    except Exception as e:
        print(f"An unexpected error occurred during Docker execution: {e}")
        stderr_str += f"\nUnexpected Error: {e}"
        exit_code = -1
    finally:
        if container:
            try:
                container.remove(force=True) # Clean up the container
            except docker.errors.NotFound:
                pass # Container might already be gone
            except Exception as e:
                print(f"Warning: Failed to remove container {container.id}: {e}")
        # Note: Closing the client might be needed if used long-term,
        # but for function scope it's often okay to let it be garbage collected.
        # if client:
        #    client.close()

    return exit_code, stdout_str, stderr_str