# An alternative lib to trt_test to let TRT_LLM developer run test using pure pytest command
import contextlib
import logging
import os
import platform
import signal
import subprocess
import sys

import psutil

general_logger = logging.getLogger("general")
general_logger.setLevel(logging.CRITICAL)

exists = os.path.exists
is_windows = lambda: platform.system() == "Windows"
is_linux = lambda: platform.system() == "Linux"
is_wsl = lambda: False  # FIXME: llm cases never on WSL?
makedirs = os.makedirs
wsl_to_win_path = lambda x: x  # FIXME: a hack for llm not run on WSL
SessionDataWriter = None  # TODO: hope never runs


@contextlib.contextmanager
def altered_env(**kwargs):
    old = {}
    for k, v in kwargs.items():
        if k in os.environ:
            old[k] = os.environ[k]
        os.environ[k] = v
    try:
        yield
    finally:
        for k in kwargs:
            if k not in old:
                os.environ.pop(k)
            else:
                os.environ[k] = old[k]


# Our own version of subprocess functions that clean up the whole process tree upon failure.
# This ensures subsequent tests won't be affected by left over processes from previous testcase.
#
# On Linux we create a new session (start_new_session) when starting subprocess to trace
# descendants even when parent processes already exited.
# Subprocesses spawned by tests usually create their own process groups, so killpg() is not
# enough here. However, they usually don't create new session, so we use it to track.
#
# On Windows we create a job object and put the subprocess into it. Descendants created by
# a process in job will also in the job. Terminate the job object in turn terminates all process in
# the job.

if is_linux():
    Popen = subprocess.Popen

    def list_process_sid(sid: int):
        current_uid = os.getuid()

        pids = []
        for proc in psutil.process_iter(['pid', 'uids']):
            if current_uid in proc.info['uids']:
                try:
                    if os.getsid(proc.pid) == sid:
                        pids.append(proc.pid)
                except (ProcessLookupError, PermissionError):
                    pass

        return pids

    def cleanup_process_tree(p: subprocess.Popen, has_session=False):
        target_pids = set()
        if has_session:
            # Session ID is the pid of the leader process
            target_pids.update(list_process_sid(p.pid))

        # Backup plan: using ppid to build subprocess tree
        try:
            target_pids.update(
                sub.pid
                for sub in psutil.Process(p.pid).children(recursive=True))
        except psutil.Error:
            pass

        print("Found leftover pids:", target_pids)
        for pid in target_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        p.kill()

elif is_windows():
    import pywintypes
    import win32api
    import win32job

    class MyHandle:

        def __init__(self, handle):
            self.handle = handle

        def __del__(self):
            win32api.CloseHandle(self.handle)

    def Popen(*popenargs, start_new_session, **kwargs):
        job_handle = None
        if start_new_session:
            job_handle = win32job.CreateJobObject(None, "")
        p = subprocess.Popen(*popenargs, **kwargs)
        if start_new_session:
            # It would be best to start with creationflags=0x04 (CREATE_SUSPENDED),
            # add process to job, and resume the primary thread.
            # However, subprocess.Popen simply discarded the thread handle and tid.
            # Instead, simply hope we add the process early enough.
            try:
                win32job.AssignProcessToJobObject(job_handle, p._handle)
                p.job_handle = MyHandle(job_handle)
            except pywintypes.error:
                p.job_handle = None
        return p

    def cleanup_process_tree(p: subprocess.Popen, has_session=False):
        target_pids = []
        try:
            target_pids = [
                sub.pid
                for sub in psutil.Process(p.pid).children(recursive=True)
            ]
        except psutil.Error:
            pass

        if has_session and p.job_handle is not None:
            process_exit_code = 3600  # Some obvious special exit code
            try:
                win32job.TerminateJobObject(p.job_handle.handle,
                                            process_exit_code)
            except pywintypes.error:
                pass

        print("Found leftover pids:", target_pids)
        for pid in target_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        p.kill()


def call(*popenargs,
         timeout=None,
         start_new_session=True,
         suppress_output_info=False,
         **kwargs):
    if not suppress_output_info:
        print(f"Start subprocess with call({popenargs}, {kwargs})")
    with Popen(*popenargs, start_new_session=start_new_session, **kwargs) as p:
        try:
            retcode = p.wait(timeout=timeout)
            if retcode and start_new_session:
                cleanup_process_tree(p, True)
            return retcode
        except:
            cleanup_process_tree(p, start_new_session)
            raise


def check_call(*popenargs, **kwargs):
    print(f"Start subprocess with check_call({popenargs}, {kwargs})")
    retcode = call(*popenargs, suppress_output_info=True, **kwargs)
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return 0


def check_output(*popenargs, timeout=None, start_new_session=True, **kwargs):
    print(f"Start subprocess with check_output({popenargs}, {kwargs})")
    with Popen(*popenargs,
               stdout=subprocess.PIPE,
               start_new_session=start_new_session,
               **kwargs) as process:
        try:
            stdout, stderr = process.communicate(None, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            cleanup_process_tree(process, start_new_session)
            if is_windows():
                exc.stdout, exc.stderr = process.communicate()
            else:
                process.wait()
            raise
        except:
            cleanup_process_tree(process, start_new_session)
            raise
        retcode = process.poll()
        if retcode:
            if start_new_session:
                cleanup_process_tree(process, True)
            raise subprocess.CalledProcessError(retcode,
                                                process.args,
                                                output=stdout,
                                                stderr=stderr)
    return stdout.decode()


def make_clean_dirs(path):
    """
    Make directories for @path, clean content if it already exists.
    """
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def print_info(message: str) -> None:
    """
    Prints an informational message.
    """
    print(f"[INFO] {message}")
    sys.stdout.flush()
    general_logger.info(message)


def print_warning(message: str) -> None:
    """
    Prints a warning message.
    """
    print(f"[WARNING] {message}")
    sys.stdout.flush()
    general_logger.warning(message)


def print_error(message: str) -> None:
    """
    Prints an error message.
    """
    print(f"[ERROR] {message}")
    sys.stdout.flush()
    general_logger.error(message)


# custom test checker
def check_call_negative_test(*popenargs, **kwargs):
    print(
        f"Start subprocess with check_call_negative_test({popenargs}, {kwargs})"
    )
    retcode = call(*popenargs, suppress_output_info=True, **kwargs)
    if retcode:
        return 0
    else:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        print(
            f"Subprocess expected to fail with check_call_negative_test({popenargs}, {kwargs}), but passed."
        )
        raise subprocess.CalledProcessError(1, cmd)
