import getpass
import os
import socket
import sys
from datetime import datetime
from logging import DEBUG, WARNING, Logger, StreamHandler, basicConfig, getLogger
from typing import List, Optional, Tuple, Union

from lightgbm.callback import _format_eval_result

CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory


def get_logger(
    fn_args: List[str],
    dir: str = "absolute/path/to/log/directory",
    fold: Optional[int] = None,
    exec_time: Optional[Union[str, int]] = None,
    tta: bool = False,
) -> Tuple[Logger, str]:
    if exec_time is None:
        exec_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = "_".join(fn_args) + "_" + str(exec_time)
    log_dir = os.path.join(dir, log_name)
    if tta:
        log_dir = os.path.join(log_dir, "tta")
    if fold is not None:
        log_dir = os.path.join(log_dir, f"cv{fold}")
    log_fn = os.path.join(log_dir, f"{log_name}.log")
    os.makedirs(log_dir, exist_ok=True)  # Create log directory

    formatter = "%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s"
    basicConfig(filename=log_fn, level=DEBUG, format=formatter, force=True)

    getLogger("matplotlib").setLevel(WARNING)  # Suppress matplotlib logging
    # getLogger("requests").setLevel(WARNING)  # Suppress requests logging
    # getLogger("urllib3").setLevel(WARNING)  # Suppress urllib3 logging
    getLogger("cpp").setLevel(WARNING)  # Suppress cpp Warning

    # Handle logging to both logging and stdout.
    getLogger().addHandler(StreamHandler(sys.stdout))

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.debug(f"{getpass.getuser()}@{socket.gethostname()}:{log_fn}")
    return logger, log_dir


def log_evaluation(logger: Logger, period: int = 1, show_stdv: bool = True, level=DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = "\t".join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, f"[{env.iteration + 1}]\t{result}")

    _callback.order = 10
    return _callback
