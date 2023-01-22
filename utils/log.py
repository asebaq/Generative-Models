import logging
import os

package_parent = '.'
script_dir = os.path.realpath(package_parent)


def setup_custom_logger(log_dir, name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    filename = os.path.abspath(os.path.join(script_dir, log_dir, 'logfile.log'))
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
