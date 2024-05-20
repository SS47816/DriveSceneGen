import os
import glob
import logging
import pickle

def chunks(input, n):
    """Yields successive n-sized chunks of input"""
    for i in range(0, len(input), n):
        yield input[i : i + n]

def get_logger(logger_name: str, level: int = logging.ERROR) -> logging.Logger:
    if logger_name is None:
        return None
    
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    
    return logger

def get_cache_name(parent_dir: str, child_dir: str) -> str:
    return os.path.join(parent_dir, f'{child_dir}_cached_filenames.pkl')

def cache_all_filenames(parent_dir: str, child_dir: str) -> str:
    filenames = glob.glob(os.path.join(parent_dir, child_dir + '/*'))
    cache = get_cache_name(parent_dir, child_dir)
    with open(cache, 'wb') as f:
        pickle.dump(filenames, f)
    return cache

def get_all_filenames(parent_dir: str, child_dir: str) -> list:
    cache = get_cache_name(parent_dir, child_dir)
    with open(cache, 'rb') as f:
        filenames = pickle.load(f)
    return filenames