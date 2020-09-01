import os
import logging
import colorlog

def init_logger(dunder_name, show_debug=False, log_to_file=False) -> logging.Logger:
    log_format = (
        '%(asctime)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(levelname)s - '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger(dunder_name)

    if show_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ## Note: these file outputs are left in place as examples
    ## Feel free to comment/uncomment and use the outputs as you like

    # Output full log
    if log_to_file:
        fh = logging.FileHandler(os.path.join('log', 'log.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
        # Output warning log
        #fh = logging.FileHandler(os.path.join('log', 'warning.log')
        #fh.setLevel(logging.WARNING)
        #formatter = logging.Formatter(log_format)
        #fh.setFormatter(formatter)
        #logger.addHandler(fh)
        #
        ## Output error log
        #fh = logging.FileHandler(os.path.join('log', 'error.log')
        #fh.setLevel(logging.ERROR)
        #formatter = logging.Formatter(log_format)
        #fh.setFormatter(formatter)
        #logger.addHandler(fh)

    return logger