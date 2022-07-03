import logging


# Define the global (package-wide) logger
logger = logging.getLogger('diffmat')


def config_logger(level: str = 'default'):
    """Configure the verbosity of the package-wide logger. This will affect the behavior of all
    child loggers in diffmat.

    Args:
        level (str, optional): Logging level.
            'none': Disable logging.
            'quiet': Logging level is set to `ERROR`; minimalistic on-screen output.
            'default': Logging level is set to `INFO`; reduced on-screen output.
            'verbose': Logging level is set to `DEBUG`; on-screen output additionally shows logger
                name and file name for easier tracking.
            Defaults to 'default'.
    """
    if level not in ('verbose', 'default', 'quiet', 'none'):
        raise ValueError("Valid logging level options are 'verbose', 'default', 'quiet', and "
                         "'none'")

    # Remove all existing handlers
    logger.handlers.clear()

    # None - no logging
    if level == 'none':
        logger.addHandler(logging.NullHandler())

    # Quiet - minimal logging and only pertaining to errors or critical issues
    elif level == 'quiet':
        logger.setLevel(logging.ERROR)

    else:
        # Default and verbose
        if level == 'default':
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            logger.setLevel(logging.INFO)
        else:
            formatter = logging.Formatter('[%(levelname)s] (%(name)s - %(filename)s) %(message)s')
            logger.setLevel(logging.DEBUG)

        # All debug messages are printed on the screen by default
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def get_logger(name: str = '') -> logging.Logger:
    """Create a logger or return the package-wide logger for user configuration.

    Args:
        name (str, optional): Logger name. Defaults to ''.

    Returns:
        Logger: A new logger bearing the provided name, or the package-wide logger if a name is
            not provided.
    """
    return logger if not name else logging.getLogger(f'diffmat.{name}')
