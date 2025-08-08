import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logging():
    logger = logging.getLogger("weather")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(os.path.join(log_dir, 'app.log'),
                                                when='midnight',
                                                backupCount=7,
                                                encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger