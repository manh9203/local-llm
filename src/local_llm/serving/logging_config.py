import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("local_llm")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
