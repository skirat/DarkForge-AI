import logging
import sys
from pathlib import Path

_configured = False


def setup_logger(name: str = "pipeline", log_dir: Path | None = None) -> logging.Logger:
    """Return a logger that writes INFO to console and DEBUG to file."""
    global _configured
    logger = logging.getLogger(name)

    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _configured = True
    return logger
