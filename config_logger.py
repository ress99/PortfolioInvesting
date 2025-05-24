"""Module to configure file logging for the PortfolioInvesting application."""


import logging
import os
from datetime import datetime
import config as c


# Format for log messages
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_LEVEL = c.LOG_LEVEL

# Create a consistent log filename with timestamp
log_filename = os.path.join("Logs", f"log_{datetime.now().strftime('%Y_%m_%d_%H-%M-%S')}.log")

# Configure logging once
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    format=LOG_FORMAT,
    level=LOG_LEVEL,
)
