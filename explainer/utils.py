import sys
import logging
import time
from datetime import datetime

import pytz


tz = pytz.timezone('Europe/Berlin')


def timestamp():
    return str(datetime.now(tz))


def tz_converter(*args):
    return datetime.now(tz).timetuple()


def get_time_mils():
    return int(round(time.time() * 1000))


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    stream=sys.stdout
)
logging.Formatter.converter = tz_converter
logger = logging.root
