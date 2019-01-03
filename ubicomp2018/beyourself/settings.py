import yaml
import os
import logging
import pytz

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("America/Chicago")

DATEFORMAT = "%m-%d-%y"

ABSOLUTE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

RELATIVE_TIME_FORMAT = "%H:%M:%S.%f"

NECKLACE_HEADER = ['Time', 'proximity', 'ambient', 'leanForward', 'qW', 'qX', 'qY', 'qZ', 'aX', 'aY', 'aZ', 'cal',
                   'energy']

NECKLACE_SENSOR = ['proximity', 'ambient', 'leanForward', 'energy']

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

setting_path = os.path.join(__location__, 'config.yaml')

with open(setting_path) as f:
    config = yaml.load(f)

    logger.debug("Loading setting file %s", setting_path)

    RAW_FOLDER = config['raw_folder']
    TEMP_FOLDER = config['tmp_folder']
    CLEAN_FOLDER = config['clean_folder']
    ANNO_FOLDER = config['anno_folder']

SAMPLING_RATE = 20

# RNN configuration
BATCH_SIZE = 10
MAX_LENGTH = 1200
MAX_LENGTH_SEC = MAX_LENGTH/SAMPLING_RATE
WIN = 60
N_CHUNK = MAX_LENGTH // WIN
N_CLASS_CHEWING_BITES = 2
N_CLASS_CHEWING_OTHERS = 2
N_SENSOR = 4


# Hidden layers
N_FEATURE_1 = 128
N_FEATURE_2 = 64

# RNN layers
RNN_UNITS_1 = 64
RNN_UNITS_2 = 32

LOG_FOLDER = 'rnnlog'
MODEL_FOLDER = 'rnnmodel'

VISUALIZE_FOLDER = 'visualize'
