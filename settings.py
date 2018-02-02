import os
from os.path import join, dirname

from binance.client import Client
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)