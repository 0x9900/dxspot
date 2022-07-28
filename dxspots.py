#!/usr/bin/env python3.9

import argparse
import logging
import os
import sqlite3

from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter, HourLocator
from matplotlib.ticker import AutoMinorLocator

from scipy.interpolate import make_interp_spline

DETECT_TYPES = sqlite3.PARSE_DECLTYPES

def adapt_datetime(t_stamp):
  return t_stamp.timestamp()

def convert_datetime(t_stamp):
  try:
    return datetime.fromtimestamp(float(t_stamp))
  except ValueError:
    return None

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter('timestamp', convert_datetime)

def read_data(dbname, bucket_size=3):
  logger.info('Reading data from: %s', dbname)
  bucket = lambda x: int(bucket_size * int(x.hour / bucket_size))
  conn = sqlite3.connect(dbname, timeout=3, detect_types=DETECT_TYPES)
  data = {}
  result = conn.execute('select de_cont, time from dxspot where de_cont != ""')
  for row in result:
    date = row[1].replace(hour=bucket(row[1]), minute=0, second=0, microsecond=0)
    if date not in data:
      data[date] = defaultdict(int)
    data[date][row[0]] += 1

  return sorted(data.items())


def graph(data, target_dir):
  graphname = os.path.join(target_dir, 'dxcc-stats.svg')
  keys = ['EU', 'AS', 'OC', 'NA', 'SA', 'AF']
  continents = {}

  logger.info('Generating graph file: %s', graphname)

  labels = np.array([d[0].timestamp() for d in data])
  xdata = np.linspace(labels.min(), labels.max(), len(labels) * 10)

  for ctn in keys:
    ydata = np.array([d[1][ctn] for d in data])
    spl = make_interp_spline(labels, ydata, k=5)
    ydata = spl(xdata)
    ydata[ydata < 0] = 0
    continents[ctn] = ydata

  _, axgc = plt.subplots(figsize=(12, 5))

  xdata = np.array([datetime.fromtimestamp(d) for d in xdata])
  labels = np.array([datetime.fromtimestamp(d) for d in labels])

  formatter = DateFormatter('%Y-%m-%d')
  plt.title('DX Spots / Continent', fontsize=18)

  for key in keys:
    plt.plot(xdata, continents[key], linewidth=1.5, label=key)

  total = np.sum(np.array(list(continents.values())), axis=0)
  plt.plot(xdata, total, linewidth=.5, label='Total', color='gray')

  axgc.xaxis.set_major_formatter(formatter)
  axgc.xaxis.set_tick_params(rotation=10, labelsize=10)
  axgc.xaxis.set_minor_locator(HourLocator())
  axgc.xaxis.set_minor_locator(AutoMinorLocator(6))
  axgc.set_xlabel('Dates (UTC)')
  axgc.set_ylabel('Numer of spots')

  axgc.grid(True)

  plt.legend(fontsize=10, facecolor='white')
  plt.savefig(graphname, transparent=False, dpi=72)


def main():
  parser = argparse.ArgumentParser(description='Graph dxcc trafic')
  parser.add_argument("--target-dir", default='/tmp',
                      help="Where to copy the graph")
  parser.add_argument("--database", required=True, help="Sqlite3 database path")
  opts = parser.parse_args()
  data = read_data(opts.database)
  graph(data, opts.target_dir)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  main()
