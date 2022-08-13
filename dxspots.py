#!/usr/bin/env python3.9

__version__ = "0.1.1"

import argparse
import logging
import os
import sqlite3

from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter, DayLocator, HourLocator, date2num

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
  bucket = lambda x: int(bucket_size * int(x.hour / bucket_size))
  start_date = datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=10)
  data = {}

  logger.info('Reading data from: %s', dbname)
  conn = sqlite3.connect(dbname, timeout=3, detect_types=DETECT_TYPES)

  result = conn.execute('SELECT de_cont, time FROM dxspot WHERE time >= ?', (start_date,))
  for row in result:
    date = row[1].replace(hour=bucket(row[1]), minute=0, second=0, microsecond=0)
    if date not in data:
      data[date] = defaultdict(int)
    data[date][row[0]] += 1

  return sorted(data.items())


def graph(data, target_dir, filename, smooth_factor=5):
  # pylint: disable=too-many-locals
  assert smooth_factor % 2 != 0, 'smooth_factor should be an odd number'
  graphname = os.path.join(target_dir, filename)
  keys = ['EU', 'AS', 'OC', 'NA', 'SA', 'AF']
  continents = {}

  logger.info('Generating graph file: %s', graphname)

  labels = np.array([d[0].timestamp() for d in data])
  xdata = np.linspace(labels.min(), labels.max(), len(labels) * 10)

  for ctn in keys:
    ydata = np.array([d[1][ctn] for d in data])
    spl = make_interp_spline(labels, ydata, k=smooth_factor)
    ydata = spl(xdata)
    ydata[ydata < 0] = 0
    continents[ctn] = ydata

  fig, axgc = plt.subplots(figsize=(12, 5))
  axgc.tick_params(labelsize=10)

  xdata = np.array([datetime.fromtimestamp(d) for d in xdata])
  labels = np.array([datetime.fromtimestamp(d) for d in labels])

  formatter = DateFormatter('%Y-%m-%d')
  plt.title('DX Spots / Continent', fontsize=18)

  for key in keys:
    plt.plot(xdata, continents[key], linewidth=1.5, label=key)

  total = np.sum(np.array(list(continents.values())), axis=0)
  plt.plot(xdata, total, linewidth=.5, label='Total', color='gray')

  weekend_days = set([])
  for time in labels:
    day = time.date()
    if day in weekend_days or day.isoweekday() not in (6, 7):
      continue
    weekend_days.add(day)

  for day in weekend_days:
    end = datetime(day.year, day.month, day.day, 23, 58)
    axgc.axvspan(date2num(day), date2num(end), color="skyblue", alpha=0.5)


  axgc.xaxis.set_major_formatter(formatter)
  axgc.xaxis.set_major_locator(DayLocator())
  axgc.xaxis.set_minor_locator(HourLocator(interval=2))
  axgc.set_ylabel('Numer of spots')
  axgc.grid(color="gray", linestyle="dotted", linewidth=.75)

  fig.autofmt_xdate(rotation=10, ha="center")
  plt.legend(fontsize=10, facecolor='white')
  plt.savefig(graphname, transparent=False, dpi=72)


def main():
  parser = argparse.ArgumentParser(description="Graph dxcc trafic")
  parser.add_argument("-b", "--bucket", type=int, default=3,
                      help="Time bucket")
  parser.add_argument("-d", "--database", required=True,
                      help="Sqlite3 database path")
  parser.add_argument("-f", "--filename", default="dxcc-stats.svg",
                      help="Graph ile name")
  parser.add_argument("-s", "--smooth", type=int, default=5,
                      help="Graph smoothing factor")
  parser.add_argument("-t", "--target-dir", default="/tmp",
                      help="Where to copy the graph")
  opts = parser.parse_args()
  if opts.smooth % 2 == 0:
    parser.error("The smoothing factor should be an odd number")

  data = read_data(opts.database, opts.bucket)
  graph(data, opts.target_dir, opts.filename, opts.smooth)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  main()
