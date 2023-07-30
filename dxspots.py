#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2022-2023 Fred W6BSD
# All rights reserved.
#
#

__version__ = "0.1.2"

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

plt.style.use(['classic', 'tableau-colorblind10'])

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

def read_data(dbname, bucket_size, days=14):
  bucket = lambda x: int(bucket_size * int(x.hour / bucket_size))
  start_date = datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=days)
  data = {}

  logger.info('Reading data from: %s', dbname)
  conn = sqlite3.connect(dbname, timeout=3, detect_types=DETECT_TYPES)
  sql = ("SELECT de_cont, strftime('%Y-%m-%d %H', datetime(time, 'unixepoch')) as tm, "
         "count(*) FROM dxspot WHERE time >= ? group by tm, de_cont;")
  record_cnt = 0
  result = conn.execute(sql, (start_date,))
  for cnt, row in enumerate(result):
    date = datetime.strptime(row[1], '%Y-%m-%d %H')
    date = date.replace(hour=bucket(date), minute=0, second=0, microsecond=0)
    if date not in data:
      data[date] = defaultdict(int)
    data[date][row[0]] += row[2]
    record_cnt = cnt
  logger.info('Records read: %d, data-size: %d', record_cnt, len(data))
  return sorted(data.items())


def graph(data, target_dir, filename, smooth_factor=5, show_total=False):
  # pylint: disable=too-many-locals
  assert smooth_factor % 2 != 0, 'smooth_factor should be an odd number'
  graphname = os.path.join(target_dir, filename)
  keys = ['EU', 'AS', 'OC', 'NA', 'SA', 'AF']
  continents = {}
  now = datetime.utcnow().strftime('%Y/%m/%d %H:%M UTC')

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

  fig.suptitle('Band Activity for each continent', fontsize=14, fontweight='bold')
  fig.text(0.01, 0.02, f'SunFluxBot By W6BSD {now}')

  for key in keys:
    plt.plot(xdata, continents[key], linewidth=1.75, label=key)

  if show_total:
    total = np.sum(np.array(list(continents.values())), axis=0)
    plt.plot(xdata, total, linewidth=.5, label='Total', color='gray')

  weekend_days = set([])
  for _time in labels:
    day = _time.date()
    if day in weekend_days or day.isoweekday() not in (6, 7):
      continue
    weekend_days.add(day)
  weekend_days = list(weekend_days)
  weekend_days.sort()

  for day in weekend_days:
    end = datetime(day.year, day.month, day.day, 23, 59)
    if labels[-1] < end:
      end = labels[-1]
    axgc.axvspan(date2num(day), date2num(end), color="skyblue", alpha=0.5)

  axgc.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
  axgc.xaxis.set_major_locator(DayLocator(interval=2))
  axgc.xaxis.set_minor_locator(HourLocator(byhour=range(0, 24, 6)))
  axgc.set_ylim(ymin=1)
  # axgc.set_yscale("log")

  axgc.set_ylabel('Spots / hour')
  axgc.grid(color="gray", linestyle="dotted", linewidth=.5)

  fig.autofmt_xdate(rotation=10, ha="center")
  legend = plt.legend(loc='upper left', fontsize=10, facecolor='white')
  for line in legend.get_lines():
    line.set_linewidth(4.0)

  plt.savefig(graphname, transparent=False, dpi=72)


def main():
  global logger
  logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                      datefmt="%H:%M:%S", level=logging.INFO)
  logger = logging.getLogger('dxspot')

  parser = argparse.ArgumentParser(description="Graph dxcc trafic")
  parser.add_argument("-b", "--bucket", type=int, default=3,
                      help="Time bucket [default: %(default)d]")
  parser.add_argument("-d", "--database", required=True,
                      help="Sqlite3 database path")
  parser.add_argument("-D", "--days", type=int, default=14,
                      help="Number of days to graph [default: %(default)d]")
  parser.add_argument("-f", "--filename", default="dxcc-stats.svg",
                      help="Graph ile name [default: %(default)s]")
  parser.add_argument("-s", "--smooth", type=int, default=5,
                      help="Graph smoothing factor [default: %(default)d]")
  parser.add_argument("-t", "--target-dir", default="/tmp",
                      help="Where to copy the graph [default: %(default)s]")
  parser.add_argument("-T", "--show_total", action="store_true", default=False,
                      help="Show the total number of sports")
  opts = parser.parse_args()
  if opts.smooth % 2 == 0:
    parser.error("The smoothing factor should be an odd number")

  logger.info('Starting: --smooth=%d --bucket=%d --days=%d', opts.smooth, opts.bucket, opts.days)
  data = read_data(opts.database, opts.bucket, opts.days)
  graph(data, opts.target_dir, opts.filename, opts.smooth, opts.show_total)

if __name__ == '__main__':
  main()
