#!/usr/bin/env python3.9

import argparse
import logging
import os
import sqlite3
import sys

from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter, HourLocator, DayLocator, MinuteLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from scipy.interpolate import make_interp_spline, BSpline

STYLE = 'ggplot'
plt.style.use(STYLE)

BUCKET_SIZE = 2

bucket = lambda x: int(BUCKET_SIZE * int(x.hour / BUCKET_SIZE))

def read_data(dbname):
    logger.info('Reading data from: %s', dbname)
    conn = sqlite3.connect(dbname, timeout=3)
    data = {}
    result = conn.execute('select * from dxspot')
    for row in result:
        if row[4] == '':
            continue
        _date = datetime.fromtimestamp(row[7])
        date = _date.replace(hour=bucket(_date), minute=0, second=0, microsecond=0)
        if date not in data:
            data[date] = defaultdict(int)
        data[date][row[4]] += 1

    return sorted(data.items())


def graph(data, target_dir):
    graphname = os.path.join(target_dir, 'dxcc-stats.svg')
    keys = ['EU', 'AS', 'OC', 'NA', 'SA', 'AF']
    continents = {}

    logger.info('Generating graph file: %s', graphname)

    labels = np.array([d[0].timestamp() for d in data])
    x = np.linspace(labels.min(), labels.max(), 1500)

    for ctn in keys:
        y = np.array([d[1][ctn] for d in data])
        spl = make_interp_spline(labels, y, k=5)
        y = spl(x)
        y[y < 0] = 0
        continents[ctn] = y

    fig, ax = plt.subplots(figsize=(12, 5))

    x = [datetime.fromtimestamp(d) for d in x]
    labels = [datetime.fromtimestamp(d) for d in labels]

    formatter = DateFormatter('%Y-%m-%d')
    plt.title('DX Spots / Continent', fontsize=18)

    for key in keys:
        plt.plot(x, continents[key], linewidth=1.5, label=key)

    total = np.sum(np.array([x for x in continents.values()]), axis=0)
    plt.plot(x, total, linewidth=.5, label='Total', color='darkgray')

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(rotation=10, labelsize=10)
    ax.xaxis.set_minor_locator(HourLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.set_xlabel('Dates (UTC)')
    ax.set_ylabel('Numer of spots')

    ax.grid(True)

    plt.legend(fontsize=10, facecolor='white', framealpha=.5)
    plt.savefig(graphname, transparent=False, dpi=72)


def main(args=sys.argv[:1]):
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
