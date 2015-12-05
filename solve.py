#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import argparse

from tasks import tasks
from inputs import inputs

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser("Advent of Code Solutions")
parser.add_argument("day", type=int)
parser.add_argument("--input", type=str)


solutions = [
    lambda i: None,
    lambda i: (sum([{"(": 1, ")": -1}[c] for c in filter(lambda e: e in "()", i)]),
               ),
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = inputs[args.day] if args.input is None else args.input
        print tasks[args.day]
        print "Solution:"
        print solutions[args.day](input)
