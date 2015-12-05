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
               (lambda m, i: m(m, i))((lambda m, r, f=0, s=0: s if f < 0 or len(r) == 0 else m(m, r[1:], f + ({"(": 1, ")": -1}[r[0]]), s + 1)), filter(lambda e: e in "()", i))
               ),
    lambda i: (sum(map(lambda (l, w, h): ((2 * l * w) + (2 * w * h) + (2 * h * l)) + min(l * w, w * h, h * l), map(lambda l: map(int, l.split("x")), i.splitlines()))),
               ),
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = inputs[args.day] if args.input is None else args.input
        print tasks[args.day]
        print "Solution:"
        print solutions[args.day](input)
