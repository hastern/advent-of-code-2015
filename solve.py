#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import sys
import argparse
import collections
import Crypto.Hash.MD5 as MD5

from tasks import tasks
from inputs import inputs

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser("Advent of Code Solutions")
parser.add_argument("day", type=int)
parser.add_argument("--input", type=str)


def walk(route, x_step={"^": 0, "v": 0, "<": -1, ">": 1}, y_step={"^": -1, "v": 1, "<": 0, ">": 0}):
    x, y = 0, 0
    history = [(x, y)]
    for step in route:
        x += x_step[step]
        y += y_step[step]
        history += [(x, y)]
    return history


def mine(key, zeros=5):
    digest = ""
    round = 0
    zeros = "0" * zeros
    while not digest.startswith(zeros):
        round += 1
        secret = key + str(round)
        digest = MD5.new(secret).hexdigest()
        if round % 1000 == 0:
            print digest, secret, "\r",
    return round, digest


solutions = [
    lambda i: None,
    lambda i: (sum([{"(": 1, ")": -1}[c] for c in filter(lambda e: e in "()", i)]),
               (lambda m, i: m(m, i))((lambda m, r, f=0, s=0: s if f < 0 or len(r) == 0 else m(m, r[1:], f + ({"(": 1, ")": -1}[r[0]]), s + 1)), filter(lambda e: e in "()", i))
               ),
    lambda i: (sum(map(lambda (l, w, h): ((2 * l * w) + (2 * w * h) + (2 * h * l)) + min(l * w, w * h, h * l), map(lambda l: map(int, l.split("x")), i.splitlines()))),
               sum(map(lambda (l, w, h): min(((2 * l) + (2 * w)), ((2 * l) + (2 * h)), ((2 * h) + (2 * w))) + (l * w * h), map(lambda l: map(int, l.split("x")), i.splitlines()))),
               ),
    lambda i: (len(collections.Counter(walk(i))),
               len(collections.Counter(walk(i[::2]) + walk(i[1::2])))
               ),
    lambda i: (mine(input),
               mine(input, zeros=6)
               ),
    lambda i: (sum([(not any(map(lambda p: p in line, ["ab", "cd", "pq", "xy"]))) and (len(filter(lambda c: c in "aeiou", line)) >= 3) and (any(map(lambda p: p in line, [c * 2 for c in "abcdefghijklmnopqrstuvwxyz"]))) for line in map(str.strip, i.splitlines())]),
               ),
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = inputs[args.day] if args.input is None else args.input
        print tasks[args.day]
        print "Solution:"
        print solutions[args.day](input)
