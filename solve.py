#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import sys
import argparse
import collections
import Crypto.Hash.MD5 as MD5
import pyparsing as pp
import operator
import itertools
import json

from tasks import tasks
from inputs import inputs

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser("Advent of Code Solutions")
parser.add_argument("day", type=int)
parser.add_argument("--input", type=str, nargs="+")


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


def parse_light_switches(instrs):
    NUMBER = pp.Word(pp.nums)
    CORNER = pp.Group(NUMBER + pp.Literal(",").suppress() + NUMBER)
    RECT = CORNER + pp.Literal("through").suppress() + CORNER
    SWITCH = pp.Literal("turn").suppress() + (pp.Literal("on") | pp.Literal("off"))
    TOGGLE = pp.Literal("toggle")
    INSTRUCTION = pp.Group((SWITCH | TOGGLE) + RECT)
    INSTRUCTIONS = pp.OneOrMore(INSTRUCTION)
    return map(lambda (i, t, b): (i, tuple(map(int, t)), tuple(map(int, b))), INSTRUCTIONS.parseString(instrs))


build_lights = lambda size=1000, initial=False: {(x, y): initial for x in range(size) for y in range(size)}


def rect(top, bottom):
    for x in range(top[0], bottom[0] + 1):
        for y in range(top[1], bottom[1] + 1):
            yield x, y


def access_lights(lights, instrs, funcs={"toggle": lambda v: not v, "on": lambda v: True, "off": lambda v: False}, after=lambda v: v):
    for i, t, b in instrs:
        for coord in rect(t, b):
            lights[coord] = after(funcs[i](lights[coord]))
    return lights


def parse_logic(input):
    CONST = pp.Word(pp.nums).setParseAction(lambda s, l, t: [int(t[0])])
    WIRE = pp.Word(pp.alphas)
    UN_OP = pp.Literal("NOT")("op") + WIRE("a")
    BIN_OP = (WIRE("a") | CONST("c")) + (pp.Literal("AND") | pp.Literal("OR") | pp.Literal("XOR"))("op") + WIRE("b")
    CONST_OP = WIRE("a") + (pp.Literal("RSHIFT") | pp.Literal("LSHIFT"))("op") + CONST("c")
    ASSIGN_OP = (pp.Empty().addParseAction(lambda s, l, t: ["ASSIGN"]))("op") + (CONST("c") | WIRE("a"))
    OP = pp.Group(UN_OP | BIN_OP | CONST_OP | ASSIGN_OP)("operation")
    INSTR = pp.Group(OP + pp.Literal("->").suppress() + WIRE("dest"))
    INSTRS = pp.OneOrMore(INSTR)
    return INSTRS.parseString(input)


class Gate(object):

    def __init__(self, name, value=None, inputs=[], operation=None, add_operands=[]):
        self.name = name
        self.value = value
        self.outputs = []
        self.inputs = list(inputs)
        self.operation = operation
        self.add_operands = add_operands

    def add_output(self, gate):
        self.outputs.append(gate)
        return self

    @property
    def resolved(self):
        return self.value is not None

    def resolve(self):
        if self.resolved or self.operation is None:
            return
        operands = []
        for input in self.inputs:
            if not input.resolved:
                return  # Missing Input: Can't resolve
            operands.append(input.value)
        self.value = self.operation(*(operands + list(self.add_operands))) & 0xFFFF
        for output in self.outputs:
            output.resolve()


def build_gates(instr):
    operations = {
        "OR": operator.or_,
        "AND": operator.and_,
        "XOR": operator.xor,
        "NOT": operator.inv,
        "LSHIFT": operator.lshift,
        "RSHIFT": operator.rshift,
        "ASSIGN": lambda *v: v[0]
    }
    for instr in instr:
        if instr.operation.op == "ASSIGN" and "c" in instr.operation:
            yield Gate(instr.dest, instr.operation.c)
        elif instr.operation.op == "ASSIGN":
            yield Gate(instr.dest, inputs=(instr.operation.a, ), operation=operations[instr.operation.op])
        elif instr.operation.op == "NOT":
            yield Gate(instr.dest, inputs=(instr.operation.a, ), operation=operations[instr.operation.op])
        elif instr.operation.op in ["OR", "AND", "XOR"]:
            if "a" in instr.operation:
                yield Gate(instr.dest,
                           inputs=(instr.operation.a, instr.operation.b),
                           operation=operations[instr.operation.op],
                           )
            elif "c" in instr.operation:
                yield Gate(instr.dest,
                           inputs=(instr.operation.b, ),
                           operation=operations[instr.operation.op],
                           add_operands=(instr.operation.c, ),
                           )
            else:
                print "UNKNOWN", instr
        elif instr.operation.op in ["LSHIFT", "RSHIFT"]:
            yield Gate(instr.dest,
                       inputs=(instr.operation.a, ),
                       operation=operations[instr.operation.op],
                       add_operands=(instr.operation.c, ),
                       )


init_gates = lambda instrs: {gate.name: gate for gate in build_gates(instrs)}


def update_gate(gates, gate, value):
    gates[gate].value = value
    return gates


def wire_gates(gates):
    # Wire gates up
    for gate in gates.itervalues():
        gate.inputs = [gates[g].add_output(gate) for g in gate.inputs]
    return gates


def resolve_logic(gates):
    readyset = [gate for gate in gates.values() if gate.resolved]
    for gate in readyset:
        map(Gate.resolve, gate.outputs)
    return gates


def unescape(s):
    chs = []
    p = 0
    while p < len(s):
        if s[p] == "\\":  # escape sequence
            if s[p + 1] == "x":
                chs.append(int(s[p + 2:p + 4], 16))
                p += 3
            else:
                chs.append(ord(s[p + 1]))
                p += 1
        else:  # Standard character
            chs.append(ord(s[p]))
        p += 1
    return chs


escape = lambda s: '"{}"'.format("".join(["\\{}".format(c) if c in ['"', '\\'] else c for c in s]))


def transform_to_gv(input, fname="graph.gv"):
    with open(fname, "w") as output:
        output.write("graph g {\n")
        for line in input.splitlines():
            parts = line.split()
            output.write("  {} -- {} [label={}]".format(*parts[::2]))
        output.write("}\n")


def fold(_iterable, func=lambda e, a: e, init=None):
    result = init
    for elem in _iterable:
        result = func(elem, result)
    return result


generate_net = lambda input: {tuple(sorted(parts[:3:2])): int(parts[4]) for parts in map(str.split, input.splitlines())}
route_length = lambda net, route: sum([net[tuple(sorted(route[c:c + 2]))] for c in range(len(route) - 1)])
all_routes = lambda net: itertools.permutations(set(list(sum(net.keys(), ()))))
shortest_route = lambda net: fold(all_routes(net), lambda e, a: min((route_length(net, e), e), a), (sys.maxint, None))
longest_route = lambda net: fold(all_routes(net), lambda e, a: max((route_length(net, e), e), a), (0, None))

shortest_route = lambda input: (lambda net: reduce(lambda a, e: min((sum([net[tuple(sorted(e[c:c + 2]))] for c in range(len(e) - 1)]), e), a), itertools.permutations(set(list(sum(net.keys(), ())))), (sys.maxint, None)))({tuple(sorted(parts[:3:2])): int(parts[4]) for parts in map(str.split, input.splitlines())})
longest_route = lambda input: (lambda net: reduce(lambda a, e: max((sum([net[tuple(sorted(e[c:c + 2]))] for c in range(len(e) - 1)]), e), a), itertools.permutations(set(list(sum(net.keys(), ())))), (0, None)))({tuple(sorted(parts[:3:2])): int(parts[4]) for parts in map(str.split, input.splitlines())})


push_tok = lambda l, t: l[-1].append(t) if l[-1][0] == t else l.append([t])
split_str = lambda s: (lambda s, l=[[None]]: (l, [push_tok(l, t) for t in s]))(s)[0][1:]
merge_str = lambda ls: "".join(["{}{}".format(len(l), l[0]) for l in ls])
look_and_say = lambda s: merge_str(split_str(s))  # My Version
look_and_say = lambda s: "".join(["{}{}".format(len(list(g)), k) for k, g in itertools.groupby(s)])  # Using Itertools

recall = lambda f, v, times=1: reduce(lambda a, e: f(a), range(times), v)

look_and_say_repeat = lambda v, n: reduce(lambda a, e: "".join(["{}{}".format(len(l), l[0]) for l in (lambda s, l=[[None]]: (l, [l[-1].append(t) if l[-1][0] == t else l.append([t]) for t in s]))(a)[0][1:]]), range(n), v)
look_and_say_repeat = lambda v, n: reduce(lambda a, e: "".join(["{}{}".format(len(list(g)), k) for k, g in itertools.groupby(a)]), range(n), v)

next_sequence = lambda s: "".join((lambda m: m(m, s, 1))(lambda m, s, carry=0: [] if len(s) == 0 else m(m, s[:-1], s[-1] == 'z' and carry == 1) + [chr(((ord(s[-1]) - ord('a') + carry) % 26) + ord('a'))]))
asc_seq = lambda cs: all([(ord(cs[i]) + 1) == ord(cs[i + 1]) for i in range(len(cs) - 1)])
contains_asc_seq = lambda s, l=3: any([asc_seq(s[i:i + l]) for i in range(len(s) - l + 1)])

valid_sequence = lambda s: not any(map(lambda c: c in 'iol', s)) and contains_asc_seq(s) and re.match("^.*(.)\\1.*(.)\\2.*$", s) is not None


def next_password(s):
    s = next_sequence(s)
    while not valid_sequence(s):
        s = next_sequence(s)
        print s, "\r",
    return s


def flatten(v, ignore_values=None):
    if isinstance(v, int):
        print v, "\r",
        yield v
    elif isinstance(v, list):
        for e in v:
            for _e in flatten(e, ignore_values):
                yield _e
    elif isinstance(v, dict):
        if ignore_values is None or ignore_values not in v.values():
            for e in v.itervalues():
                for _e in flatten(e, ignore_values):
                    yield _e


def json_all_numbers(i, ignore_values=None):
    raw = json.loads(i)
    for num in flatten(raw, ignore_values):
        yield num

solutions = [
    lambda *i: None,
    lambda *i: (sum([{"(": 1, ")": -1}[c] for c in filter(lambda e: e in "()", i[0])]),
                (lambda m, i: m(m, i))((lambda m, r, f=0, s=0: s if f < 0 or len(r) == 0 else m(m, r[1:], f + ({"(": 1, ")": -1}[r[0]]), s + 1)), filter(lambda e: e in "()", i[0]))
                ),
    lambda *i: (sum(map(lambda (l, w, h): ((2 * l * w) + (2 * w * h) + (2 * h * l)) + min(l * w, w * h, h * l), map(lambda l: map(int, l.split("x")), i[0].splitlines()))),
                sum(map(lambda (l, w, h): min(((2 * l) + (2 * w)), ((2 * l) + (2 * h)), ((2 * h) + (2 * w))) + (l * w * h), map(lambda l: map(int, l.split("x")), i[0].splitlines()))),
                ),
    lambda *i: (len(collections.Counter(walk(i[0]))),
                len(collections.Counter(walk(i[0][::2]) + walk(i[0][1::2])))
                ),
    lambda *i: (mine(i[0]),
                mine(i[0], zeros=6)
                ),
    lambda *i: (sum([(not any(map(lambda p: p in line, ["ab", "cd", "pq", "xy"]))) and (len(filter(lambda c: c in "aeiou", line)) >= 3) and (any(map(lambda p: p in line, [c * 2 for c in "abcdefghijklmnopqrstuvwxyz"]))) for line in map(str.strip, i[0].splitlines())]),
                sum([re.match("^.*(.).\\1.*$", line) is not None and re.match("^.*(..).*\\1.*$", line) is not None for line in map(str.strip, i[0].splitlines())]),
                ),
    lambda *i: (collections.Counter(access_lights(build_lights(), parse_light_switches(i[0])).itervalues())[True],
                sum(access_lights(build_lights(), parse_light_switches(i[0]), funcs={"toggle": lambda v: v + 2, "on": lambda v: v + 1, "off": lambda v: v - 1}, after=lambda v: max(v, 0)).itervalues())
                ),
    lambda *i: (resolve_logic(wire_gates(init_gates(parse_logic(i[0]))))['a'].value,
                resolve_logic(wire_gates(update_gate(init_gates(parse_logic(i)), "b", resolve_logic(wire_gates(init_gates(parse_logic(i[0]))))['a'].value)))['a'].value
                ),
    lambda *i: (sum(map(len, i[0].splitlines())) - sum(map(len, map(unescape, map(lambda s: s[1:-1], i[0].splitlines())))),
                sum(map(len, map(escape, i[0].splitlines()))) - sum(map(len, i[0].splitlines()))
                ),
    lambda *i: (shortest_route(i[0]),
                longest_route(i[0]),
                ),
    lambda i, t1=40, t2=50: (len(look_and_say_repeat(i, int(t1))),
                             len(look_and_say_repeat(i, int(t2))),
                             ),
    lambda *i: (next_password(i[0]),
                next_password(next_password(i[0])),
                ),
    lambda *i: (sum(json_all_numbers(i[0])),
                sum(json_all_numbers(i[0], ignore_values="red")),
                )
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = (inputs[args.day],) if args.input is None else args.input
        print tasks[args.day]
        print "Solution:"
        print solutions[args.day](*input)
