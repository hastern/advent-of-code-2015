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


def escape(s):
    chs = []
    for c in s:
        if c in ['"', '\\']:
            chs.append("\\")
        chs.append(c)
    return '"{}"'.format("".join(chs))


def transform_to_gv(input, fname="graph.gv"):
    with open(fname, "w") as output:
        output.write("graph g {\n")
        for line in input.splitlines():
            parts = line.split()
            output.write("  {} -- {} [label={}]".format(*parts[::2]))
        output.write("}\n")


def build_net(input):
    routes = collections.defaultdict(dict)
    for line in input.splitlines():
        parts = line.split()
        routes[parts[0]][parts[2]] = int(parts[4])
        routes[parts[2]][parts[0]] = int(parts[4])
    return routes


def all_routes(net):
    for p in itertools.permutations(net.keys()):
        yield p


def route_length(net, route):
    l = 0
    src = route[0]
    for dst in route[1:]:
        l += net[src][dst]
        src = dst
    return l


def find_route(net, pred=min, init=(sys.maxint, None)):
    result = init
    for route in all_routes(net):
        result = pred((route_length(net, route), route), result)
    return result


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
               sum([re.match("^.*(.).\\1.*$", line) is not None and re.match("^.*(..).*\\1.*$", line) is not None for line in map(str.strip, i.splitlines())]),
               ),
    lambda i: (collections.Counter(access_lights(build_lights(), parse_light_switches(i)).itervalues())[True],
               sum(access_lights(build_lights(), parse_light_switches(i), funcs={"toggle": lambda v: v + 2, "on": lambda v: v + 1, "off": lambda v: v - 1}, after=lambda v: max(v, 0)).itervalues())
               ),
    lambda i: (resolve_logic(wire_gates(init_gates(parse_logic(inputs[7]))))['a'].value,
               resolve_logic(wire_gates(update_gate(init_gates(parse_logic(inputs[7])), "b", resolve_logic(wire_gates(init_gates(parse_logic(inputs[7]))))['a'].value)))['a'].value
               ),
    lambda i: (sum(map(len, i.splitlines())) - sum(map(len, map(unescape, map(lambda s: s[1:-1], i.splitlines())))),
               sum(map(len, map(escape, i.splitlines()))) - sum(map(len, i.splitlines()))
               ),
    lambda i: (find_route(build_net(inputs[9])),
               find_route(build_net(inputs[9]), pred=max, init=(0, None)),
               ),
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = inputs[args.day] if args.input is None else args.input
        print tasks[args.day]
        print "Solution:"
        print solutions[args.day](input)
