#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import sys
import argparse
import time
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
parser.add_argument("--parts", type=int, nargs="+", default=[1, 2])


walk = lambda route, x_step={"^": 0, "v": 0, "<": -1, ">": 1}, y_step={"^": -1, "v": 1, "<": 0, ">": 0}: (
    (lambda history=[(0, 0)]: ((
        history,
        [
            history.append(
                (history[-1][0] + x_step[step], history[-1][1] + y_step[step])
            ) for step in route
        ]
    )[0]))()
)

mine = lambda key, zeros=5, state={"ready": False}: (
    (lambda z: (
        state.update(ready=False),
        reduce(
            lambda (k, p, d), e: (
                k, e, ((lambda digest: (
                    digest,
                    sys.stdout.write(digest + "\r") if e % 10000 == 0 else None,
                    state.update(ready=digest.startswith(z)),
                ))(MD5.new(k + str(e)).hexdigest()))[0],
            ),
            (i for i in xrange(10000000) if not state['ready']),
            (key, 0, None)
        )
    )[0])("0" * zeros)
)


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

unescape = lambda s, state={"counter": 0}: (
    (lambda counter, chs: (
        (chs, state.update(counter=0), [chs.append(
            (
                (ord(s[p]), state.update(counter=p + 1))
                if s[p] != "\\"
                else (
                    (int(s[p + 2:p + 4], 16), state.update(counter=p + 4))
                    if s[p + 1] == "x"
                    else (ord(s[p + 1]), state.update(counter=p + 2))
                )
            )[0]
        ) for p in counter])[0]
    ))(
        (state['counter'] for _ in xrange(len(s)) if state['counter'] < len(s)),
        list()
    )
)

escape = lambda s: '"{}"'.format("".join(["\\{}".format(c) if c in ['"', '\\'] else c for c in s]))


def transform_to_gv(input, fname="graph.gv"):
    with open(fname, "w") as output:
        output.write("graph g {\n")
        for line in input.splitlines():
            parts = line.split()
            output.write("  {} -- {} [label={}]".format(*parts[::2]))
        output.write("}\n")


brute_force_dict = lambda input, key_gen, val_gen, dict_prep, predicate, quality, permutation, init: (
    (lambda d: reduce(
        lambda a, e: predicate((quality(e, d), e), a),
        permutation(d),
        init
    ))({
        key_gen(parts): val_gen(parts)
        for parts in dict_prep(input.splitlines())
    })
)

find_route = lambda input, predicate, init: (
    brute_force_dict(
        input,
        lambda parts: tuple(sorted(parts[:3:2])),
        lambda parts: int(parts[4]),
        lambda lines: map(str.split, lines),
        predicate,
        lambda e, d: sum(d[tuple(sorted(e[c:c + 2]))] for c in range(len(e) - 1)),
        lambda d: itertools.permutations(set(list(sum(d.keys(), ())))),
        init,
    )
)

longest_route = lambda input: find_route(input, max, (0, None))
shortest_route = lambda input: find_route(input, min, (sys.maxint, None))

look_and_say_repeat = lambda v, n: reduce(
    lambda a, e: "".join("{}{}".format(len(list(g)), k) for k, g in itertools.groupby(a)),
    range(n),
    v
)

next_password = lambda previous: (
    (lambda generator, validator, state={"ready": False}:
        # Abusing reduce to create a loop and not run out of stack space
        # I hope that 1 million candidates are enough
        reduce(lambda a, e: (lambda new: (new, state.update(ready=validator(new)))[0])(generator(a)),
               (None for _ in xrange(1000000) if not state['ready']),
               previous
               )
        # # Y-Combinator for iterating possible candidates
        # (lambda f: f(f, previous))(
        #     # Using the inner lambda as closure for generated candidate
        #     lambda self, candidate: (lambda pw: pw if validator(pw) else self(self, pw))(generator(candidate))
        # )
     )(
        # Password sequence generator
        lambda sequence: "".join(
            (lambda m: m(m, sequence, 1))(
                lambda m, s, carry=0: (
                    [] if len(s) == 0 else m(m, s[:-1], s[-1] == 'z' and carry == 1) + [chr(((ord(s[-1]) - ord('a') + carry) % 26) + ord('a'))]
                )
            )
        ),
        # Password validator
        lambda sequence: (lambda *predicates: all(map(lambda p: p(sequence), predicates)))(
            # Passwords may not contain the letters i, o, or l, as these letters can
            # be mistaken for other characters and are therefore confusing.
            (lambda s: not any(map(lambda c: c in 'iol', s))),
            # Passwords must include one increasing straight of at least three
            # letters, like abc, bcd, cde, and so on, up to xyz. They cannot skip
            # letters; abd doesn't count.
            (lambda s, l=3: any(
                (lambda cs: all((ord(cs[i]) + 1) == ord(cs[i + 1]) for i in range(len(cs) - 1)))(s[i:i + l])
                for i in range(len(s) - l + 1)
            )),
            # Passwords must contain at least two different, non-overlapping pairs
            # of letters, like aa, bb, or zz.
            (lambda s: re.match("^.*(.)\\1.*(.)\\2.*$", s) is not None)
        )
    )
)

# First Combinator, defines a common closure for all four fold-functions
sum_up = lambda input, ignore=None: (lambda fe, fl, fd, fv: fe(fe, fl, fd, fv, json.loads(input), ignore))(
    # Fold Element: Select the appropriate fold-function based on the elements type
    #               Uses dict.get to define default behaviour for unknown types
    (lambda fe, fl, fd, fv, v, i, t="e": {int: fv, list: fl, dict: fd}.get(type(v), lambda fe, fl, fd, fv, v, i: 0)(fe, fl, fd, fv, v, i)),
    # Fold list: Combinator for folding all elements of the list
    (lambda fe, fl, fd, fv, v, i, t="l": (lambda f: f(f, v, 0, i))(lambda f, l, t, i: t if len(l) == 0 else f(f, l[1:], t + fe(fe, fl, fd, fv, l[0], i), i))),
    # Fold dict: Combinator for folding all values of a dict,
    #            except when one of the value is to be ignored
    (lambda fe, fl, fd, fv, v, i, t="d": (lambda f: f(f, v.items(), 0, i))(lambda f, d, t, i: t if len(d) == 0 else (0 if d[0][1] == ignore else f(f, d[1:], t + fe(fe, fl, fd, fv, d[0][1], i), i)))),
    # Fold value: To have the same interface for all types
    (lambda fe, fl, fd, fv, v, i, t="v": v)
)

happy_place = lambda input, add_keys=[]: (
    brute_force_dict(
        input,
        lambda (k, g): k,
        lambda (k, g): dict(map(lambda e: (e[2], e[1]), g)),
        lambda lines: itertools.groupby(
            map(
                lambda l: (lambda *p: (p[0], int(p[3]) if p[2] == "gain" else -int(p[3]), p[-1][:-1]))(*l.split()),
                lines
            ),
            lambda k: k[0]
        ),
        max,
        lambda sitting, people: sum(
            people.get(sitting[p], {}).get(sitting[(p - 1) % len(sitting)], 0) + people.get(sitting[p], {}).get(sitting[(p + 1) % len(sitting)], 0)
            for p in range(len(sitting))
        ),
        lambda people: itertools.permutations(people.keys() + add_keys),
        (0, None),
    )
)

fastest_reindeer = lambda input, times=1, points=False: (
    (lambda reindeers, move, round, distance, position: (
        {True: position, False: distance}[points](round, reindeers, move)
    ))(
        {  # Reindeer: Name is key, value is (speed, flying time, resting time)
            parts[0]: (int(parts[3]), int(parts[6]), int(parts[13]))
            for parts in map(str.split, input.splitlines())
        },
        # move a single reindeer
        lambda speed, fly_t, rest_t, total_t=1: (
            ((fly_t * (total_t / (fly_t + rest_t))) + \
                (min(total_t % (fly_t + rest_t), fly_t))) * speed
        ),
        # Move all reindeers, and select the best (most distance) one
        lambda rs, m, t: max(
            (m(*r, total_t=t), name)
            for name, r in rs.iteritems()
        ),
        # Distance scoring: select the furthest reindeer
        lambda r, rs, m: r(rs, m, times),
        # Point scoring: select the reindeer longest time in the lead
        lambda r, rs, m: (
            collections.Counter(
                r(rs, m, t)[1]
                for t in range(1, times + 1)
            ).most_common(1)[0]
        )
    )
)


best_ingredients = lambda input, spoons=100, calorie_value=0, keys=["capacity", "durability", "flavor", "texture"]: (
    (lambda ingredients, generator, value:
        reduce(
            lambda a, e: max((value(ingredients, e), e), a),
            generator(ingredients.keys()),
            (0, None)
        )
     )(
        {
            name: {attr: int(val) for attr, val in map(str.split, values.split(","))}
            for name, values in map(lambda l: l.split(":"), input.splitlines())
        },
        # Generator for all permutations of ingredient mixtures
        # yields dictionaries with ingredients as keys, and amounts as values.
        lambda ingredients: [
            (yield {
                i: b - a - 1
                for i, a, b in zip(ingredients, (-1,) + c, c + (spoons + len(ingredients) - 1,))
            })
            for c in itertools.combinations(range(spoons + len(ingredients) - 1), len(ingredients) - 1)
        ],
        # Calculate values of all ingredients, returns 0 if the calorie value is not matched
        lambda ingredients, amounts:
            reduce(lambda a, e: a * e, (
                max(0, sum(
                    ingredients[ingredient][key] * amount
                    for ingredient, amount in amounts.iteritems()
                ) if calorie_value == 0 or sum(
                    ingredients[ingredient]["calories"] * amount
                    for ingredient, amount in amounts.iteritems()
                ) == calorie_value else 0)
                for key in keys
            ), 1),
    )
)

sue_who = lambda input, ticker={"children": 3,
                                "cats": 7,
                                "samoyeds": 2,
                                "pomeranians": 3,
                                "akitas": 0,
                                "vizslas": 0,
                                "goldfish": 5,
                                "trees": 3,
                                "cars": 2,
                                "perfumes": 1,
                                }, greater=(), lesser=(): (
    (lambda list_of_sue, ticker_set, compare_sets: (
        filter(
            lambda (nr, sue): compare_sets(sue, ticker_set(ticker)),
            enumerate(list_of_sue, start=1)
        )
    ))(
        (
            frozenset((fact, int(count)) for fact, count in map(lambda f: map(str.strip, f.split(":", 1)), facts.split(",")))
            for sue, facts in map(lambda l: map(str.strip, l.split(":", 1)), input.splitlines())
        ),
        # Generate a set from the ticker output
        lambda t: frozenset((k, v) for k, v in t.iteritems()),
        # Manual set comparison, since we need fuzzy comparison for part 2
        # The inner lambda is used only if lesser or greater are set.
        #  It act's as a closure for the counters.
        lambda s, t: (lambda sc, st, compare: (
            all(compare(k, sc[k], st[k]) for k in st if k in sc)
        ))(
            collections.Counter({k: v for k, v in s}),
            collections.Counter({k: v for k, v in t}),
            lambda k, l, r: l > r if k in greater else l < r if k in lesser else l == r,
        ) if (len(greater) + len(lesser)) > 0 else s <= t
    )
)

eggnog_bottles = lambda input, volume=150, min_only=False: (
    (lambda bottles, filled, find_all, find_min: (
        {False: find_all, True: find_min}[min_only](bottles, filled)
    ))(
        map(int, input.splitlines()),
        # Generate all possible lists of containers filling exactly the volume
        lambda bs: (
            b for b in itertools.chain(*(
                itertools.combinations(bs, i) for i in xrange(len(bs))
            )) if sum(b) == volume),
        # Part 1: Only count the all matching possibilities
        lambda bs, fill: len(list(fill(bs))),
        # Part 2: Count only possibilities with a minimum number of containers
        lambda bs, fill: (lambda m: (
            len(filter(lambda c: len(c) == m, fill(bs)))
        ))(min(map(len, fill(bs))))
    )
)

# Implementing the game of life in a single line
game_of_light = lambda input, rounds=100, rules={
    "#": lambda neighbors: (
        "#" if len(filter(lambda e: e == "#", neighbors)) in [2, 3] else "."
    ),
    ".": lambda neighbors: (
        "#" if len(filter(lambda e: e == "#", neighbors)) in [3] else "."
    )
}, stuck = {}: (
    (lambda lights, neighbors, step, count, render: (
        count(reduce(
            lambda a, e: step(a, neighbors),
            (i for i in xrange(rounds)),
            {c: stuck[c] if c in stuck else v for c, v in lights.iteritems()}
        ))
    ))(
        {  # Read the field
            (x, y): c
            for y, line in enumerate(input.splitlines())
            for x, c in enumerate(line)
        },
        # List of all neighbors of a single cell
        lambda field, (x, y): filter(
            lambda e: e is not None,
            (field.get(neighbor, None)
             for neighbor in ((x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                              (x - 1, y + 0),             (x + 1, y + 0),
                              (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)))
        ),
        # Advance the field one step
        lambda field, neighbors: {
            coord: rules[value](neighbors(field, coord)) if coord not in stuck else stuck[coord]
            for coord, value in field.iteritems()
        },
        # Count specific cells in the field
        lambda field, value="#": len(filter(lambda v: v == value, field.itervalues())),
        # Render function writes the field to a stream
        lambda field, stream: (lambda width, height: [
            stream.write(
                "{}\n".format(
                    "".join(field[(x, y)] for x in xrange(width + 1))
                )
            ) for y in xrange(height + 1)
        ])(*max(field.keys())),
    )
)

alchemy = lambda input, calibrate=True: (
    (lambda (replacements, molecule), calibration, lookup: (
        {True: calibration, False: lookup}[calibrate](replacements, molecule)
    ))(
        (lambda repl, mol: (
            map(lambda l: l.split(" => "), repl.splitlines()),
            map("".join, reduce(
                lambda a, i: (
                    a,
                    a.append([mol[i]]) if mol[i].isupper() else a[-1].append(mol[i])
                )[0],
                (i for i in xrange(len(mol))),
                []
            ))
        ))(*input.split("\n\n", 1)),
        lambda repls, mol: (
            (lambda repl_group: (
                len(reduce(
                    lambda s, (p, a): (
                        s,
                        s.update(set(
                            "".join(mol[:p] + [rep] + mol[p + 1:])
                            for rep in repl_group.get(a, [])
                        ))
                    )[0],
                    ((pos, atom) for pos, atom in enumerate(mol)),
                    set()
                ))
            ))({
                k: list(map(lambda e: e[1], g))
                for k, g in
                itertools.groupby(repls, lambda e: e[0])
            })
        ),
        lambda repls, mol: (
            (lambda inv_repl, state={"done": False}: (
                reduce(
                    lambda best, keys: ((
                        lambda candidate: (
                            state.update(done=candidate[0] == "e"),
                            min(
                                [candidate, best],
                                key=lambda e: e[1]
                            ) if candidate[0] == "e" else best
                        )[-1]
                    )((
                        sys.stdout.write("{:80}\r".format("")),
                        sys.stdout.write("{:80}\r".format(best[1])),
                        state.update(atoms=keys),
                        reduce(
                            lambda m, i: (
                                (
                                    (
                                        (m[0].replace(state['atoms'][0], inv_repl[state['atoms'][0]], 1), m[1] + 1),
                                        state.update(atoms=keys),
                                    )[0] if state['atoms'][0] in m[0] else (
                                        m,
                                        state.update(atoms=state['atoms'][1:]),
                                    )[0],
                                )[0]
                            ),
                            (i for i in xrange(10000) if len(state['atoms']) > 0),
                            ("".join(mol), 0)
                        )
                    )[-1])),
                    (lambda gen: [
                        (yield list(reversed(next(gen))))
                        if not state['done'] else None
                        for _ in xrange(100000)
                    ])(itertools.permutations(inv_repl.keys())),
                    # (list(reversed(p)) for p in itertools.permutations(inv_repl.keys()) if not state['done']),
                    (None, sys.maxint)
                )
            ))(
                {v: k for k, v in repls}
            )
        )
    )
)

sieve_of_elves = lambda input, count=10, max_houses=sys.maxint, skip=100: (
    (lambda presents, houses, house_value, prime_factors, primes, divisors, state={}: (
        (lambda left, right: (
            state.update(done=False),
            reduce(
                lambda _, house: (lambda value: (
                    state.update(done=value * count >= presents),
                    # I like watching the script work, even if it means a
                    # slowdown
                    sys.stdout.write("{:09}: {:<20}\r".format(house, value)),
                    (house, value * count)
                )[-1])(house_value(house, divisors, prime_factors, primes)),
                (house for house in xrange(left, right)
                    if not state['done'] and house % skip == 0),
                None,
            )
        ))(  # Get the search range
            presents / count / count * 2, presents / count
        )
    )[-1])(
        int(input),
        # Calculate the number of presents delievered to a house
        lambda houses: sum(elf for elf in xrange(1, houses) if houses % elf == 0) * count,
        # Calculate the value of a house by generating adding divisors
        lambda house, div, prime_factors, primes, accumulate=sum: accumulate(
            itertools.ifilter(
                lambda d: house / d <= max_houses,
                div(collections.Counter(prime_factors(house, primes())).items())
            )
        ),
        # prime factorization using a sieve of Erathosthenes as generator
        lambda number, prime_gen: (
            (lambda factors, state=dict(prime=2, num=number): (
                reduce(
                    lambda num, prime: (
                        (factors.append(prime), num / prime)
                        if num % prime == 0
                        else (state.update(prime=next(prime_gen), num=num), num)
                    )[-1],
                    (state['prime'] for _ in xrange(number) if state["num"] > 1),
                    number,
                ),
                factors
            )[-1])(list())
        ),
        # Sieve of Erathosthenes for prime number generation
        lambda init=2: (
            # Close to crate a namespace for the number generator
            # and the sieves
            (lambda numbers, sieve, state={"current": None}: (
                (
                    state.update(current=sieve(numbers(init), init)),
                    (yield init),
                    # Main "loop": Each iteration will create a new sieve
                    # at the end if the chain.
                    [(yield (
                        (lambda prime: (
                            (
                                state.update(current=sieve(state['current'], prime)),
                                prime
                            )[-1]
                        ))(next(state['current']))
                    )) for _ in xrange(1, 10000)]
                )[-1]
            ))(
                # Number generator: Generates all an "infinite" number generator
                # Acts as initial "sieve"
                lambda num: (i for i in xrange(num, 1000000)),
                # A single sieve: Query parent sieve for the next candidate
                # Checks weither the candidate is divisible by its own prime
                lambda parent, num: [(yield candidate) if candidate % num != 0 else None for candidate in parent],
            )
        ),
        # Get all divisors for given number, based on its prime factors
        lambda primes: (
            sorted(itertools.imap(
                lambda facs: reduce(
                    lambda prod, fac: fac * prod,
                    facs,
                    1
                ),
                itertools.product(*([n**i for i in range(0, e + 1)] for n, e in primes))
            ))
        )
    )
)


boss_fight = lambda input, hitpoints=100, shop={
    "Weapons": {
        "Dagger":     {"Cost":  8,  "Damage": 4, "Armor": 0},
        "Shortsword": {"Cost": 10,  "Damage": 5, "Armor": 0},
        "Warhammer":  {"Cost": 25,  "Damage": 6, "Armor": 0},
        "Longsword":  {"Cost": 40,  "Damage": 7, "Armor": 0},
        "Greataxe":   {"Cost": 74,  "Damage": 8, "Armor": 0},
    },
    "Armor": {
        "Plain":      {"Cost":   0, "Damage": 0, "Armor": 0},
        "Leather":    {"Cost":  13, "Damage": 0, "Armor": 1},
        "Chainmail":  {"Cost":  31, "Damage": 0, "Armor": 2},
        "Splintmail": {"Cost":  53, "Damage": 0, "Armor": 3},
        "Bandedmail": {"Cost":  75, "Damage": 0, "Armor": 4},
        "Platemail":  {"Cost": 102, "Damage": 0, "Armor": 5},
    },
    "Rings": {
        "Copper":     {"Cost":   0, "Damage": 0, "Armor": 0},
        "Steel":      {"Cost":   0, "Damage": 0, "Armor": 0},
        "Damage +1":  {"Cost":  25, "Damage": 1, "Armor": 0},
        "Damage +2":  {"Cost":  50, "Damage": 2, "Armor": 0},
        "Damage +3":  {"Cost": 100, "Damage": 3, "Armor": 0},
        "Defense +1": {"Cost":  20, "Damage": 0, "Armor": 1},
        "Defense +2": {"Cost":  40, "Damage": 0, "Armor": 2},
        "Defense +3": {"Cost":  80, "Damage": 0, "Armor": 3},
    }
}, crook=False: (
    (lambda boss, hero, inventory, fight, round: (
        reduce(
            lambda best, items: (lambda (result, h): (
                {True: max, False: min}[crook]([(h['Cost'], h['Items'], h['Hitpoints']), best], key=lambda e: e[0])
                if result is not crook else best
            ))(fight(boss, hero(*items), round)),
            inventory(),
            (sys.maxint if not crook else 0, [], 0)
        )
    ))(
        {
            key.replace(" ", "").title(): int(val) for key, val in
            map(lambda l: l.split(": "), input.splitlines())
        },
        # Build Player based on items
        lambda *items: dict(
            Items=[name for name, item in items],
            Hitpoints=hitpoints,
            Damage=sum(item['Damage'] for name, item in items),
            Armor=sum(item['Armor'] for name, item in items),
            Cost=sum(item['Cost'] for name, item in items)
        ),
        # Shopping Time: Pick 1 Weapon, 1 Armor and 2 Rings
        lambda: (items for items in itertools.product(
            shop['Weapons'].items(),
            shop['Armor'].items(),
            shop['Rings'].items(),
            shop['Rings'].items()
        ) if items[2][0] != items[3][0]),
        # The fight!
        lambda boss, hero, round: (
            (lambda f: f(f, round, boss, hero))(
                lambda f, r, b, h: (
                    (
                        # sys.stdout.write("Hero deals {}-{}={} damage; The boss goes down to {} hit points\n".format(
                        #     h['Damage'], b['Armor'],
                        #     h['Damage'] - b['Armor'],
                        #     b['Hitpoints'] - (h['Damage'] - b['Armor']))
                        # ),
                        # sys.stdout.write("Boss deals {}-{}={} damage; The hero goes down to {} hit points\n".format(
                        #     b['Damage'], h['Armor'],
                        #     b['Damage'] - h['Armor'],
                        #     h['Hitpoints'] - (b['Damage'] - h['Armor']))
                        # ),
                        f(f, r, *r(b, h)),
                    ) if b['Hitpoints'] > 0 and h['Hitpoints'] > 0
                    else ((h['Hitpoints'] > 0, h),)
                )[-1]
            )
        ),
        # A single round of the battle
        lambda boss, hero: (
            dict(Hitpoints=boss['Hitpoints'] - (hero['Damage'] - boss['Armor']),
                 Armor=boss['Armor'],
                 Damage=boss['Damage'],
                 ),
            dict(Hitpoints=hero['Hitpoints'] - (boss['Damage'] - hero['Armor']),
                 Armor=hero['Armor'],
                 Damage=hero['Damage'],
                 Cost=hero['Cost'],
                 Items=hero['Items'],
                 ),
        ),
    )
)

wizard_duel = lambda input, hitpoints=50, mana=500, spells={
    "Magic Missile": {"Mana":  53, "Duration": 0, "Damage": 4, "Armor": 0, "Heal": 0, "Gain":   0},
    "Drain":         {"Mana":  73, "Duration": 0, "Damage": 2, "Armor": 0, "Heal": 2, "Gain":   0},
    "Shield":        {"Mana": 113, "Duration": 6, "Damage": 0, "Armor": 7, "Heal": 0, "Gain":   0},
    "Poison":        {"Mana": 173, "Duration": 6, "Damage": 3, "Armor": 0, "Heal": 0, "Gain":   0},
    "Recharge":      {"Mana": 229, "Duration": 5, "Damage": 0, "Armor": 0, "Heal": 0, "Gain": 101},
}, drain=0: (
    (lambda boss, hero, cast, attack, apply_effects, round, fight, state={}: (
        # sys.stdout.write("Minimum Spells: {}, Maximum Spells: {}\n".format(
        #     boss['Hitpoints'] / sum(s['Damage'] for s in spells.itervalues()),
        #     hitpoints / (boss['Damage'] - sum(s['Armor'] for s in spells.itervalues()) + drain)
        # )),
        state.update(min_spells=hitpoints / (boss['Damage'] - sum(s['Armor'] for s in spells.itervalues())) + drain),
        reduce(
            lambda _, spell_count: reduce(
                lambda best, spell_list: (
                    # sys.stdout.write(
                    #     "I prepared {:2} spells this morning: {}. Minimum Mana: {} for {} Spells.".format(
                    #         len(spell_list),
                    #         "".join(s[0] for s in spell_list),
                    #         best[0],
                    #         state['min_spells'],
                    #     )
                    # ),
                    (lambda (result, b, h): (
                        state.update(min_spells=state['min_spells'] if not result else min(state['min_spells'], len(spell_list))),
                        (
                            min([(sum(spells[s]['Mana'] for s in spell_list), spell_list, b, h),
                                 best
                                 ], key=lambda e: e[0])
                        ) if result else best,
                    )[-1])(fight(boss, hero, spell_list, round, cast, attack, apply_effects)),
                )[-1],
                (spell_list for spell_list in itertools.ifilter(
                    lambda spls: all(
                        (spells[spls[i]]['Duration'] == 0 or
                         spls[i] not in spls[i + 1: i + 1 + (spells[spls[i]]['Duration'] / 2)])
                        for i in xrange(len(spls))
                    ),
                    itertools.product(spells, repeat=spell_count)
                )),
                (sys.maxint, [], {}, {})
            ),
            (spell_count for spell_count in xrange(
                boss['Hitpoints'] / sum(s['Damage'] for s in spells.itervalues()),
                hitpoints / (boss['Damage'] - sum(s['Armor'] for s in spells.itervalues()) + drain)
            ) if spell_count <= state['min_spells']),
            None,
        )
    )[-1])(
        {
            key.replace(" ", "").title(): int(val) for key, val in
            map(lambda l: l.split(": "), input.splitlines())
        },
        {"Mana": mana, "Hitpoints": hitpoints, "Spells": []},
        # Hero casts a spell
        # Casting a spell that is already active will simply result in
        # having no effect
        lambda b, h, s: (
            # sys.stdout.write("Hero casts {}\n".format(s)),
            (
                dict(Hitpoints=b['Hitpoints'],
                     Damage=b['Damage'],
                     ),
                dict(Hitpoints=h['Hitpoints'] - drain,
                     Mana=h['Mana'] - spells[s]['Mana'],
                     Spells=[spell for spell in h['Spells']] + ([(s, spells[s]["Duration"])] if spells[s]['Mana'] <= h['Mana'] else []),
                     ),
            ),
        )[-1],
        # Boss attacks the hero
        lambda b, h: (
            # sys.stdout.write("Boss attacks for {} damage\n".format(max(1, b['Damage'] - sum(spells[e]['Armor'] for e, d in h['Spells'])))),
            (
                dict(Hitpoints=b['Hitpoints'],
                     Damage=b['Damage'],
                     ),
                dict(Hitpoints=h['Hitpoints'] - max(1, b['Damage'] - sum(spells[e]['Armor'] for e, d in h['Spells'])),
                     Mana=h['Mana'],
                     Spells=[spell for spell in h['Spells']],
                     ),
            ) if b['Hitpoints'] > 0 else (b, h),
        )[-1],
        # Magic effects are applied to both combatants
        lambda b, h: (
            dict(Hitpoints=b['Hitpoints'] - sum(spells[e]['Damage'] for e, d in h['Spells']),
                 Damage=b['Damage'],
                 ),
            dict(Hitpoints=h['Hitpoints'] + sum(spells[e]['Heal'] for e, d in h['Spells']),
                 Mana=h['Mana'] + sum(spells[e]['Gain'] for e, d in h['Spells']),
                 Spells=[(spell, duration - 1) for spell, duration in h['Spells'] if duration > 1],
                 ),
        ),
        # A single round of battle
        lambda b, h, s, cst, atk, eff: (
            reduce(
                lambda (b, h), f: (
                    # sys.stdout.write("Boss {}\n".format(b)),
                    # sys.stdout.write("Hero {}\n".format(h)),
                    f(b, h),
                )[-1],
                [
                    # Hero's turn
                    lambda b, h: eff(b, h),
                    lambda b, h: cst(b, h, s),
                    # Bosses turn
                    lambda b, h: eff(b, h),
                    lambda b, h: atk(b, h),
                ],
                (b, h)
            )
        ),
        # Fight!
        lambda b, h, sps, rnd, cst, atk, eff: (
            (lambda f: f(f, sps, b, h))(
                lambda f, sps, b, h: (
                    f(f, sps[1:], *rnd(b, h, sps[0], cst, atk, eff))
                    if (len(sps) > 0 and
                        h['Mana'] >= 0 and
                        h['Hitpoints'] > 0 and
                        b['Hitpoints'] > 0
                        ) else (
                        # sys.stdout.write(" Winner: {:40}\r".format(
                        #     "Boss - The hero is out of mana" if h['Mana'] <= 0 else
                        #     ("Boss" if h['Hitpoints'] < 0 else
                        #      ("Hero" if b["Hitpoints"] < 0 else
                        #       "None"))
                        # )),
                        (b['Hitpoints'] <= 0 and h['Mana'] >= 0 and h['Hitpoints'] > 0, b, h),
                    )[-1]
                )
            )
        ),
    )
)

turing_lock = lambda input, a=0, b=0, c=0: (
    (lambda program, instructions, registers={'a': a, 'b': b, 'c': c}: (
        reduce(
            lambda regs, instr: (
                # sys.stdout.write("A {r[a]} B {r[b]} C {r[c]} -> ".format(r=registers)),
                # sys.stdout.write("{:4}{:5} => ".format(instr[0], " ".join(instr[1]))),
                registers.update(instructions[instr[0]](registers, *instr[1])),
                # sys.stdout.write("A {r[a]} B {r[b]} C {r[c]}\n".format(r=registers)),
                registers,
            )[-1],
            (program[registers['c']] for _ in xrange(10000) if 0 <= registers['c'] < len(program)),
            registers,
        )
    ))(
        [
            (op, params.split(", "))
            for op, params in map(lambda line: line.split(" ", 1), input.splitlines())
        ],
        {
            "hlf": lambda regs, r: dict(
                a=regs['a'] / (2 if r == 'a' else 1),
                b=regs['b'] / (2 if r == 'b' else 1),
                c=regs['c'] + 1,
            ),
            "tpl": lambda regs, r: dict(
                a=regs['a'] * (3 if r == 'a' else 1),
                b=regs['b'] * (3 if r == 'b' else 1),
                c=regs['c'] + 1,
            ),
            "inc": lambda regs, r: dict(
                a=regs['a'] + (1 if r == 'a' else 0),
                b=regs['b'] + (1 if r == 'b' else 0),
                c=regs['c'] + 1,
            ),
            "jmp": lambda regs, o: dict(
                a=regs['a'],
                b=regs['b'],
                c=regs['c'] + int(o),
            ),
            "jie": lambda regs, r, o: dict(
                a=regs['a'],
                b=regs['b'],
                c=regs['c'] + (int(o) if int(regs[r]) % 2 == 0 else 1),
            ),
            "jio": lambda regs, r, o: dict(
                a=regs['a'],
                b=regs['b'],
                c=regs['c'] + (int(o) if int(regs[r]) == 1 else 1),
            ),
        }
    )
)


solutions = [
    (lambda *i: None,
     lambda *i: None
     ),
    (lambda *i: sum([{"(": 1, ")": -1}[c] for c in filter(lambda e: e in "()", i[0])]),
     lambda *i: (lambda m, i: m(m, i))((lambda m, r, f=0, s=0: s if f < 0 or len(r) == 0 else m(m, r[1:], f + ({"(": 1, ")": -1}[r[0]]), s + 1)), filter(lambda e: e in "()", i[0]))
     ),
    (lambda *i: sum(map(lambda (l, w, h): ((2 * l * w) + (2 * w * h) + (2 * h * l)) + min(l * w, w * h, h * l), map(lambda l: map(int, l.split("x")), i[0].splitlines()))),
     lambda *i: sum(map(lambda (l, w, h): min(((2 * l) + (2 * w)), ((2 * l) + (2 * h)), ((2 * h) + (2 * w))) + (l * w * h), map(lambda l: map(int, l.split("x")), i[0].splitlines()))),
     ),
    (lambda *i: len(collections.Counter(walk(i[0]))),
     lambda *i: len(collections.Counter(walk(i[0][::2]) + walk(i[0][1::2])))
     ),
    (lambda *i: mine(i[0]),
     lambda *i: mine(i[0], zeros=6)
     ),
    (lambda *i: sum([(not any(map(lambda p: p in line, ["ab", "cd", "pq", "xy"]))) and (len(filter(lambda c: c in "aeiou", line)) >= 3) and (any(map(lambda p: p in line, [c * 2 for c in "abcdefghijklmnopqrstuvwxyz"]))) for line in map(str.strip, i[0].splitlines())]),
     lambda *i: sum([re.match("^.*(.).\\1.*$", line) is not None and re.match("^.*(..).*\\1.*$", line) is not None for line in map(str.strip, i[0].splitlines())]),
     ),
    (lambda *i: collections.Counter(access_lights(build_lights(), parse_light_switches(i[0])).itervalues())[True],
     lambda *i: sum(access_lights(build_lights(), parse_light_switches(i[0]), funcs={"toggle": lambda v: v + 2, "on": lambda v: v + 1, "off": lambda v: v - 1}, after=lambda v: max(v, 0)).itervalues())
     ),
    (lambda *i: resolve_logic(wire_gates(init_gates(parse_logic(i[0]))))['a'].value,
     lambda *i: resolve_logic(wire_gates(update_gate(init_gates(parse_logic(i)), "b", resolve_logic(wire_gates(init_gates(parse_logic(i[0]))))['a'].value)))['a'].value
     ),
    (lambda *i: sum(map(len, i[0].splitlines())) - sum(map(len, map(unescape, map(lambda s: s[1:-1], i[0].splitlines())))),
     lambda *i: sum(map(len, map(escape, i[0].splitlines()))) - sum(map(len, i[0].splitlines()))
     ),
    (lambda *i: shortest_route(i[0]),
     lambda *i: longest_route(i[0]),
     ),
    (lambda i, t1=40, t2=50: len(look_and_say_repeat(i, int(t1))),
     lambda i, t1=40, t2=50: len(look_and_say_repeat(i, int(t2))),
     ),
    (lambda *i: next_password(i[0]),
     lambda *i: next_password(next_password(i[0])),
     ),
    (lambda *i: sum_up(i[0]),
     lambda *i: sum_up(i[0], ignore="red"),
     ),
    (lambda *i: happy_place(i[0]),
     lambda *i: happy_place(i[0], add_keys=["self"]),
     ),
    (lambda i, times=2503: fastest_reindeer(i, times),
     lambda i, times=2503: fastest_reindeer(i, times, True),
     ),
    (lambda *i: best_ingredients(i[0]),
     lambda *i: best_ingredients(i[0], calorie_value=500),
     ),
    (lambda *i: sue_who(i[0]),
     lambda *i: sue_who(i[0], greater=("cats", "trees"), lesser=("pomeranians", "goldfish"))
     ),
    (lambda i, volume=150: eggnog_bottles(i, int(volume)),
     lambda i, volume=150: eggnog_bottles(i, int(volume), more=True),
     ),
    (lambda *i: game_of_light(i[0]),
     lambda *i: game_of_light(i[0], stuck={(0, 0): "#", (99, 0): "#", (0, 99): "#", (99, 99): "#"}),
     ),
    (lambda *i: alchemy(i[0]),
     lambda *i: alchemy(i[0], calibrate=False)
     ),
    (lambda i, count=10, max_house=sys.maxint, skip=100: sieve_of_elves(int(i), int(count), int(max_house), int(skip)),  # 831600
     lambda i, count=11, max_house=50, skip=10: sieve_of_elves(int(i), int(count), int(max_house), int(skip)),  #
     ),
    (lambda *i: boss_fight(i[0]),
     lambda *i: boss_fight(i[0], crook=True),
     ),
    (lambda *i: wizard_duel(i[0]),
     lambda *i: wizard_duel(i[0], drain=1),
     ),
    (lambda *i: turing_lock(i[0]),
     lambda *i: turing_lock(i[0], a=1),
     )
]

if __name__ == "__main__":
    args = parser.parse_args()

    if args.day <= len(solutions):
        input = (inputs[args.day],) if args.input is None else map(lambda i: i.replace("\\n", "\n"), args.input)
        for i, part in zip(range(len(solutions[args.day])), args.parts):
            print tasks[args.day][part - 1]
            print ""
            start = time.time()
            print solutions[args.day][part - 1](*input)
            print "{:.03f} sec".format(time.time() - start)
            print ""
