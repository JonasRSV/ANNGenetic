import numpy as np
import random
import math
from time import time
import sys


ERR_MESSAGE = "The random selection method i've used \
has failed, try to reduce the number of \
family members or increase the number at \
the chance variable in the selection function \
in the ann file or improve my selection function!"


class Layer(object):

    def __init__(self, con, activation=lambda x: x, w = None):
        """
        Connections: con
        Activation Function: act
        """
        self.con = con

        """Pre allocate vector, its major speedup."""
        self.palloc_mm = np.zeros(con)

        self.activation = activation
        self.w = w

    def connect(self, pcon):
        """Set weigth matrix."""
        self.w = np.random.rand(self.con, pcon)

        return self

    def fprop(self, activations):
        """Propagate activations forward."""
        self.palloc_mm[:] = self.w @ activations
        return self.activation(self.palloc_mm)


class ANN(object):

    def __init__(self, input_sz):
        self.isz = input_sz

        """Input for next layer."""
        self.tisz = input_sz
        self.network = []

    def add_layer(self, layer):
        """Add a layer to network."""
        self.network.append(layer.connect(self.tisz))
        self.tisz = layer.con

    def prop(self, i):
        """Propagate through network."""
        a = i
        for layer in self.network:
            a = layer.fprop(a)

        return a

    def inheritws(self, ann, degree=0.4):
        """Inherit degree amount of weigths."""
        for i, l in enumerate(ann):
            ws = l.w

            (h, w) = ws.shape
            for _ in range(int(ws.size * degree)):
                hinh = int(np.random.rand() * h)
                winh = int(np.random.rand() * w)

                self.network[i].w[hinh][winh] = ws[hinh][winh]

        return self

    def mutate(self, m, degree=0.4):
        """Mutate self degree amount."""
        for layer in self.network:

            (h, w) = layer.w.shape
            for _ in range(int(layer.w.size * degree)):
                hinh = int(np.random.rand() * h)
                winh = int(np.random.rand() * w)

                layer.w[hinh][winh] = m(layer.w[hinh][winh])

        return self

    def deep_copy(self):
        """
        Deep copy itself.

        used when breeding children so that
        mutability don't fuck shit up.
        """
        cnet = ANN(self.isz)
        for layer in self:
            wcopy = np.copy(layer.w)
            clay = Layer(layer.con, layer.activation, wcopy)
            cnet.network.append(clay)

        return cnet

    def __iter__(self):
        return iter(self.network)


class Genetic(object):

    def __init__(self, family_sz, selection_bias=0.5, verbose=True,
                 mutation_chance=0.9, mutation_severity=0.4, inheritance=0.4):

        self.family_sz = family_sz
        self.sb = selection_bias
        self.family = None
        self.mchance = mutation_chance
        self.msev = mutation_severity
        self.inh = inheritance
        self.verbose = verbose

        self.generation = 0

    def create_family(self, network):
        """
        Create family of networks.

        This family of members will be used in
        the selection.
        """
        family = []
        for _ in range(self.family_sz):
            member = ANN(network.isz)

            for layer in network:
                m_layer = Layer(layer.con, layer.activation)
                member.add_layer(m_layer)

            family.append(member)

        self.family = family

    def evolve(self, evl):
        """
        Evolve.

        It's hard to implement a general eval algorithm for
        all kinds of problems thus i believe it's easier
        if the user just provides the evaluations.

        this evaluation is commonly called the fitness function.
        """

        timestamp = time()

        s = selection(self.family,
                      evl,
                      self.sb,
                      self.verbose)

        self.family = crossmut(s,
                               self.mchance,
                               self.msev,
                               self.inh,
                               self.verbose)

        if self.verbose:
            print("Current Generation {}, evolution took: {}"
                  .format(self.generation, time() - timestamp))

        self.generation += 1

        return self

    def __iter__(self):
        return iter(self.family)


def selection(family, evl, sb, verbose):
    """
    Select new family.

    Selection should be based on variation aswell as
    fitness, this current implementation only depends on fitness.

    A PR is welcome! :)
    """
    timestamp = time()
    eo = list(enumerate(evl))

    """Evaluation Order."""
    eo.sort(key=lambda x: x[1], reverse=True)

    """Remove fitness not needed after order is decided."""
    eo = list(map(lambda x: x[0], eo))

    """Children to Select."""
    sts = int(len(eo) / 2)

    """Shelter the best from chance to avoid regression."""
    s = [family[eo[0]]]
    eo = eo[1:]

    """Prealloc. This is major speedup when family is HUGE."""
    items = len(eo)
    cache = [None] * len(eo)

    while sts:
        for i in range(items):

            """Rank Space."""
            chance = pow(1 - sb, i) * sb

            if i == items - 1:
                """Take best rather than worst."""
                s.append(family[eo[0]])
                cache = cache[1:]
                break

            elif np.random.rand() < chance:
                s.append(family[eo[i]])
                cache[i:] = eo[i + 1:]
                break

            else:
                cache[i] = eo[i]

        eo[:] = cache[:]
        items -= 1
        sts -= 1

    if verbose:
        print("Selection: {}".format(time() - timestamp))

    return s


def crossmut(selection, mchance, msev, inh, verbose):
    """
    Double population again after selection.
    by breeding children and applying mutation.

    Shuffle list to cause random breeding.
    """
    timestamp = time()

    """Maybe this is slow?."""
    random.shuffle(selection)

    """Select Breeding Pairs."""
    grps = len(selection)
    bgs = int(grps / 2)

    """Pre Alloc."""
    bps = [None] * int(grps / 2 + 0.5)
    for i in range(bgs):
        j = i * 2

        if j + 1 < grps:
            bps[i] = (selection[j], selection[j + 1])

    if bps[-1] is None:
        bps[-1] = (selection[0], selection[-1])

    """Breed."""
    family = []
    """Don't mind the variable name choices."""
    for mom, dad in bps:
        kid1 = mom.deep_copy()
        kid2 = dad.deep_copy()

        kid1.inheritws(dad, inh)
        kid2.inheritws(mom, inh)

        if np.random.rand() < mchance:
            kid1.mutate(mutation, msev)

        if np.random.rand() < mchance:
            kid2.mutate(mutation, msev)

        family.append(kid1)
        family.append(kid2)

    family.extend(selection)

    if verbose:
        print("Crossing and mutation: {}".format(time() - timestamp))

    return family


def mutation(x):
    """Mutation function."""
    return x + np.random.rand() * random.choice([1, -1])


def sigmoid(x):
    """Vectorized sigmoid."""
    return 1 / (1 + np.exp(-x))

