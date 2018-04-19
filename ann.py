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

    def __init__(self, con, act=lambda x: x, w = None):
        """
        Connections: con
        Activation Function: act
        """
        self.con = con
        self.act = np.vectorize(act)

        self.w = w

    def connect(self, pcon):
        """Set weigth matrix."""
        self.w = np.random.rand(self.con, pcon)

        return self

    def fprop(self, a):
        """Propagate activations forward."""
        return self.act(self.w @ a)


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
                
    def mutate(self, mutation, degree=0.4):
        """Mutate self degree amount."""
        for layer in self.network:
            ws = layer.w

            (h, w) = ws.shape
            for _ in range(int(ws.size * degree)):
                hinh = int(np.random.rand() * h)
                winh = int(np.random.rand() * w)

                ws[hinh][winh] = mutation(ws[hinh][winh])

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
            clay = Layer(layer.con, layer.act, wcopy)
            cnet.network.append(clay)

        return cnet
    
    def __iter__(self):
        return iter(self.network)


class Genetic(object):

    def __init__(self, family_sz, selection_bias=0.75, verbose=True,
                 mutation_chance=0.5, mutation_severity=0.4, inheritance=0.4):

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
                m_layer = Layer(layer.con, layer.act)
                member.add_layer(m_layer)

            family.append(member)

        self.family = family

    def evolve(self, evl):
        """
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

        return self


    def __iter__(self):
        return iter(self.family)

    
def selection(family, evl, sb, verbose):
    """
    Selection should be based on variation aswell as
    fitness, this current implementation only depends on fitness.

    A PR is welcome! :)
    """
    timestamp = time()

    e = list(enumerate(evl)) 
    e.sort(key = lambda x: x[1], reverse=True)
    sp = []
    for idx, (i, _) in enumerate(e):
        """
        Need a better solution for selecting a random element
        but without a uniform chance, this might use to much
        memory.
        """
        chance = int(10000 * pow(1 - sb, idx) * sb)
        sp.extend([i] * chance)

    """
    chance is based on rank space from

    https://www.youtube.com/watch?v=kHyNqSnzP8Y at around
    23:15
    """

    """Please make sure family is even."""
    half = int(len(family) / 2)


    """This might be abit slow."""
    s = []
    for _ in range(half):
        if len(sp) == 0:
            sys.stderr.write(ERR_MESSAGE)
            break

        choice = random.choice(sp)
        s.append(family[choice])

        sp = [x for x in sp if x != choice]

    if verbose:
        print("Selection: {}".format(time() - timestamp))

    return s


def crossmut(selection, mchance, msev, inh, verbose):
    """
    Double population again after selection
    by breeding children and applying mutation.

    Shuffle list to cause random breeding.
    """
    timestamp = time()
    random.shuffle(selection)

    b = []
    grps = len(selection)
    bgs = int(grps / 2)
    for i in range(bgs):
        j = i * 2

        if j + 1 < grps:
            b.append((selection[j], selection[j + 1]))

    if grps % 2 == 1:
        b.append((selection[0], selection[-1]))


    family = []
    """Don't mind the variable name choices."""
    for mom, dad in b:
        kid1 = mom.deep_copy()
        kid2 = dad.deep_copy()

        kid1.inheritws(dad, inh)
        kid2.inheritws(mom, inh)

        if np.random.rand() > mchance:
            kid1.mutate(mutation, msev)

        if np.random.rand() > mchance:
            kid2.mutate(mutation, msev)

        family.append(kid1)
        family.append(kid2)

    family.extend(selection)

    if verbose:
        print("Crossing and mutation: {}".format(time() - timestamp))

    return family


def mutation(x):
    """Mutation function."""
    return x * random.choice([1.5, 1 / 1.5]) + random.choice([1, -1])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))





























    




