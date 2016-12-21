# -*- coding: utf-8 -*-

from random import randint
from fpylll import BKZ, Enumeration, EnumerationError, IntegerMatrix
#from fpylll import LLL, BKZ, GSO, Enumeration, EnumerationError, IntegerMatrix, prune
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.util import gaussian_heuristic
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from fpylll.algorithms.bkz_stats import dummy_tracer
from math import ceil
import time

YOLO_PREPROC_MIN_BLOCK_SIZE = 50
YOLO_PRUNER_MIN_BLOCK_SIZE = 50
YOLO_GAP_PREPROC_BLOCK_SIZE = 10
YOLO_MAX_BLOCK_SIZE = 200
YOLO_MEMORY_LENGTH = 6
GH_FACTOR = 1.1
NODE_PER_SEC = 2**26
RESTART_PENALTY = 0.01


class Timer:
    def __init__(self):
        self.start = time.clock()

    reset = __init__

    def elapsed(self):
        return time.clock() - self.start

DEFAULT_STRATEGIES = BKZ.Param(block_size=1, strategies="default.json").strategies


class Tuner(object):
    def __init__(self, b):
        self.last_prunings = None
        self.data = {}
        self.counts = {}
        self.b = b
        self.proba = .5
        if b > 1 and b < max(YOLO_PRUNER_MIN_BLOCK_SIZE, YOLO_PREPROC_MIN_BLOCK_SIZE):
            self.strategy = DEFAULT_STRATEGIES[b]

    def get_variations(self, preprocessing):
        V = [preprocessing]

        minb = 10
        if len(preprocessing) == 0:
            V.append(tuple([minb]))
            # V.append(tuple([(self.b/3, .5)]))
            return V
        if len(preprocessing) == 1:
            b = preprocessing[0]
            if b < minb + 6:
                V.append(tuple([]))
            for bb in reversed(range(max(b-2, minb), min(b+3, self.b - YOLO_GAP_PREPROC_BLOCK_SIZE))):
                V.append(tuple([bb]))
            return V
        assert False

    def preprocess(self):
        # return tuple()
        # self.count += 1
        if self.b < YOLO_PREPROC_MIN_BLOCK_SIZE:
            return self.strategy.preprocessing_block_sizes
        if len(self.data) == 0:
            return tuple()
        best = max(self.data, key=self.data.get)
        best_efficiency = self.data[best]
        variations = self.get_variations(best)
        for variation in variations:
            if variation not in self.data:
                return variation
            if self.counts[variation]**2 < self.counts[best]:
                return variation

        variation = variations[randint(0, len(variations)-1)]
        variation_efficiency = self.data[variation]
        # print self.b, best, variations
        ratio = best_efficiency / variation_efficiency
        p = ceil(ratio)
        if randint(0, p) == 0:
            return variation
        else:
            return best

    def enum(self, M, k, target_prob, preproc_time):
        b = self.b

        radius = M.get_r(k, k) * .99
        root_det = M.get_root_det(k, k + b - 1)
        gh_radius, ge = gaussian_heuristic(radius, 0, b, root_det, 1.)
        if b > 30:
            radius = min(radius, 1.21 * gh_radius * 2**ge)

        if b < YOLO_PRUNER_MIN_BLOCK_SIZE:
            return radius, self.strategy.get_pruning(radius, gh_radius * 2**ge)

        R = tuple([M.get_r(i, i) for i in range(k, k+b)])
        overhead = (preproc_time + RESTART_PENALTY) * NODE_PER_SEC
        start_from = self.last_prunings
        pruning = prune(radius, overhead, target_prob, [R],
                        descent_method="gradient", precision=53, start_from=start_from)
        self.last_prunings = pruning.coefficients
        self.proba = (self.proba * YOLO_MEMORY_LENGTH) + pruning.probability
        self.proba /= YOLO_MEMORY_LENGTH + 1
        return radius, pruning

    def enum_for_hints(self, M, k, b, preproc_time):
        return 0, None

    def feedback(self, preprocessing, pruning, time):
        if pruning is None:
            efficiency = 1. / time
        else:
            efficiency = pruning.probability / time
        if preprocessing in self.data:
            x = self.data[preprocessing]
            c = self.counts[preprocessing]
            f = min(c, YOLO_MEMORY_LENGTH)
            x = f * efficiency + x
            x /= (f+1)
            c += 1
        else:
            x = efficiency
            c = 1
        self.data[preprocessing] = x
        self.counts[preprocessing] = c


class BKZReduction(BKZ2):
    def __init__(self, A, tuners=None, recycle=True):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        BKZBase.__init__(self, A)
        self.recycle = recycle

        if tuners is None:
            self.tuners = [Tuner(b) for b in range(YOLO_MAX_BLOCK_SIZE)]
        else:
            self.tuners = tuners

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        tracer = BKZTreeTracer(self, verbosity=params.flags & BKZ.VERBOSE)

        if params.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        cputime_start = time.clock()

        # self.M.discover_all_rows()
        # with tracer.context("lll"):
        #     self.lll_obj()

        i = 0
        while True:
            print
            with tracer.context("tour", i):
                clean = self.tour(params, min_row, max_row, tracer)
            print "proba %.4f" % self.tuners[params.block_size].proba,
            for x in sorted(self.tuners[params.block_size].data.keys()):
                try:
                    print x, "\t %d \t %.2f " % (self.tuners[params.block_size].counts[x], self.tuners[params.block_size].data[x])
                except:
                    pass
            print
            i += 1
            if (not clean) or params.block_size >= self.A.nrows:
                break
            if (params.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
            if (params.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= params.max_time:
                break

        self.trace = tracer.trace
        return clean

    def svp_call(self, kappa, block_size, radius, pruning, for_hints=False, tracer=dummy_tracer):
        """Call SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: Coordinates of SVP solution or ``None`` if none was found.

        ..  note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.
        """
        solutions = []
        try:
            if self.recycle:
                enum_obj = Enumeration(self.M, block_size/2)
            else:
                enum_obj = Enumeration(self.M, 1)
            if pruning is None:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=1.):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0)
            else:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=pruning.probability):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0, pruning=pruning.coefficients)
            return [sol for (sol, _) in solutions[0:]]
        except EnumerationError:
            return None, []

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """Find shortest vector in projected lattice of dimension ``block_size`` and insert into
        current basis.

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """

        timer = Timer()
        rem_prob, inserted = 1.0, 1
        target_prob = params.min_success_probability

        while rem_prob > 1. - target_prob:
            tmp_target_prob = 1.01 * (target_prob - 1)/rem_prob + 1.01

            if inserted == 0:
                with tracer.context("randomize"):
                    self.randomize_block(kappa+1, kappa+block_size)

            with tracer.context("preprocessing"):
                preprocessing = self.tuners[block_size].preprocess()
                self.svp_preprocessing(kappa, block_size, params, preprocessing, tracer)

            with tracer.context("pruner"):
                radius, pruning = self.tuners[block_size].enum(self.M, kappa, tmp_target_prob, timer.elapsed())
            solutions = self.svp_call(kappa, block_size, radius, pruning, tracer=tracer)
            solution = solutions[0]
            if solution is None:
                hints = []
            else:
                hints = solutions[1:]

            if pruning is None:
                rem_prob = 0
            else:
                rem_prob *= (1 - pruning.probability)

            # radius, pruning = self.tuner.enum_for_hints(self.M, kappa, block_size, timer.elapsed())
            # if radius>0:
            #     hints += self.svp_call(kappa, block_size, radius, pruning, for_hints=False, tracer=tracer)
            # hints = self.filter_hints(hints)[:block_size/2]

            self.tuners[block_size].feedback(preprocessing, pruning, timer.elapsed())
            timer.reset()
            with tracer.context("postprocessing"):
                inserted = self.svp_postprocessing(kappa, block_size, solution, hints, tracer)

        return True

    def svp_preprocessing(self, kappa, block_size, param, preprocessing_block_sizes=None, tracer=dummy_tracer):
        clean = True

        clean &= BKZBase.svp_preprocessing(self, kappa, block_size, param, tracer)

        for preproc in preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND)
            clean &= self.tour(prepar, kappa, kappa + block_size, tracer=tracer)

        return clean

    def svp_postprocessing(self, kappa, block_size, solution, hints, tracer):
        """Insert SVP solution into basis and LLL reduce.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics
        :param hints: other interesting vectors

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        M = self.M

        if (solution is not None) and len(hints) == 0:
            nonzero_vectors = len([x for x in solution if x])
            if nonzero_vectors == 1:
                first_nonzero_vector = None
                for i in range(block_size):
                    if abs(solution[i]) == 1:
                        first_nonzero_vector = i
                        break

                M.move_row(kappa + first_nonzero_vector, kappa)
                with tracer.context("lll"):
                    self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1)
                return 1

        if solution is not None:
            vectors = [solution] + hints
        else:
            if len(hints) == 0:
                return 0
            vectors = hints
        l = len(vectors)

        for vector in vectors:
            M.create_row()
            with M.row_ops(M.d-1, M.d):
                for i in range(block_size):                   
                    M.row_addmul(M.d-1, kappa + i, vector[i])

        for i in reversed(range(l)):
            M.move_row(M.d-1, kappa)

        with tracer.context("postproc"):
            self.lll_obj(kappa, kappa, kappa+block_size+l)

        for i in range(l):
            M.move_row(kappa+block_size, M.d-1)
            M.remove_last_row()

        return l

    def filter_hints(self, hints):
        return [v for v in hints if sum([x*x for x in v]) > 1.5]

p = BKZ.Param(45, max_loops=8, min_success_probability=0.5, flags=BKZ.VERBOSE)
n = 160
A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
yBKZ = BKZReduction(A)

t = time.time()
yBKZ(p)
print yBKZ.trace.report()
t = time.time() - t
print "  time: %.2fs" % (t,)
