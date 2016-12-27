# -*- coding: utf-8 -*-

from sys import stderr
from copy import copy

from random import randint
from fpylll import LLL, BKZ, Enumeration, EvaluatorStrategy, EnumerationError, IntegerMatrix, prune
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


class BKZ3Param():
    def __init__(self, block_size, strategies=None,
                 delta=LLL.DEFAULT_DELTA, flags=BKZ.DEFAULT,
                 max_loops=0, max_time=0,
                 auto_abort=None,
                 gh_factor=None,
                 min_success_probability=BKZ.DEFAULT_MIN_SUCCESS_PROBABILITY,
                 rerandomization_density=BKZ.DEFAULT_RERANDOMIZATION_DENSITY,
                 dump_gso_filename=None):
        """
        Create BKZ parameters object.

        :param block_size: an integer from 1 to ``nrows``
        :param strategies: a filename or a list of Strategies
        :param delta: LLL parameter `0.25 < δ < 1.0`
        :param flags: flags
        :param max_loops: maximum number of full loops
        :param max_time: stop after time seconds (up to loop completion)
        :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not
            decrease fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such
            that the algorithm will terminate if for ``max_iter`` loops the slope is not smaller
            than ``scale * old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is
            given, this is equivalent to providing ``(1.0,5)`` which is fpLLL's default.
        :param gh_factor: heuristic, if set then the enumeration bound will be set to
            ``gh_factor`` times the Gaussian Heuristic.  If ``True`` then ``gh_factor`` is set to
            1.1, which is fpLLL's default.
        :param min_success_probability: minimum success probability in an SVP reduction (when using
            pruning)
        :param rerandomization_density: density of rerandomization operation when using extreme
            pruning
        :param dump_gso_filename: if this is not ``None`` then the logs of the norms of the
            Gram-Schmidt vectors are written to this file after each BKZ loop.
        """
        self.bkz_param = BKZ.Param(block_size, strategies,
                                   delta, flags,
                                   max_loops, max_time,
                                   auto_abort,
                                   gh_factor,
                                   min_success_probability,
                                   rerandomization_density,
                                   dump_gso_filename)
        self.lll_eta = 0.51   # For weaker inner LLL
        self.nr_hints = 0.5   # Relative to block_size. For enumerating more than the SVP to do multiple insertions
        self.hints_bound = 1  # Keep the solution whose norm are <= hints_bound * norm(SVP)

    def set_lll_eta(self, eta):
        self.lll_eta = eta


class Timer:
    def __init__(self):
        self.start = time.clock()

    reset = __init__

    def elapsed(self):
        return time.clock() - self.start

DEFAULT_STRATEGIES = BKZ.Param(block_size=1, strategies="default.json").strategies


class Tuner(object):
    def __init__(self, block_size):
        self.last_prunings = None
        self.data = {}
        self.counts = {}
        self.block_size = block_size
        self.proba = .5
        if block_size > 1 and block_size < max(YOLO_PRUNER_MIN_BLOCK_SIZE, YOLO_PREPROC_MIN_BLOCK_SIZE):
            self.strategy = DEFAULT_STRATEGIES[block_size]

    def get_variations(self, preprocessing):
        V = [preprocessing]

        minb = 10
        if len(preprocessing) == 0:
            V.append(tuple([minb]))
            # V.append(tuple([(self.block_size/3, .5)]))
            return V
        if len(preprocessing) == 1:
            block_size = preprocessing[0]
            if block_size < minb + 6:
                V.append(tuple([]))
            for bb in reversed(range(max(block_size-2, minb), min(block_size+3, self.block_size - YOLO_GAP_PREPROC_BLOCK_SIZE))):
                V.append(tuple([bb]))
            return V
        assert False

    def get_preprocessing_block_sizes(self):
        # return tuple()
        # self.count += 1
        if self.block_size < YOLO_PREPROC_MIN_BLOCK_SIZE:
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
        # print self.block_size, best, variations
        ratio = best_efficiency / variation_efficiency
        p = ceil(ratio)
        if randint(0, p) == 0:
            return variation
        else:
            return best

    def get_pruning(self, M, kappa, target_prob, preproc_time):
        block_size = self.block_size

        radius = M.get_r(kappa, kappa) * .99
        root_det = M.get_root_det(kappa, kappa + block_size - 1)
        gh_radius, ge = gaussian_heuristic(radius, 0, block_size, root_det, 1.)
        if block_size > 30:
            radius = min(radius, 1.21 * gh_radius * 2**ge)

        if block_size < YOLO_PRUNER_MIN_BLOCK_SIZE:
            return radius, self.strategy.get_pruning(radius, gh_radius * 2**ge)

        R = tuple([M.get_r(i, i) for i in range(kappa, kappa+block_size)])
        overhead = (preproc_time + RESTART_PENALTY) * NODE_PER_SEC
        self.last_prunings = prune(self.last_prunings, radius, overhead, target_prob, [R],
                        descent_method="gradient", metric="probability", float_type="double", reset=False)
        self.proba = (self.proba * YOLO_MEMORY_LENGTH) + self.last_prunings.metric
        self.proba /= YOLO_MEMORY_LENGTH + 1
        return radius, self.last_prunings

    def enum_for_hints(self, M, kappa, block_size, preproc_time):
        return 0, None

    def feedback(self, preprocessing, pruning, time):
        if pruning is None:
            efficiency = 1. / time
        else:
            efficiency = pruning.metric / time
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
            self.tuners = [Tuner(block_size) for block_size in range(YOLO_MAX_BLOCK_SIZE)]
        else:
            self.tuners = tuners

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        tracer = BKZTreeTracer(self, verbosity=params.bkz_param.flags & BKZ.VERBOSE)
        self.params = params

        if params.bkz_param.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)
        if params.lll_eta is not LLL.DEFAULT_ETA:
            self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT, eta=params.lll_eta)

        cputime_start = time.clock()

        # self.M.discover_all_rows()
        # with tracer.context("lll"):
        #     self.lll_obj()

        i = 0
        self.ith_tour = 0
        while True:
            with tracer.context("tour", i):
                self.ith_block = 0
                self.ith_tour += 1
                clean = self.tour(params.bkz_param, min_row, max_row, tracer)
            print "proba %.4f" % self.tuners[params.bkz_param.block_size].proba
            for x in sorted(self.tuners[params.bkz_param.block_size].data.keys()):
                try:
                    print x, "\t %d \t %.2f " % (self.tuners[params.bkz_param.block_size].counts[x], self.tuners[params.bkz_param.block_size].data[x])
                except:
                    pass
            print
            i += 1
            if (not clean) or params.bkz_param.block_size >= self.A.nrows:
                break
            if (params.bkz_param.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.bkz_param.flags & BKZ.MAX_LOOPS) and i >= params.bkz_param.max_loops:
                break
            if (params.bkz_param.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= params.bkz_param.max_time:
                break

        self.trace = tracer.trace
        return clean

    def svp_call(self, kappa, block_size, radius, pruning, nr_hints=0, tracer=dummy_tracer):
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
            enum_obj = Enumeration(self.M, nr_hints+1)
            if pruning is None:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=1.):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0)
            else:
                with tracer.context("enumeration", enum_obj=enum_obj, probability=pruning.metric):
                    solutions = enum_obj.enumerate(kappa, kappa + block_size, radius, 0, pruning=pruning.coefficients)
            return [sol for (sol, _) in solutions]
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

        if (block_size == 60):
            self.ith_block += 1
        while rem_prob > 1. - target_prob:
            tmp_target_prob = 1.01 * (target_prob - 1)/rem_prob + 1.01

            if inserted == 0:
                with tracer.context("randomize"):
                    self.randomize_block(kappa+1, kappa+block_size)

            with tracer.context("preprocessing"):
                preprocessing = self.tuners[block_size].get_preprocessing_block_sizes()
#                if len(preprocessing) > 0:
#                    print >> stderr, self.ith_tour, self.ith_block, block_size, preprocessing
                self.svp_preprocessing(kappa, block_size, params, preprocessing, tracer)

            with tracer.context("pruner"):
                radius, pruning = self.tuners[block_size].get_pruning(self.M, kappa, tmp_target_prob, timer.elapsed())
            solutions = self.svp_call(kappa, block_size, radius, pruning, nr_hints=0, tracer=tracer)
            solution = solutions[0]
            if solution is None or len(solutions[1:]) == 0:
                hints = []
            else:
                hints = self.filter_hints(solutions[1:], self.params.hints_bound*sum([i*i for i in solution]))

            if pruning is None:
                rem_prob = 0
            else:
                rem_prob *= (1 - pruning.metric)

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
        #if (l > 1):
        #    print >> stderr, "Tour", self.ith_tour, "Block", self.ith_block, "(", block_size,") inserting", l, "vectors"

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

    def filter_hints(self, hints, bound):
        return [v for v in hints if sum([x*x for x in v]) <= bound]

n = 160
bs = 25
loops = 1
A = IntegerMatrix.random(n, "qary", k=n//2, bits=30)
p = BKZ3Param(bs, max_loops=loops, min_success_probability=0.5, flags=BKZ.VERBOSE | BKZ.BOUNDED_LLL)
p.set_lll_eta(0.51)
p.nr_hints = 0.25
p.hints_bound=0.9
yBKZ = BKZReduction(copy(A))
print "Go!"

t = time.time()
yBKZ(p)
t = time.time() - t
print "  time: %.2fs" % (t,)

