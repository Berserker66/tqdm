from __future__ import print_function, division

from nose.plugins.skip import SkipTest

from contextlib import contextmanager

import inspect
import re
import sys
from time import sleep, time
from bytecode_tracer import BytecodeTracer
from bytecode_tracer import rewrite_function

from tqdm import trange
from tqdm import tqdm

try:
    from StringIO import StringIO
except:
    from io import StringIO
# Ensure we can use `with closing(...) as ... :` syntax
if getattr(StringIO, '__exit__', False) and \
   getattr(StringIO, '__enter__', False):
    def closing(arg):
        return arg
else:
    from contextlib import closing

try:
    _range = xrange
except:
    _range = range

# Use relative/cpu timer to have reliable timings when there is a sudden load
try:
    from time import process_time
except ImportError:
    from time import clock
    process_time = clock


def get_relative_time(prevtime=0):
    return process_time() - prevtime


def cpu_sleep(t):
    '''Sleep the given amount of cpu time'''
    start = process_time()
    while((process_time() - start) < t):
        pass


def checkCpuTime(sleeptime=0.2):
    '''Check if cpu time works correctly'''
    # First test that sleeping does not consume cputime
    start1 = process_time()
    sleep(sleeptime)
    t1 = process_time() - start1

    # secondly check by comparing to cpusleep (where we actually do something)
    start2 = process_time()
    cpu_sleep(sleeptime)
    t2 = process_time() - start2

    return (abs(t1) < 0.0001 and (t1 < t2 / 10))


@contextmanager
def relative_timer():
    start = process_time()
    elapser = lambda: process_time() - start
    yield lambda: elapser()
    spent = process_time() - start
    elapser = lambda: spent


class MockFileNoWrite(StringIO):
    """ Wraps StringIO to mock a file with no I/O """
    def write(self, data):
        return


def simple_progress(iterable=None, total=None, file=sys.stdout, desc='', leave=False, miniters=1, mininterval=0.1, width=60):
    """ Simple progress bar reproducing tqdm's major features """
    n = [0]  # use a closure
    start_t = [time()]
    last_n = [0]
    last_t = [0]
    if iterable is not None:
        total = len(iterable)

    def format_interval(t):
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)

    def update_and_print(i=1):
        n[0] += i
        if (n[0] - last_n[0]) >= miniters:
            last_n[0] = n[0]
            
            cur_t = time()
            if (cur_t - last_t[0]) >= mininterval:
                last_t[0] = cur_t

                spent = cur_t - start_t[0]
                spent_fmt = format_interval(spent)
                eta = spent / n[0] * total if n[0] else 0
                frac = n[0] / total
                rate =  n[0] / spent if spent > 0 else 0
                eta = (total - n[0]) / rate if rate > 0 else 0
                eta_fmt = format_interval(eta)
                if 0.0 < rate < 1.0:
                    rate_fmt = "%.2fs/it" % (1.0 / rate)
                else:
                    rate_fmt = "%.2fit/s" % rate
                percentage = int(frac * 100)
                bar = "#" * int(frac * width)
                barfill = " " * int((1.0 - frac) * width)
                bar_length, frac_bar_length = divmod(
                    int(frac * width * 10), 10)
                bar = '#' * bar_length
                frac_bar = chr(48 + frac_bar_length) if frac_bar_length \
                    else ' '
                file.write("\r%s %i%%|%s%s%s| %i/%i [%s<%s, %s]" % (desc,
                                    percentage, bar, frac_bar, barfill, n[0],
                                    total, spent_fmt, eta_fmt, rate_fmt))
                if n[0] == total and leave:
                    file.write("\n")
                file.flush()

    def update_and_yield():
        for elt in iterable:
            yield elt
            update_and_print()

    update_and_print(0)
    if iterable is not None:
        return update_and_yield
    else:
        return update_and_print


btracer = BytecodeTracer()


def trace(frame, event, arg):
    '''Custom tracer callback with bytecode offset instead of line number'''
    bytecode_events = list(btracer.trace(frame, event))
    if bytecode_events:
        for ev, rest in bytecode_events:
            if ev == 'c_call':
                func, pargs, kargs = rest
                print("C_CALL", func.__name__, repr(pargs), repr(kargs))
            elif ev == 'c_return':
                print("C_RETURN", repr(rest))
            elif ev == 'print':
                print("PRINT", repr(rest))
            elif ev == 'print_to':
                value, output = rest
                print("PRINT_TO", repr(value), repr(output))
            else:
                print("C_OTHER:", repr(value), repr(rest))
    else:
        if event == 'call':
            args = inspect.getargvalues(frame)
            try:
                args = str(args)
            except Exception:
                args = "<unknown>"
            print("CALL", frame.f_code.co_name, args)
        elif event == 'return':
            print("RETURN", frame.f_code.co_name, repr(arg))
        elif event == 'exception':
            print("EXCEPTION", arg)
        elif event == 'line':
            # Most important statement for us: show each executed line and its
            # bytecode offset
            print("LINE", frame.f_code.co_filename, frame.f_lineno)
        else:
            print("OTHER", event, frame.f_code.co_name, repr(arg))
    return trace


@contextmanager
def captureStdOut(output):
    '''Capture stdout temporarily, use along a with statement'''
    stdout = sys.stdout
    sys.stdout = output
    yield
    sys.stdout = stdout


def getOpcodes(func, *args, **kwargs):
    '''Get the bytecode opcodes for a function'''
    # Redirect all printed outputs to a variable
    out = StringIO()
    with captureStdOut(out):
        # Setup bytecode tracer
        btracer.setup()

        #  dis.dis(func)  # not needed in our case

        # Rewrite the function to allow bytecode tracing
        rewrite_function(func)

        # Start the tracing
        sys.settrace(trace)
        try:
            # Execute the function
            func(*args, **kwargs)
        finally:
            # Stop the tracer
            sys.settrace(None)
            btracer.teardown()

    return out


def getOpcodesCount(func, *args, **kwargs):
    '''Get the total number of bytecode opcodes for a function'''
    out = getOpcodes(func, *args, **kwargs)

    # Filter tracing events to keep only executed lines
    opcodes = [s for s in out.getvalue().split('\n')
               if s.startswith('LINE') or s.startswith('C_CALL')]

    # Return the total number of opcodes
    return len(opcodes)


def getOpcodesCountHard(func, *args, **kwargs):
    '''Get the total number of bytecode opcodes for a function (pessimistic)'''
    out = getOpcodes(func, *args, **kwargs)

    # Hard mode: extract bytecode offsets and get the highest number for each
    # sequence, this should theoretically compute the exact timesteps taken for
    # each statement.
    # TODO: not sure this is correct, we may be overestimating a lot!
    RE_opcodes = re.compile(r'\S+\s+\S+\s+(\d+)')
    opcodes = [s for s in out.getvalue().split('\n') if s.startswith('LINE')]
    opcodes_offsets = [int(RE_opcodes.search(s).group(1))
                       if s.startswith('LINE') else 0 for s in opcodes]

    opcodes_total = 0
    for i in _range(1, len(opcodes_offsets)):
        if opcodes_offsets[i] <= opcodes_offsets[i - 1]:
            opcodes_total += opcodes_offsets[i]
    opcodes_total += opcodes_offsets[-1]

    return opcodes_total


def test_iter_overhead():
    """ Test overhead of iteration based tqdm """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e6)

    with closing(MockFileNoWrite()) as our_file:
        a = 0
        with relative_timer() as time_tqdm:
            for i in trange(total, file=our_file):
                a += i
        assert(a == (total * total - total) / 2.0)

        a = 0
        with relative_timer() as time_bench:
            for i in _range(total):
                a += i
                our_file.write(a)

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 3 * time_bench())
    except AssertionError:
        raise AssertionError('trange(%g): %f, range(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_manual_overhead():
    """ Test overhead of manual tqdm """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e6)

    with closing(MockFileNoWrite()) as our_file:
        t = tqdm(total=total * 10, file=our_file, leave=True)
        a = 0
        with relative_timer() as time_tqdm:
            for i in _range(total):
                a += i
                t.update(10)

        a = 0
        with relative_timer() as time_bench:
            for i in _range(total):
                a += i
                our_file.write(a)

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 10 * time_bench())
    except AssertionError:
        raise AssertionError('tqdm(%g): %f, range(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_iter_overhead_hard():
    """ Test overhead of iteration based tqdm (hard) """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e5)

    with closing(MockFileNoWrite()) as our_file:
        a = 0
        with relative_timer() as time_tqdm:
            for i in trange(total, file=our_file, leave=True,
                            miniters=1, mininterval=0, maxinterval=0):
                a += i
        assert(a == (total * total - total) / 2.0)

        a = 0
        with relative_timer() as time_bench:
            for i in _range(total):
                a += i
                our_file.write(("%i" % a) * 40)

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 60 * time_bench())
    except AssertionError:
        raise AssertionError('trange(%g): %f, range(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_manual_overhead_hard():
    """ Test overhead of manual tqdm (hard) """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e5)

    with closing(MockFileNoWrite()) as our_file:
        t = tqdm(total=total * 10, file=our_file, leave=True,
                 miniters=1, mininterval=0, maxinterval=0)
        a = 0
        with relative_timer() as time_tqdm:
            for i in _range(total):
                a += i
                t.update(10)

        a = 0
        with relative_timer() as time_bench:
            for i in _range(total):
                a += i
                our_file.write(("%i" % a) * 40)

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 100 * time_bench())
    except AssertionError:
        raise AssertionError('tqdm(%g): %f, range(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_iter_overhead_simplebar_hard():
    """ Test overhead of iteration based tqdm vs simple progress bar (hard) """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e4)

    with closing(MockFileNoWrite()) as our_file:
        a = 0
        with relative_timer() as time_tqdm:
            for i in trange(total, file=our_file, leave=True,
                            miniters=1, mininterval=0, maxinterval=0):
                a += i
        assert(a == (total * total - total) / 2.0)

        a = 0
        with relative_timer() as time_bench:
            simplebar_iter = simple_progress(_range(total), file=our_file,
                                        leave=True, miniters=1, mininterval=0)
            for i in simplebar_iter():
                a += i

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 1.5 * time_bench())
    except AssertionError:
        raise AssertionError('trange(%g): %f, simple_progress(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_manual_overhead_simplebar_hard():
    """ Test overhead of manual tqdm vs simple progress bar (hard) """
    try:
        assert checkCpuTime()
    except:
        raise SkipTest

    total = int(1e4)

    with closing(MockFileNoWrite()) as our_file:
        t = tqdm(total=total * 10, file=our_file, leave=True,
                 miniters=1, mininterval=0, maxinterval=0)
        a = 0
        with relative_timer() as time_tqdm:
            for i in _range(total):
                a += i
                t.update(10)

        simplebar_update = simple_progress(total=total, file=our_file,
                                        leave=True, miniters=1, mininterval=0)
        a = 0
        with relative_timer() as time_bench:
            for i in _range(total):
                a += i
                simplebar_update(10)

    # Compute relative overhead of tqdm against native range()
    try:
        assert(time_tqdm() < 1.5 * time_bench())
    except AssertionError:
        raise AssertionError('tqdm(%g): %f, simple_progress(%g): %f' %
                             (total, time_tqdm(), total, time_bench()))


def test_iter_overhead_hard_opcodes():
    """ Test overhead of iteration based tqdm (hard with opcodes) """
    try:
        import imputil
    except ImportError:
        raise SkipTest

    total = int(10)

    def f1():
        with closing(MockFileNoWrite()) as our_file:
            a = 0
            for i in trange(total, file=our_file, leave=True,
                            miniters=1, mininterval=0, maxinterval=0):
                a += i
            assert(a == (total * total - total) / 2.0)

    def f2():
        with closing(MockFileNoWrite()) as our_file:
            a = 0
            for i in _range(total):
                a += i
                our_file.write(("%i" % a) * 40)

    # Compute opcodes overhead of tqdm against native range()
    count1 = getOpcodesCount(f1)
    count2 = getOpcodesCount(f2)
    count1h = getOpcodesCountHard(f1)
    count2h = getOpcodesCountHard(f2)
    try:
        assert(count1 < 7 * count2)
        assert(count1h < 20 * count2h)
    except AssertionError:
        raise AssertionError('trange(%g): %i-%i, range(%g): %i-%i' %
                             (total, count1, count1h, total, count2, count2h))


def test_manual_overhead_hard_opcodes():
    """ Test overhead of manual tqdm (hard with opcodes) """
    try:
        import imputil
    except ImportError:
        raise SkipTest

    total = int(10)

    def f1():
        with closing(MockFileNoWrite()) as our_file:
            t = tqdm(total=total * 10, file=our_file, leave=True,
                     miniters=1, mininterval=0, maxinterval=0)
            a = 0
            for i in _range(total):
                a += i
                t.update(10)

    def f2():
        with closing(MockFileNoWrite()) as our_file:
            a = 0
            for i in _range(total):
                a += i
                our_file.write(("%i" % a) * 40)

    # Compute opcodes overhead of tqdm against native range()
    count1 = getOpcodesCount(f1)
    count2 = getOpcodesCount(f2)
    count1h = getOpcodesCountHard(f1)
    count2h = getOpcodesCountHard(f2)
    try:
        assert(count1 < 20 * count2)
        assert(count1h < 20 * count2h)
    except AssertionError:
        raise AssertionError('tqdm(%g): %i-%i, range(%g): %i-%i' %
                             (total, count1, count1h, total, count2, count2h))
