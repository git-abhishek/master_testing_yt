"""
Runner mechanism for answer testing

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2010-2011 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib; matplotlib.use("Agg")
import os, shelve, cPickle, sys, imp, tempfile

from yt.config import ytcfg; ytcfg["yt","serialize"] = "False"
import yt.utilities.cmdln as cmdln
from .xunit import Xunit

from output_tests import test_registry, MultipleOutputTest, \
                         RegressionTestException

def clear_registry():
    test_registry.clear()

class FileNotExistException(Exception):
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self):
        return "FileNotExistException: %s" % (self.filename)


def registry_entries():
    return test_registry.keys()

class RegressionTestStorage(object):
    def __init__(self, results_id, path = "."):
        self.id = results_id
        if results_id == "":
            self._path = os.path.join(path, "results")
        else:
            self._path = os.path.join(path, "results_%s" % self.id)
        if not os.path.isdir(self._path): os.mkdir(self._path)
        if os.path.isfile(self._path): raise RuntimeError

    def _fn(self, tn):
        return os.path.join(self._path, tn)

    def __setitem__(self, test_name, result):
        # We have to close our shelf manually,
        # as the destructor does not necessarily do this.
        # Context managers would be more appropriate.
        f = open(self._fn(test_name), "wb")
        cPickle.dump(result, f, protocol=-1)
        f.close()

    def __getitem__(self, test_name):
        if not os.path.exists(self._fn(test_name)):
            raise FileNotExistException(self._fn(test_name))
        f = open(self._fn(test_name), "rb")
        tr = cPickle.load(f)
        f.close()
        return tr

class RegressionTestRunner(object):
    def __init__(self, results_id, compare_id = None,
                 results_path = ".", compare_results_path = ".",
                 io_log = "OutputLog"):
        # This test runner assumes it has been launched with the current
        # working directory that of the test case itself.
        self.io_log = io_log
        self.id = results_id
        if compare_id is not None:
            self.old_results = RegressionTestStorage(
                                    compare_id, path=compare_results_path)
        else:
            self.old_results = None
        self.results = RegressionTestStorage(results_id, path=results_path)
        self.plot_list = {}
        self.passed_tests = {}

    def run_all_tests(self):
        plot_list = []
        for i,name in enumerate(sorted(test_registry)):
            self.run_test(name)
        return self.passed_tests

    def run_test(self, name):
        # We'll also need to call the "compare" operation,
        # but for that we'll need a data store.
        test = test_registry[name]
        plot_list = []
        if test.output_type == 'single':
            mot = MultipleOutputTest(self.io_log)
            for i,fn in enumerate(mot):
                # This next line is to keep the shelve module
                # from happily gobbling the disk
                #if i > 5: break 
                test_instance = test(fn)
                test_instance.name = "%s_%s" % (
                    os.path.basename(fn), test_instance.name )
                self._run(test_instance)

        elif test.output_type == 'multiple':
            test_instance = test(self.io_log)
            self._run(test_instance)

    watcher = None
    def _run(self, test):
        if self.watcher is not None:
            self.watcher.start()
        print self.id, "Running", test.name,
        test.setup()
        test.run()
        self.plot_list[test.name] = test.plot()
        self.results[test.name] = test.result
        success, msg = self._compare(test)
        if success == True: print "SUCCEEDED"
        else: print "FAILED"
        self.passed_tests[test.name] = success
        if self.watcher is not None:
            if success == True:
                self.watcher.addSuccess(test.name)
            else:
                self.watcher.addFailure(test.name, msg)

    def _compare(self, test):
        if self.old_results is None:
            return (True, "New Test")
        try:
            old_result = self.old_results[test.name]
        except FileNotExistException:
            return (False, "File %s does not exist." % test.name)
        try:
            test.compare(old_result)
        except RegressionTestException, exc:
            return (False, sys.exc_info())
        return (True, "Pass")

    def run_tests_from_file(self, filename):
        for line in open(filename):
            test_name = line.strip()
            if test_name not in test_registry:
                if test_name[0] != "#":
                    print "Test '%s' not recognized, skipping" % (test_name)
                continue
            print "Running '%s'" % (test_name)
            self.run_test(line.strip())

class EnzoTestRunnerCommands(cmdln.Cmdln):
    name = "enzo_tests"

    def _load_modules(self, test_modules):
        for fn in test_modules:
            if fn.endswith(".py"): fn = fn[:-3]
            print "Loading module %s" % (fn)
            mname = os.path.basename(fn)
            f, filename, desc = imp.find_module(mname, [os.path.dirname(fn)])
            project = imp.load_module(mname, f, filename, desc)

    def _update_io_log(self, opts, kwargs):
        if opts.datasets is None or len(opts.datasets) == 0: return
        f = tempfile.NamedTemporaryFile()
        kwargs['io_log'] = f.name
        for d in opts.datasets:
            fn = os.path.expanduser(d)
            print "Registered dataset %s" % fn
            f.write("DATASET WRITTEN %s\n" % fn)
        f.flush()
        f.seek(0)
        return f

    @cmdln.option("-f", "--dataset", action="append",
                  help="override the io_log and add this to the new one",
                  dest="datasets")
    @cmdln.option("-p", "--results-path", action="store",
                  help="which directory should results be stored in",
                  dest="results_path", default=".")
    def do_store(self, subcmd, opts, name, *test_modules):
        """
        ${cmd_name}: Run and store a new dataset.

        ${cmd_usage}
        ${cmd_option_list}
        """
        sys.path.insert(0, ".")
        self._load_modules(test_modules)
        kwargs = {}
        f = self._update_io_log(opts, kwargs)
        test_runner = RegressionTestRunner(name,
                results_path = opts.results_path,
                **kwargs)
        test_runner.run_all_tests()

    @cmdln.option("-o", "--output", action="store",
                  help="output results to file",
                  dest="outputfile", default=None)
    @cmdln.option("-p", "--results-path", action="store",
                  help="which directory should results be stored in",
                  dest="results_path", default=".")
    @cmdln.option("-n", "--nose", action="store_true",
                  help="run through nose with xUnit testing",
                  dest="run_nose", default=False)
    @cmdln.option("-f", "--dataset", action="append",
                  help="override the io_log and add this to the new one",
                  dest="datasets")
    def do_compare(self, subcmd, opts, reference, comparison, *test_modules):
        """
        ${cmd_name}: Compare a reference dataset against a new dataset.  The
        new dataset will be run regardless of whether it exists or not.

        ${cmd_usage}
        ${cmd_option_list}
        """
        if comparison == "__CURRENT__":
            import pkg_resources
            yt_provider = pkg_resources.get_provider("yt")
            path = os.path.dirname(yt_provider.module_path)
            from yt.utilities.command_line import _get_hg_version
            comparison = _get_hg_version(path)[:12]
            print "Setting comparison to: %s" % (comparison)
        sys.path.insert(0, ".")
        self._load_modules(test_modules)
        kwargs = {}
        f = self._update_io_log(opts, kwargs)
        test_runner = RegressionTestRunner(comparison, reference,
                            results_path=opts.results_path,
                            **kwargs)
        if opts.run_nose:
            test_runner.watcher = Xunit()
        results = test_runner.run_all_tests()
        if opts.run_nose:
            test_runner.watcher.report()
        if opts.outputfile is not None:
            f = open(str(opts.outputfile), "w")
            for testname, success in sorted(results.items()):
                f.write("%s %s\n" % (testname.ljust(100), success))

def run_main():
    etrc = EnzoTestRunnerCommands()
    sys.exit(etrc.main())

if __name__ == "__main__":
    run_main()
