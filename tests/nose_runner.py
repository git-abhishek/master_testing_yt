from __future__ import print_function

import argparse
import base64
import datetime
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cloudinary
import cloudinary.uploader
import nose
import numpy
import yaml
from yt.utilities.command_line import FileStreamer

from yt.config import ytcfg
from yt.extern.six import StringIO
from yt.utilities.answer_testing.framework import AnswerTesting
from yt.utilities.exceptions import YTException

numpy.set_printoptions(threshold=5, edgeitems=1, precision=4)

ANSWER_DIR = os.path.join("answer-store")

class NoseWorker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print("%s: Exiting" % proc_name)
                self.task_queue.task_done()
                break
            print('%s: %s' % (proc_name, next_task))
            result = next_task()
            self.task_queue.task_done()
            self.result_queue.put(result)
            if next_task.exclusive:
                print("%s: Exiting (exclusive)" % proc_name)
                break
        return

class NoseTask(object):
    def __init__(self, job):
        argv, exclusive = job
        self.argv = argv
        self.name = argv[0]
        self.exclusive = exclusive

    def __call__(self):
        old_stderr = sys.stderr
        sys.stderr = mystderr = StringIO()
        test_dir = ytcfg.get("yt", "test_data_dir")
        answers_dir = os.path.join(test_dir, "answers")
        if '--with-answer-testing' in self.argv and \
                not os.path.isdir(os.path.join(answers_dir, self.name)):
            nose.run(argv=self.argv + ['--answer-store'],
                     addplugins=[AnswerTesting()], exit=False)
        if os.path.isfile("{}.xml".format(self.name)):
            os.remove("{}.xml".format(self.name))
        nose.run(argv=self.argv, addplugins=[AnswerTesting()], exit=False)
        sys.stderr = old_stderr
        return mystderr.getvalue()

    def __str__(self):
        return 'WILL DO self.name = %s' % self.name


def generate_tasks_input():
    pyver = "py{}{}".format(sys.version_info.major, sys.version_info.minor)
    if sys.version_info < (3, 0, 0):
        DROP_TAG = "py3"
    else:
        DROP_TAG = "py2"

    test_dir = ytcfg.get("yt", "test_data_dir")
    answers_dir = os.path.join(test_dir, "answers")
    with open('tests/tests.yaml', 'r') as obj:
        lines = obj.read()
    data = '\n'.join([line for line in lines.split('\n')
                      if DROP_TAG not in line])
    tests = yaml.load(data)

    base_argv = ['--local-dir=%s' % answers_dir, '-s', '--nologcapture',
                 '--with-answer-testing', '--answer-big-data', '--local']
    args = []

    for test in list(tests["other_tests"].keys()):
        args.append(([test] + tests["other_tests"][test], True))
    for answer in list(tests["answer_tests"].keys()):
        if tests["answer_tests"][answer] is None:
            continue
        argv = ["{}_{}".format(pyver, answer)]
        argv += base_argv
        argv.append('--answer-name=%s' % argv[0])
        argv += tests["answer_tests"][answer]
        args.append((argv, False))

    args = [(item + ['--xunit-file=%s.xml' % item[0]], exclusive)
            for item, exclusive in args]
    return args

def generate_cloud_answer_tasks():
    """Generates the answer test commands

    This function reads the file tests/cloud_answer_tests.yaml and generates
    commands for each of the tests listed in the file.

    Returns
    -------
    list of list of string
        The inner list of string specifies the full command parameters that
        need to be executed for a given answer test

    """
    pyver = "py{}{}".format(sys.version_info.major, sys.version_info.minor)
    if sys.version_info < (3, 0, 0):
        DROP_TAG = "py3"
    else:
        DROP_TAG = "py2"

    test_file = os.path.join("tests", "cloud_answer_tests.yaml")

    with open(test_file, 'r') as obj:
        lines = obj.read()
    data = '\n'.join([line for line in lines.split('\n')
                      if DROP_TAG not in line])
    tests = yaml.load(data)

    base_argv = ['coverage', 'run', '-a', '$(which nosetests)', '--with-answer-testing', '--nologcapture',
                 '-d', '-v', '--local', '--local-dir=%s' % ANSWER_DIR]
    args = []
    for answer in list(tests["answer_tests"]):
        if tests["answer_tests"][answer] is None:
            continue
        argv = []
        argv += base_argv
        argv.append('--answer-name=%s' % "{}_{}".format(pyver, answer))
        argv += tests["answer_tests"][answer]
        args.append(argv)

    return args


def generate_webpage(failed_answers):
    """Generates  html for the failed answer tests

    This function creates a html and embeds the images (actual, expected,
    difference) in it for the failed answers.

    Parameters
    ----------
    failed_answers : list of tuples (string, dict)
        Collection of tuples where the first part is a string denoting the
        test name, the second part is a dictionary that stores the actual,
        expected and difference plot file locations of the test.

    Returns
    -------
    string
        a html page

    """

    html_template = """<html><head>
    <style media="screen" type="text/css">
    img{{
      width:100%;
      max-width:800px;
    }}
    </style>
    <h1>{header}</h1>
    </head><body>
    {body}
    </body></html>
    """

    row_template = """
    <tr>
    <td align="center">Actual</td>
    <td align="center">Expected</td>
    <td align="center">Difference</td>
    </tr>
    <tr>
    <td><img src="data:image/png;base64,{0}"></td>
    <td><img src="data:image/png;base64,{1}"></td>
    <td><img src="data:image/png;base64,{2}"></td>
    </tr>
    <tr><td align="center" colspan="3"><b>{3}</b><hr></td></tr>
    """

    table_template = """<table>{rows}</table>"""
    rows = []

    for test_name, images in failed_answers:
        encoded_images = {}
        for key in images:
            with open(images[key], "rb") as img:
                img_data = base64.b64encode(img.read()).decode()
                encoded_images[key] = img_data

        formatted_row = row_template.format(encoded_images["Actual"],
                                            encoded_images["Expected"],
                                            encoded_images["Difference"],
                                            test_name)
        rows.append(formatted_row)

    body = table_template.format(rows='\n'.join(rows))
    html = html_template.format(header="Failed Answer Tests", body=body)
    return html

def load_cloudinary_config():
    """Set the configuarion parameters of cloudinary upload service

    This function picks the cloudinary api key and secret from the environment.
    It picks the value from environment variables of Travis or Appveyor if
    running on Travis/Appveyor. If running locally on an user machine, it picks
    the configuration from user's yt config file.

    """
    config = {}
    if "TRAVIS" in os.environ:
        config["cloud_name"] = os.environ["TRAVIS_CLOUDINARY_NAME"]
        config["api_key"] = os.environ["TRAVIS_CLOUDINARY_API_KEY"]
        config["api_secret"] = os.environ["TRAVIS_CLOUDINARY_API_SECRET"]
    elif "APPVEYOR" in os.environ:
        config["cloud_name"] = os.environ["APPVEYOR_CLOUDINARY_NAME"]
        config["api_key"] = os.environ["APPVEYOR_CLOUDINARY_API_KEY"]
        config["api_secret"] = os.environ["APPVEYOR_CLOUDINARY_API_SECRET"]
    elif ytcfg.has_option("yt", "cloudinary_name"):
        config["cloud_name"] = ytcfg.get("yt", "cloudinary_name")
        config["api_key"] = ytcfg.get("yt", "cloudinary_api_key")
        config["api_secret"] = ytcfg.get("yt", "cloudinary_api_secret")
    else:
        raise YTException("Unable to retrieve Cloudinary API details from the "
                          "environment ({Travis, AppVeyor, ~/.config/yt/ytrc}).")
    cloudinary.config(
        cloud_name=config["cloud_name"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )

def upload_to_curldrop(data, filename):
    """Uploads file to yt's curldrop server

    Uploads the file specified by `filename` to cloudinary at the location
    yt/{current-date}/{subdir}

    Parameters
    ----------
    filename : string
        Path of the file to be uploaded

    subdir : string
        string specifying a directory name under which the file gets uploaded.
        If nothing is specified, the parent directory will be yt/{current-date}.
        The directory structure is as follows:
        .
        +-- yt
        |   +-- current-date
        |   |   +-- subdir

    Returns
    -------
    json
        Response returned by cloudinary

    """
    upload_url = ytcfg.get("yt", "curldrop_upload_url")
    response = requests.put(upload_url + "/" + os.path.basename(filename), data=data)
    return response

def upload_failed_answers(failed_answers):
    """Uploads the result of failed answer tests

    Uploads a html page of the failed answer tests.

    Parameters
    ----------
    failed_answers : list of tuples (string, dict)
        Collection of tuples where the first part is a string denoting the
        test name, the second part is a dictionary that stores the actual,
        expected and difference plot file locations of the test.

    Returns
    -------
    json
        Response as returned by the upload service

    """
    html = generate_webpage(failed_answers)
    html = html.encode()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "failed_answers_{}.html".format(timestamp)
    response = upload_to_curldrop(data=html, filename=filename)
    return response

def generate_missing_answers(answer_dir, missing_answers):
    """Generate golden-answers

    Generates golden-answers for the list of answers in `missing_answers` and
    saves them at `answer_dir`.

    Parameters
    ----------
    answer_dir : string
        directory location to save the generated answers

    missing_answers : list of list of string
        Collection of missing answer tests, where the inner list of string
        specifies the full command line parameters to be executed for a given
        answer test.

    Returns
    -------
    bool
        True, if all the missing answers are successfully generated
        False, otherwise

    """
    status = True
    base_argv = ['nosetests', '--with-answer-testing', '--nologcapture',
                 '-d', '-v', '--local', '--local-dir=%s' % answer_dir, '--answer-store']
    for job in missing_answers:
        try:
            print("Generating answers for", job[-1], end=" ")
            new_job = []
            new_job += base_argv
            new_job += job
            subprocess.check_output(' '.join(new_job), stderr=subprocess.STDOUT,
                                    universal_newlines=True, shell=True)
            print("... ok")
        except subprocess.CalledProcessError as e:
            status = False
            print("E")
            print(e.output)

    return status

def upload_missing_answers(missing_answers):
    """Uploads answers not present in answer-store

    This function generates the answers for tests that are not present in
    answer store and uploads a zip file of the same.

    Parameters
    ----------
    missing_answers : list of list of string
        Collection of missing answer tests, where the inner list of string
        specifies the full command line parameters to be executed for a given
        answer test.

    Returns
    -------
    json
        for the case when answers are successfully uploaded
    None
        for the case when there was some error while generating the missing
        golden-answers

    """
    tmpdir = tempfile.mkdtemp()
    answer_dir = os.path.join(tmpdir, "answer-store")
    zip_file = os.path.join(tmpdir, "new-answers")
    status = generate_missing_answers(answer_dir, missing_answers)
    if status:
        zip_file = shutil.make_archive(zip_file, 'zip', answer_dir)
        data = iter(FileStreamer(open(zip_file, 'r')))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "new_answers_{}.zip".format(timestamp)
        response = upload_to_curldrop(data=data, filename=filename)
        shutil.rmtree(tmpdir)
        return response
    return None

def run_answer_test_cloud():
    """Run answer tests on cloud platform like Travis, Appveyor

    This function execute yt answer tests on cloud platform. In case, answer
    store does not has golden-answer, it uploads the missing answers and fails
    the test execution finally by returning status 1. If the test fail due to
    difference in actual and expected images, this function uploads a html page
    having all the plots which got failed.

    Returns
    -------
    int
        0, if all the answer tests runs successfully
        1, otherwise

    """
    # 0 on success and 1 on failure
    status = 0
    failed_answers = []
    missing_answers = []
    for job in generate_cloud_answer_tasks():
        # check if golden answer exits?
        answer_name = job[-2].split("=")[1]
        answer_dir = os.path.join(ANSWER_DIR, answer_name)
        if not os.path.exists(answer_dir):
            missing_answers.append(job[-2:])
            continue
        try:
            # execute the nosetests command
            print(job[-1], end=" ")
            result = subprocess.check_output(' '.join(job), stderr=subprocess.STDOUT,
                                             universal_newlines=True, shell=True)
            time_regex = r"Ran 1 test in (\d*.\d*s)"
            result = re.search(time_regex, result, re.MULTILINE)
            if result is not None:
                time = result.group(1)
                print("... ok [%s]" % time)
        except subprocess.CalledProcessError as e:

            unknown_failure = False
            base_regex = r"\s*\n\s*(.*?.png)"
            img_regex = {"Actual": "Actual:" + base_regex,
                         "Expected": "Expected:" + base_regex,
                         "Difference": "Difference:" + base_regex}
            img_path = {}
            for key in img_regex:
                result = re.search(img_regex[key], e.output, re.MULTILINE)
                if result is None:
                    unknown_failure = True
                    print("E")
                    print(e.output)
                    status = 1
                    break
                # store the locations of actual, expected and diff plot files
                img_path[key] = result.group(1)
            if not unknown_failure:
                print("F")
                failed_answers.append((job[-1], img_path))

    bright_mageneta = "\033[35;1m"
    reset_color = "\033[0m"
    # upload plot differences of the failed answer tests
    if failed_answers:
        status = 1
        response = upload_failed_answers(failed_answers)
        if response.ok:
            msg = bright_mageneta
            msg += "\nSuccessfully uploaded failed answer tests."
            msg += response.text + reset_color
            print(msg)

    # upload missing answers, if any
    if missing_answers:
        status = 1
        response = upload_missing_answers(missing_answers)
        if response.ok:
            msg = bright_mageneta
            msg += "\nSuccessfully uploaded missing answer tests."
            msg += response.text + reset_color
            print(msg)

    return status

if __name__ == "__main__":
    import requests
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runAnswerTestOnCloud", action="store_true",
                        help="Run answer tests on cloud platforms like Travis, "
                             "AppVeyor.")
    args = parser.parse_args()
    if args.runAnswerTestOnCloud:
        status = run_answer_test_cloud()
        sys.exit(status)

    # multiprocessing.log_to_stderr(logging.DEBUG)
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    num_consumers = int(os.environ.get('NUM_WORKERS', 6))
    consumers = [NoseWorker(tasks, results) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    num_jobs = 0
    for job in generate_tasks_input():
        if job[1]:
            num_consumers -= 1  # take into account exclusive jobs
        tasks.put(NoseTask(job))
        num_jobs += 1

    for i in range(num_consumers):
        tasks.put(None)

    tasks.join()

    while num_jobs:
        result = results.get()
        num_jobs -= 1
