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

    base_argv = ['nosetests', '--with-answer-testing', '--nologcapture',
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

def upload_to_cloudinary(filename, subdir=""):
    if cloudinary.config().cloud_name is None:
        load_cloudinary_config()

    # Create a folder with current date
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    folder = "yt/{}/{}".format(date, subdir)
    if '\\' in filename:
        filename = filename.replace("\\", "/")
    response = cloudinary.uploader.upload(filename,
                                          folder=folder,
                                          use_filename=True,
                                          resource_type="auto")
    return response

def upload_failed_answers(failed_answers):
    html = generate_webpage(failed_answers)

    tmpdir = tempfile.mkdtemp()
    filename = os.path.join(tmpdir, "index.html")
    with open(filename, "w") as outfile:
        outfile.write(html)

    response = upload_to_cloudinary(filename)
    shutil.rmtree(tmpdir)
    return response

def generate_missing_answers(answer_dir, missing_answers):
    status = True
    for job in missing_answers:
        try:
            new_job = job[:6]
            new_job += ['--local-dir=%s' % answer_dir, '--answer-store']
            new_job += job[-2:]
            print("Generating answers for", job[-1], end=" ")
            print(new_job)
            sys.stdout.flush()
            subprocess.check_output(' '.join(new_job), stderr=subprocess.STDOUT,
                                    universal_newlines=True, shell=True)
            print("... ok")
        except subprocess.CalledProcessError as e:
            status = False
            print("E")
            print(e.output)

    return status

def upload_missing_answers(missing_answers):
    tmpdir = tempfile.mkdtemp()
    answer_dir = os.path.join(tmpdir, "answer-store")
    zip_file = os.path.join(tmpdir, "new-answers")
    status = generate_missing_answers(answer_dir, missing_answers)
    if status:
        filename = shutil.make_archive(zip_file, 'zip', answer_dir)
        response = upload_to_cloudinary(filename, subdir="answers")
        shutil.rmtree(tmpdir)
        return response
    return None

def run_answer_test_cloud():
    # 0 on success and 1 on failure
    status = 0
    failed_answers = []
    missing_answers = []
    for job in generate_cloud_answer_tasks():
        answer_name = job[-2].split("=")[1]
        answer_dir = os.path.join(ANSWER_DIR, answer_name)
        if not os.path.exists(answer_dir):
            missing_answers.append(job)
            continue
        try:
            print("Running answer tests...")
            print(job[-1], end=" ")
            sys.stdout.flush()
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
                img_path[key] = result.group(1)
            if not unknown_failure:
                print("F")
                failed_answers.append((job[-1], img_path))

    # upload images if any, of the failed answer tests
    if failed_answers:
        status = 1
        response = upload_failed_answers(failed_answers)
        print("\nUploaded the failed answer tests result at following urls:")
        print("  secure url:", response["secure_url"])
        print("  url:", response["url"])

    if missing_answers:
        status = 1
        response = upload_missing_answers(missing_answers)
        if response is not None:
            print("\nUploaded missing answer tests at following urls:")
            print("  secure url:", response["secure_url"])
            print("  url:", response["url"])

    return status

if __name__ == "__main__":
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
