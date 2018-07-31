"""
Answer Tests runner on cloud platforms like Travis

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import base64
import datetime
import logging
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import nose
import numpy
import requests

from yt.config import ytcfg
from yt.utilities.answer_testing.framework import AnswerTesting
from yt.utilities.command_line import FileStreamer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cloud-answer-test-runner')
numpy.set_printoptions(threshold=5, edgeitems=1, precision=4)

def generate_webpage(failed_answers):
    """Generates html for the failed answer tests

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

    html_template = """
    <html><head>
    <style media="screen" type="text/css">
    img{{
      width:100%;
      max-width:800px;
    }}
    </style>
    <h1 style="text-align: center;">Failed Answer Tests</h1>
    <p>
      This report shows images of answer tests that failed when running 
      the answer tests.
    </p>
    <p>
      <strong>Acutal Image:</strong> plot generated while running the test<br/> 
      <strong>Expected Image:</strong> golden answer image<br/> 
      <strong>Difference Image:</strong> difference in the "actual" 
      and "expected" image
    </p>
    <hr/>
    </head><body>
    <table>{rows}</table>
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
    <tr><td align="center" colspan="3"><b>Test: {3}</b><hr/></td></tr>
    """

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

    html = html_template.format(rows='\n'.join(rows))
    return html

def upload_to_curldrop(data, filename):
    """Uploads file to yt's curldrop server

    Uploads bytes `data` by the name `filename` to yt curldrop server.

    Parameters
    ----------
    data : bytes
        Content to be uploaded

    filename : string
        Name of file at curldrop's upload server

    Returns
    -------
    requests.models.Response
        Response returned by curldrop server

    """
    base_url = ytcfg.get("yt", "curldrop_upload_url")
    upload_url = base_url + "/" + os.path.basename(filename)
    response = requests.put(upload_url, data=data)
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
    requests.models.Response
        Response as returned by the upload service

    """
    html = generate_webpage(failed_answers)
    # convert html str to bytes
    html = html.encode()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "failed_answers_{}.html".format(timestamp)

    response = upload_to_curldrop(data=html, filename=filename)
    return response

def generate_missing_answers(answer_dir, missing_answers):
    """Generate golden answers

    Generates golden answers for the list of answers in `missing_answers` and
    saves them at `answer_dir`.

    Parameters
    ----------
    answer_dir : string
        directory location to save the generated answers

    missing_answers : list of string
        Collection of missing answer tests specifying full name of the test.
        eg. ['yt.visualization.tests.test_line_plots:test_multi_line_plot']

    Returns
    -------
    bool
        True, if all the missing answers are successfully generated
        False, otherwise

    """
    status = True
    test_argv = [os.path.basename(__file__), '--with-answer-testing',
                 '--nologcapture', '-s', '-d', '-v', '--local',
                 '--local-dir=%s' % answer_dir, '--answer-store']

    for job in missing_answers:
        log.info(" Generating answers for " + job)
        status &= nose.run(argv=test_argv+[job], addplugins=[AnswerTesting()],
                           exit=False)
    return status

def upload_missing_answers(missing_answers):
    """Uploads answers not present in answer-store

    This function generates the answers for tests that are not present in
    answer store and uploads a zip file of the same.

    Parameters
    ----------
    missing_answers : list of string
        Collection of missing answer tests specifying full name of the test.
        eg. ['yt.visualization.tests.test_line_plots:test_multi_line_plot']

    Returns
    -------
    requests.models.Response
        Response as returned by the upload service when answers are
        successfully uploaded

    None
        for the case when there was some error while generating the missing
        golden-answers

    """
    # Create temporary location to save new answers
    tmpdir = tempfile.mkdtemp()
    answer_dir = os.path.join(tmpdir, "answer-store")
    zip_file = os.path.join(tmpdir, "new-answers")

    status = generate_missing_answers(answer_dir, missing_answers)
    if status:
        zip_file = shutil.make_archive(zip_file, 'zip', answer_dir)
        data = iter(FileStreamer(open(zip_file, 'rb')))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "new_answers_{}.zip".format(timestamp)
        response = upload_to_curldrop(data=data, filename=filename)
        shutil.rmtree(tmpdir)
        return response
    return None

def extract_image_locations(error_string):
    """Regex based function to extract image file locations.

    Parameters
    ----------
    error_string : String
        The input string having file locations of 'Actual', 'Expected' and
        'Difference' plots. This string is generated by yt's answer-testing
        plugin, when the plot generated in the test does not match to its
        golden answer image.

    Returns
    -------
    dict
        If the `error_string` is successfully parsed to extract plot locations,
        then a dictionary with the keys 'Actual', 'Expected','Difference' and
        values having corresponding plot file locations is returned.
        eg. {'Actual': '/usr/tmp/tmp43la9b0w.png',
             'Expected': '/usr/tmp/tmpbpaqbgi3.png',
             'Difference': '/usr/tmp/tmp43la9b0w-failed-diff.png'}
    None
        When `error_string` does not conform to yt's answer-testing error
        message, which has the information for plot file locations on disk.

    """
    unknown_failure = False
    base_regex = r"\s*\n\s*(.*?.png)"
    img_regex = {"Actual": "Actual:" + base_regex,
                 "Expected": "Expected:" + base_regex,
                 "Difference": "Difference:" + base_regex}
    img_path = {}
    for key in img_regex:
        result = re.search(img_regex[key], error_string, re.MULTILINE)
        if not result:
            unknown_failure = True
            break
        # store the locations of actual, expected and diff plot files
        img_path[key] = result.group(1)

    if not unknown_failure:
        return img_path
    return None

def parse_nose_xml(nose_xml):
    """Parse xml file generated by nosetests.

    Parse nose xml file to find following details:
     Failed tests: These could be due to difference in golden answer image and
                   corresponding test plot.
     Missing tests: These errors occur when a corresponding golden answer image
                    is not found.

    Parameters
    ----------
    nose_xml : string
        full path of xml file to be parsed

    Returns
    -------
    tuple : (failed_answers, missing_answers)

        failed_answers : list of tuples (string, dict)
        Collection of tuples where the first part is a string denoting the
        test name, the second part is a dictionary that stores the actual,
        expected and difference plot file locations of the test.
        eg. [('yt.visualization.tests.test_line_plots:test_line_plot',
                {'Actual': '/usr/tmp/tmp43la9b0w.png',
                'Expected': '/usr/tmp/tmpbpaqbgi3.png',
                'Difference': '/usr/tmp/tmp43la9b0w-failed-diff.png'}
            )]

        missing_answers : list of string
        Collection of missing answer tests specifying full name of the test.
        eg. ['yt.visualization.tests.test_line_plots:test_multi_line_plot']

    """
    missing_answers = []
    failed_answers = []

    tree = ET.parse(nose_xml)
    testsuite = tree.getroot()
    print("Reading nose_xml:", testsuite)
    for testcase in testsuite:
        for error in testcase:
            test_name = testcase.attrib["classname"] + ":" \
                        + testcase.attrib["name"]
            print("test_name:", test_name, "Error (type):", error.attrib["type"]
                  , "Error (msg):", error.attrib["message"])
            if "No such file or directory" in error.attrib["message"]:
                missing_answers.append(test_name)
            elif "Items are not equal" in error.attrib["message"]:
                img_path = extract_image_locations(error.attrib["message"])
                if img_path:
                    failed_answers.append((test_name, img_path))
    return failed_answers, missing_answers

if __name__ == "__main__":
    """Run answer tests on cloud platforms like Travis, Appveyor

    This script executes yt answer tests on cloud platform.
    If the test fail due to difference in actual and expected images, 
    this function uploads a html page having all the plots which got failed 
    (if executed with `-f` command line argument).
    In case, answer store does not has a golden answer and if executed with 
    `-m` argument, it uploads missing answers zip file to yt's curldrop server.
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--upload-failed-tests", action="store_true",
                        help="Upload a comparison report of failed answer tests"
                             " to yt's curldrop server.")
    parser.add_argument("-m", "--upload-missing-answers", action="store_true",
                        help="Upload tests' answers that are not found in "
                             "answer-store.")
    args = parser.parse_args()

    # ANSI color codes
    COLOR_BLUE = '\x1b[34;1m'
    COLOR_CYAN = '\x1b[36;1m'
    COLOR_RESET = '\x1b[0m'

    ANSWER_STORE = ytcfg.get("yt", "test_storage_dir")
    if not ANSWER_STORE or ANSWER_STORE == "/does/not/exist":
        ANSWER_STORE = "answer-store"

    NOSETEST_XML = "answer_testing_nosetests.xml"

    # first parameter is program name
    # https://github.com/python/cpython/blob/master/Lib/unittest/main.py#L99
    test_argv = [os.path.basename(__file__), '--with-answer-testing',
                 '--nologcapture', '-s', '-d', '-v', '--local',
                 '--attr=answer_test', '--local-dir=%s' % ANSWER_STORE,
                 '--with-xunit', '--xunit-file=%s' % NOSETEST_XML, 'yt']
    result = nose.run(argv=test_argv, addplugins=[AnswerTesting()], exit=False)
    print("Nose run status: ", result)
    log.info("logging nose result:" + str(result))
    if args.upload_failed_tests or args.upload_missing_answers:
        failed_answers, missing_answers = parse_nose_xml(NOSETEST_XML)
    print("failed_answers", failed_answers)
    print("missing_answers", missing_answers)
    if args.upload_failed_tests and failed_answers:
        response = upload_failed_answers(failed_answers)
        print("Failed response",response)
        if response.ok:
            msg = " Successfully uploaded failed answer tests result."
            log.info(COLOR_BLUE + msg + COLOR_RESET)
            log.info(COLOR_BLUE + " " + response.text + COLOR_RESET)
            print(msg+"\n"+response.text)

    if args.upload_missing_answers and missing_answers:
        response = upload_missing_answers(missing_answers)
        print("Missing response", response)
        if response.ok:
            msg = " Successfully uploaded missing answer tests."
            log.info(COLOR_CYAN + msg + COLOR_RESET)
            log.info(COLOR_CYAN + " " + response.text + COLOR_RESET)
            print(msg + "\n" + response.text)
    # 0 on success and 1 on failure
    sys.exit(not result)