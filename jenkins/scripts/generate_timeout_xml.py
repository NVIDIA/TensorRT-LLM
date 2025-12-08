import argparse
import sys

def parse_xml_classname_name_file_from_testname(testname, stage_name):
    classname, name, file = "", "", ""

    # Get file name
    if testname.startswith("unittest/"):
        file = "test_unittests.py"
    else:
        file = testname.split("::")[0]

    # Get test name
    if testname.startswith("unittest/"):
        name = "test_unittests_v2[" + testname + "]"
    else:
        name = testname.split("::")[-1]

    # Get class name
    if testname.startswith("unittest/"):
        classname = stage_name + ".test_unittests"
    elif len(testname.split("::")) == 3:
        classname = stage_name + "." + testname.split("::")[0].replace(".py", "").replace("/", ".") + "." + testname.split("::")[1]
    else:
        classname = stage_name + "." + testname.split("::")[0].replace(".py", "").replace("/", ".")
        if testname.startswith("accuracy/") or (testname.startswith("examples/") and "[" not in testname):
            classname = ""

    return classname, name, file


def generate_timeout_xml(testList, stage_name):
    xmlContent = f"""<?xml version="1.0" encoding="UTF-8"?><testsuites>
        <testsuite name="{stage_name}" errors="{len(testList)}" failures="0" skipped="0" tests="{len(testList)}" time="1.00">"""
    
    for test in testList:
        classname, name, file = parse_xml_classname_name_file_from_testname(test, stage_name)
        xmlContent += f"""<testcase classname="{classname}" name="{name}" file="{file}" time="1.0">
        <error message="Test terminated unexpectedly"> Test terminated unexpectedly
        </error></testcase>"""
    xmlContent += "</testsuite></testsuites>"
    print(xmlContent)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests',
                        required=True,
                        help='Test list file')
    parser.add_argument('--stage-name',
                        required=True,
                        help='Stage name')
    args = parser.parse_args(sys.argv[1:])
    testList = args.tests.split('\n')
    generate_timeout_xml(testList, args.stage_name)
