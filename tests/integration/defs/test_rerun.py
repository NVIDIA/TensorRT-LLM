import os
import sys
import xml.etree.ElementTree as ET


def parse_name(name, filename):
    if name.startswith("test_unittests_v2[unittest/") and filename == "test_unittests.py":
        return name[len("test_unittests_v2[") : -1]
    else:
        return filename + "::" + name


def generate_rerun_tests_list(outdir, xml_filename):
    # Generate rerun test lists:
    # 1. Parse the test results xml file
    # 2. For failed tests:
    #    - If test duration <= 5 min: add to rerun_2.txt (will rerun 2 times)
    #    - If test duration > 5 min and <= 10 min: add to rerun_1.txt (will rerun 1 time)
    #    - If test duration > 10 min but contains fail signatures in error message: add to rerun_1.txt
    #    - If test duration > 10 min and no known failure signatures: add to rerun_0.txt (will not rerun)

    # todo: change
    # failSignaturesList = trtllm_utils.getFailSignaturesList()
    failSignaturesList = ["bad luck"]  

    rerun_0_filename = os.path.join(outdir, 'rerun_0.txt')
    rerun_1_filename = os.path.join(outdir, 'rerun_1.txt')
    rerun_2_filename = os.path.join(outdir, 'rerun_2.txt')

    tree = ET.parse(xml_filename)
    root = tree.getroot()
    suite = root.find('testsuite')

    with open(rerun_0_filename, 'w') as rerun_0_file, \
         open(rerun_1_filename, 'w') as rerun_1_file, \
         open(rerun_2_filename, 'w') as rerun_2_file:
        for case in suite.findall('testcase'):
            if case.find('failure') is not None or case.find('error') is not None:
                duration = float(case.attrib.get('time', 0))
                test_name = parse_name(case.attrib.get('name', ''), case.attrib.get('file', ''))
                if duration <= 5 * 60:
                    rerun_2_file.write(test_name + '\n')
                    print(test_name + " will rerun 2 times")
                elif duration <= 10 * 60:
                    rerun_1_file.write(test_name + '\n')
                    print(test_name + " will rerun 1 time")
                elif any(failSig.lower() in ET.tostring(case, encoding='unicode').lower() for failSig in failSignaturesList):
                    rerun_1_file.write(test_name + '\n')
                    print(test_name + " will rerun 1 time, because of fail signature")
                else:
                    rerun_0_file.write(test_name + '\n')
                    print(test_name + " will not rerun")

    for filename in [rerun_0_filename, rerun_1_filename, rerun_2_filename]:
        if os.path.getsize(filename) == 0:
            os.remove(filename)


def merge_junit_xmls(merged_xml_filename, xml_filenames):
    merged_root = ET.Element('testsuites')
    merged_suite = ET.Element('testsuite')

    attribs = {'name': '', 'errors': 0, 'failures': 0, 'skipped': 0, 'tests': 0, 'time': 0.0, 'timestamp': '', 'hostname': ''}

    # Iterate through all input files
    for xml_filename in xml_filenames:
        if not os.path.exists(xml_filename):
            continue
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        suite = root.find('testsuite')
        
        if attribs['name'] == "":
            attribs['name'] = suite.attrib.get('name', '')
        if attribs['timestamp'] == "":
            attribs['timestamp'] = suite.attrib.get('timestamp', '')
        if attribs['hostname'] == "":
            attribs['hostname'] = suite.attrib.get('hostname', '')
        attribs['time'] += float(suite.attrib.get('time', 0))

        for case in suite.findall('testcase'):
            # Check if test case with same name and classname already exists
            existing_case = merged_suite.find(f"testcase[@name='{case.attrib['name']}'][@classname='{case.attrib['classname']}']")
            if existing_case is not None:
                # Remove existing case and add the new one (which is presumably from a rerun)
                merged_suite.remove(existing_case)
            merged_suite.append(case)

    for case in merged_suite.findall('testcase'):
        attribs['tests'] += 1
        if case.find('failure') is not None:
            attribs['failures'] += 1
        elif case.find('error') is not None:
            attribs['errors'] += 1
        elif case.find('skipped') is not None:
            attribs['skipped'] += 1

    # Set attributes for merged_root
    for key, value in attribs.items():
        merged_root.set(key, str(value))
    merged_root.append(merged_suite)
    
    # Write to new file
    tree = ET.ElementTree(merged_root)
    tree.write(merged_xml_filename, encoding='utf-8', xml_declaration=True)

def xml_to_html(xml_filename, html_filename):
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rerun Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 10px; }}
            .summary {{ margin-bottom: 10px; }}
            .success {{ color: #1e8e3e; }}
            .failure {{ color: #d93025; }}
            .skipped {{ color: #666666; }}
            .testcase {{ 
                border-left: 4px solid #ddd;
                margin: 5px 0;
                background: white;
            }}
            .testcase.success {{ border-left-color: #1e8e3e; }}
            .testcase.failure {{ border-left-color: #d93025; }}
            .testcase.skipped {{ border-left-color: #666666; }}
            .test-time {{
                color: #666;
                font-size: 12px;
                margin-left: 10px;
                float: right;
            }}
            .test-details {{
                padding: 10px;
                background: #f5f5f5;
                border-radius: 3px;
            }}
            pre {{
                margin: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
                background: #2b2b2b;
                color: #cccccc;
                padding: 10px;
                counter-reset: line;
            }}
            pre + pre {{
                border-top: none;
                padding-top: 0;
            }}
            pre span {{
                display: block;
                position: relative;
                padding-left: 4em;
            }}
            pre span:before {{
                counter-increment: line;
                content: counter(line);
                position: absolute;
                left: 0;
                width: 3em;
                text-align: right;
                color: #666;
                padding-right: 1em;
            }}
            details summary {{
                cursor: pointer;
                outline: none;
            }}
            details[open] summary {{
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h2>Rerun Test Results</h2>
        <div class="summary">
            {summary}
        </div>
        <div class="test-cases">
            {test_cases}
        </div>
    </body>
    </html>
    """

    # Parse XML
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    suite = root.find('testsuite')
    
    failed_tests_count = int(suite.attrib.get('failures', 0)) + int(suite.attrib.get('errors', 0))
    skipped_tests_count = int(suite.attrib.get('skipped', 0))
    passed_tests_count = int(suite.attrib.get('tests', 0)) - failed_tests_count - skipped_tests_count

    # Generate summary
    summary = f"""
        <p>Stage: {suite.attrib.get('name', '')}</p>
        <p>Tests: {suite.attrib.get('tests', 0)} | <span class="failure">Failed: {failed_tests_count}</span> | <span class="skipped">Skipped: {skipped_tests_count}</span> | <span class="success">Passed: {passed_tests_count}</span></p>
    """

    # Generate test case details
    test_cases_html = []
    
    # First collect all test cases and sort them by status
    all_test_cases = []
    for testcase in suite.findall('testcase'):
        status = "success"
        if testcase.find('failure') is not None or testcase.find('error') is not None:
            status = "failure"
        elif testcase.find('skipped') is not None:
            status = "skipped"
        all_test_cases.append((status, testcase))
    
    # Sort test cases: failure/error first, then skipped, then success
    status_order = {"failure": 0, "skipped": 1, "success": 2}
    all_test_cases.sort(key=lambda x: status_order[x[0]])
    
    # Generate HTML for sorted test cases
    for status, testcase in all_test_cases:
        test_name = f"{testcase.attrib['name']} - {testcase.attrib.get('classname', 'N/A')}"
        test_time = f"{float(testcase.attrib['time']):.3f}s"
        
        if status == "failure":
            failure = testcase.find('failure') or testcase.find('error')
            system_out = testcase.find('system-out')
            system_err = testcase.find('system-err')
            details = f"""
                <details class="test-details">
                    <summary>{test_name}<span class="test-time">{test_time}</span></summary>
                    <pre>{''.join(f'<span>{line}</span>' for line in failure.get('message', '').splitlines(True))}</pre>
                    <pre>{''.join(f'<span>{line}</span>' for line in (failure.text or '').splitlines(True))}</pre>
                    <pre>{''.join(f'<span>{line}</span>' for line in (system_out.text or '').splitlines(True))}</pre>
                    <pre>{''.join(f'<span>{line}</span>' for line in (system_err.text or '').splitlines(True))}</pre>
                </details>
            """
        elif status == "skipped":
            skipped_message = testcase.find('skipped').attrib.get('message', '')
            details = f"""
                <details class="test-details">
                    <summary>{test_name}<span class="test-time">{test_time}</span></summary>
                    <pre>{''.join(f'<span>{line}</span>' for line in skipped_message.splitlines(True))}</pre>
                </details>
            """
        else:
            system_out = testcase.find('system-out')
            system_err = testcase.find('system-err')
            details = f"""
                <details class="test-details">
                    <summary>{test_name}<span class="test-time">{test_time}</span></summary>
                    <pre>{''.join(f'<span>{line}</span>' for line in (system_out.text or '').splitlines(True))}</pre>
                    <pre>{''.join(f'<span>{line}</span>' for line in (system_err.text or '').splitlines(True))}</pre>
                </details>
            """

        test_case_html = f"""
            <div class="testcase {status}">
                {details}
            </div>
        """
        test_cases_html.append(test_case_html)

    # Generate complete HTML
    html_content = html_template.format(
        summary=summary,
        test_cases="\n".join(test_cases_html)
    )

    # Write to file
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    if (sys.argv[1] == "generate_rerun_tests_list"):
        generate_rerun_tests_list(sys.argv[2], sys.argv[3])
    elif (sys.argv[1] == "merge_junit_xmls"):
        merge_junit_xmls(sys.argv[2], sys.argv[3:])
    elif (sys.argv[1] == "xml_to_html"):
        xml_to_html(sys.argv[2], sys.argv[3])
