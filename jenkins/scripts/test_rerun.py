import argparse
import os
import sys
import xml.etree.ElementTree as ET


def parse_name(classname, name, filename):
    if "test_unittests_v2[unittest/" in name and \
       filename == "test_unittests.py":
        return name[name.find("test_unittests_v2[unittest/") + 18:-1]
    elif filename in name:
        return name[name.find('/') + 1:]
    elif filename[:-2].replace("/", ".") in classname:
        return filename + "::" + classname.split(".")[-1] + "::" + name
    else:
        return filename + "::" + name


def generate_rerun_tests_list(outdir, xml_filename, failSignaturesList):
    # Generate rerun test lists:
    # 1. Parse the test results xml file
    # 2. For failed tests:
    #    - If test duration <= 5 min: add to rerun_2.txt (will rerun 2 times)
    #    - If test duration > 5 min and <= 10 min: add to rerun_1.txt (will rerun 1 time)
    #    - If test duration > 10 min but contains fail signatures in error message: add to rerun_1.txt
    #    - If test duration > 10 min and no known failure signatures: add to rerun_0.txt (will not rerun)
    print(failSignaturesList)

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
            if case.find('failure') is not None or \
               case.find('error') is not None:
                duration = float(case.attrib.get('time', 0))
                test_name = parse_name(case.attrib.get('classname', ''), \
                                       case.attrib.get('name', ''), \
                                       case.attrib.get('file', ''))
                if duration <= 5 * 60:
                    rerun_2_file.write(test_name + '\n')
                    print(test_name + " will rerun 2 times")
                elif duration <= 10 * 60:
                    rerun_1_file.write(test_name + '\n')
                    print(test_name + " will rerun 1 time")
                elif any(failSig.lower() in ET.tostring(
                        case, encoding='unicode').lower()
                         for failSig in failSignaturesList):
                    rerun_1_file.write(test_name + '\n')
                    print(test_name +
                          " will rerun 1 time, because of fail signature")
                else:
                    rerun_0_file.write(test_name + '\n')
                    print(test_name + " will not rerun")

    # Remove empty files
    for filename in [rerun_0_filename, rerun_1_filename, rerun_2_filename]:
        if os.path.getsize(filename) == 0:
            os.remove(filename)


def merge_junit_xmls(merged_xml_filename, xml_filenames, deduplicate=False):
    # Merge xml files into one.
    # If deduplicate is true, remove duplicate test cases.
    merged_root = ET.Element('testsuites')
    merged_suite_map = {}

    for xml_filename in xml_filenames:
        if not os.path.exists(xml_filename):
            continue

        suites = ET.parse(xml_filename).getroot()
        suite_list = suites.findall('testsuite')
        for suite in suite_list:
            suite_name = suite.attrib.get('name', '')
            if suite_name not in merged_suite_map:
                merged_suite_map[suite_name] = suite
            else:
                original_suite = merged_suite_map[suite_name]
                case_list = suite.findall('testcase')
                for case in case_list:
                    existing_case = original_suite.find(
                        f"testcase[@name='{case.attrib['name']}'][@classname='{case.attrib['classname']}']"
                    )
                    # find the duplicate case in original_suite
                    if existing_case is not None:
                        if deduplicate:
                            # remove the duplicate case in original_suite
                            original_suite.remove(existing_case)
                        else:
                            # add rerun flag to the new case for rerun report
                            case.set('isrerun', 'true')
                original_suite.extend(case_list)

    # Update suite attributes
    for suite in merged_suite_map.values():
        attribs = {'errors': 0, 'failures': 0, 'skipped': 0, 'tests': 0}
        for case in suite.findall('testcase'):
            attribs['tests'] += 1
            if case.find('failure') is not None:
                attribs['failures'] += 1
            elif case.find('error') is not None:
                attribs['errors'] += 1
            elif case.find('skipped') is not None:
                attribs['skipped'] += 1
        for key, value in attribs.items():
            suite.set(key, str(value))

        # add suite to merged_root
        merged_root.append(suite)

    if os.path.exists(merged_xml_filename):
        os.remove(merged_xml_filename)

    # Write to new file
    tree = ET.ElementTree(merged_root)
    tree.write(merged_xml_filename, encoding='utf-8', xml_declaration=True)


def escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def xml_to_html(xml_filename, html_filename, sort_by_name=False):
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rerun Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 10px; }}
            .suite-container {{
                margin-bottom: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .suite-header {{
                padding: 10px;
                background: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }}
            .summary {{ margin-bottom: 10px; }}
            .success {{ color: #1e8e3e; }}
            .failure {{ color: #d93025; }}
            .error {{ color: #d93025; }}
            .skipped {{ color: #666666; }}
            .testcase {{
                border-left: 4px solid #ddd;
                margin: 5px 0;
                background: white;
            }}
            .testcase.success {{ border-left-color: #1e8e3e; }}
            .testcase.failure {{ border-left-color: #d93025; }}
            .testcase.error {{ border-left-color: #d93025; }}
            .testcase.skipped {{ border-left-color: #666666; }}
            .test-time {{
                color: #666;
                font-size: 12px;
                margin-left: 10px;
                float: right;
            }}
            .test-rerun-sig {{
                font-size: 12px;
                color: #666;
                margin-left: 5px;
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
        {test_suites}
    </body>
    </html>
    """

    # Parse XML
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    suite_list = root.findall('testsuite')

    all_suites_html = []

    for suite in suite_list:
        tests_count = int(suite.attrib.get('tests', 0))
        failed_tests_count = int(suite.attrib.get('failures', 0)) + \
                         int(suite.attrib.get('errors', 0))
        skipped_tests_count = int(suite.attrib.get('skipped', 0))
        passed_tests_count = tests_count - failed_tests_count - skipped_tests_count

        # Generate summary for the suite
        summary = f"""
            <div class="suite-header">
                <h3>Stage: {suite.attrib.get('name', '')}</h3>
                <p>Tests: {tests_count} |
                   <span class="failure">Failed: {failed_tests_count}</span> |
                   <span class="skipped">Skipped: {skipped_tests_count}</span> |
                   <span class="success">Passed: {passed_tests_count}</span>
                </p>
            </div>
        """

        # Generate test case details for the suite
        test_cases_html = []
        all_test_cases = []

        for testcase in suite.findall('testcase'):
            status = "success"
            if testcase.find('failure') is not None:
                status = "failure"
            elif testcase.find('error') is not None:
                status = "error"
            elif testcase.find('skipped') is not None:
                status = "skipped"
            all_test_cases.append((status, testcase))

        if sort_by_name:
            all_test_cases.sort(key=lambda x: x[1].attrib.get('name', '') + \
                                x[1].attrib.get('classname', ''))
        else:
            # Sort test cases: failure/error first, then skipped, then success
            status_order = {
                "failure": 0,
                "error": 1,
                "skipped": 2,
                "success": 3
            }
            all_test_cases.sort(key=lambda x: status_order[x[0]])

        # Generate HTML for sorted test cases
        for status, testcase in all_test_cases:
            test_rerun_sig = '[RERUN]' if testcase.attrib.get(
                'isrerun', '') == 'true' else ''
            test_name = f"{testcase.attrib.get('name', '')} - {testcase.attrib.get('classname', '')}"
            test_time = f"{float(testcase.attrib.get('time', 0)):.3f}s"

            if status == "failure":
                failure = testcase.find('failure')
                system_out = testcase.find('system-out')
                system_err = testcase.find('system-err')
                details = f"""
                    <details class="test-details">
                        <summary>{test_name}<span class="test-rerun-sig">{test_rerun_sig}</span><span class="test-time">{test_time}</span></summary>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in failure.get('message', '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (failure.text or '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_out.text or '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_err.text or '').splitlines(True))}</pre>
                    </details>
                """
            elif status == "error":
                error = testcase.find('error')
                system_out = testcase.find('system-out')
                system_err = testcase.find('system-err')
                details = f"""
                    <details class="test-details">
                        <summary>{test_name}<span class="test-rerun-sig">{test_rerun_sig}</span><span class="test-time">{test_time}</span></summary>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in error.get('message', '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (error.text or '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_out.text or '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_err.text or '').splitlines(True))}</pre>
                    </details>
                """
            elif status == "skipped":
                skipped_message = testcase.find('skipped').attrib.get(
                    'message', '')
                details = f"""
                    <details class="test-details">
                        <summary>{test_name}<span class="test-rerun-sig">{test_rerun_sig}</span><span class="test-time">{test_time}</span></summary>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in skipped_message.splitlines(True))}</pre>
                    </details>
                """
            else:
                system_out = testcase.find('system-out')
                system_err = testcase.find('system-err')
                details = f"""
                    <details class="test-details">
                        <summary>{test_name}<span class="test-rerun-sig">{test_rerun_sig}</span><span class="test-time">{test_time}</span></summary>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_out.text or '').splitlines(True))}</pre>
                        <pre>{''.join(f'<span>{escape_html(line)}</span>' for line in (system_err.text or '').splitlines(True))}</pre>
                    </details>
                """

            test_case_html = f"""
                <div class="testcase {status}">
                    {details}
                </div>
            """
            test_cases_html.append(test_case_html)

        # Combine summary and test cases for this suite
        suite_html = f"""
            <div class="suite-container">
                {summary}
                <div class="test-cases">
                    {' '.join(test_cases_html)}
                </div>
            </div>
        """
        all_suites_html.append(suite_html)

    # Generate complete HTML
    html_content = html_template.format(test_suites='\n'.join(all_suites_html))

    # Write to file
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)


def filter_failed_tests(xml_filename, output_filename):
    # Filter failed tests from the xml file
    filtered_root = ET.Element('testsuites')

    root = ET.parse(xml_filename).getroot()
    suite_list = root.findall('testsuite')
    for suite in suite_list:
        filtered_suite = ET.Element('testsuite')
        # Copy attributes one by one
        for key, value in suite.attrib.items():
            filtered_suite.set(key, value)
        filtered_suite.set(
            'tests',
            str(
                int(suite.attrib.get('failures', '0')) +
                int(suite.attrib.get('errors', '0'))))
        filtered_suite.set('skipped', '0')

        filtered_cases = []
        for case in suite.findall('testcase'):
            if case.find('failure') is not None or \
               case.find('error') is not None:
                filtered_cases.append(case)

        filtered_suite.extend(filtered_cases)
        filtered_root.append(filtered_suite)

    tree = ET.ElementTree(filtered_root)
    tree.write(output_filename, encoding='utf-8', xml_declaration=True)


def generate_rerun_report(output_filename, input_filenames):
    # Merge the input xml files (filter failed tests for results.xml)
    # and generate the rerun report html file.
    new_filename_list = []
    for input_filename in input_filenames:
        new_filename = input_filename
        if "/results.xml" in input_filename:
            new_filename = input_filename.replace("results.xml",
                                                  "failed_results.xml")
            filter_failed_tests(input_filename, new_filename)
        new_filename_list.append(new_filename)

    print(new_filename_list)
    merge_junit_xmls(output_filename, new_filename_list)
    xml_to_html(output_filename,
                output_filename.replace(".xml", ".html"),
                sort_by_name=True)


if __name__ == '__main__':
    if (sys.argv[1] == "generate_rerun_tests_list"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-dir',
                            required=True,
                            help='Output directory for rerun test lists')
        parser.add_argument('--input-file',
                            required=True,
                            help='Input XML file containing test results')
        parser.add_argument('--fail-signatures',
                            required=True,
                            help='List of failure signatures to match')
        args = parser.parse_args(sys.argv[2:])
        failSignaturesList = args.fail_signatures.split(',')
        generate_rerun_tests_list(args.output_dir, args.input_file,
                                  failSignaturesList)

    elif (sys.argv[1] == "merge_junit_xmls"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-file',
                            required=True,
                            help='Output XML file')
        parser.add_argument('--input-files',
                            required=True,
                            help='Input XML files to merge')
        parser.add_argument('--deduplicate',
                            action='store_true',
                            help='Deduplicate test cases')
        args = parser.parse_args(sys.argv[2:])
        input_files = args.input_files.split(',')
        deduplicate = args.deduplicate
        merge_junit_xmls(args.output_file, input_files, deduplicate)

    elif (sys.argv[1] == "xml_to_html"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input-file',
                            required=True,
                            help='Input XML file')
        parser.add_argument('--output-file',
                            required=True,
                            help='Output HTML file')
        parser.add_argument('--sort-by-name',
                            action='store_true',
                            help='Sort test cases by name')
        args = parser.parse_args(sys.argv[2:])
        xml_to_html(args.input_file, args.output_file, args.sort_by_name)

    elif (sys.argv[1] == "generate_rerun_report"):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output-file',
                            required=True,
                            help='Output XML file')
        parser.add_argument('--input-files',
                            required=True,
                            help='Input XML files to merge')
        args = parser.parse_args(sys.argv[2:])
        input_files = args.input_files.split(',')
        generate_rerun_report(args.output_file, input_files)
