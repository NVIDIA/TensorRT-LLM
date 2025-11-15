import argparse
import json
import os
import sys

import requests

OPEN_SEARCH_QUERY_URL = (
    "http://gpuwa.nvidia.com/opensearch/df-swdl-trtllm-infra-ci-prod-test_info-*/_search"
)
headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}


def queryJobEvents(commitID="", stageName="", onlySuccess=True):
    mustConditions = []
    if commitID:
        mustConditions.append({"term": {"s_trigger_mr_commit": commitID}})
    if stageName:
        mustConditions.append({"term": {"s_stage_name": stageName}})
    if onlySuccess:
        mustConditions.append({"term": {"s_status": "PASSED"}})

    all_results = []
    page_size = 1000
    from_index = 0

    while True:
        requestBody = {
            "query": {"bool": {"must": mustConditions}},
            "_source": [
                "s_job_name",
                "s_status",
                "s_build_id",
                "s_turtle_name",
                "s_test_name",
                "s_gpu_type",
            ],
            "size": page_size,
            "from": from_index,
            "sort": [{"_id": "asc"}],
        }

        formattedRequestBody = json.dumps(requestBody)
        response = requests.post(OPEN_SEARCH_QUERY_URL, headers=headers, data=formattedRequestBody)
        data = response.json()

        hits = data["hits"]["hits"]
        if not hits:
            break

        all_results.extend(hits)
        from_index += page_size

        print(f"Fetched {len(all_results)} records...")

    return all_results


def writeTestListToFile(testList, fileName):
    os.makedirs(os.path.dirname(fileName), exist_ok=True)

    with open(fileName, "w") as f:
        for test in testList:
            f.write(test + "\n")


def getPassedTestList(commitID, stageName, outputFile):
    hits = queryJobEvents(commitID=commitID, stageName=stageName, onlySuccess=True)
    testList = []
    for hit in hits:
        testList.append(hit["_source"]["s_turtle_name"])
    writeTestListToFile(testList, outputFile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-id", required=True, help="Commit ID")
    parser.add_argument("--stage-name", required=True, help="Stage Name")
    parser.add_argument("--output-file", required=True, help="Output File")
    args = parser.parse_args(sys.argv[1:])
    getPassedTestList(
        commitID=args.commit_id, stageName=args.stage_name, outputFile=args.output_file
    )
