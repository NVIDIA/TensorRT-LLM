import openai

from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.scaffolding import TaskStatus
from tensorrt_llm.scaffolding.contrib.mcp.chat_task import ChatTask

ExecutorCls = GenerationExecutor


# helper function
# add first non-None candidate_values to params with key
def add_param_if_not_none(params, key, candidate_values):
    for value in candidate_values:
        if value is not None:
            params[key] = value
            return


def combine_params_with_chat_task(worker, params: dict, task: ChatTask):
    params["messages"] = task.messages

    add_param_if_not_none(params, "max_tokens",
                          [task.max_tokens, worker.max_tokens])
    add_param_if_not_none(params, "temperature",
                          [task.temperature, worker.temperature])
    add_param_if_not_none(params, "top_p", [task.top_p, worker.top_p])

    add_param_if_not_none(params, "tools", [task.tools])


def fill_chat_task_with_response(task: ChatTask, response: openai.Completion):
    task.output_str = response.choices[0].message.content
    task.finish_reason = response.choices[0].finish_reason
    task.tool_calls = response.choices[0].message.tool_calls
    task.logprobs = response.choices[0].logprobs


async def chat_handler(worker, task: ChatTask) -> TaskStatus:
    params = {}
    # Set required parameters
    params["model"] = worker.model

    combine_params_with_chat_task(worker, params, task)

    # Make the API call
    try:
        response = await worker.async_client.chat.completions.create(**params)
        fill_chat_task_with_response(task, response)
        return TaskStatus.SUCCESS

    except Exception as e:
        # Handle errors
        print('Openai chat client get exception: ' + str(e))
        return TaskStatus.WORKER_EXECEPTION
