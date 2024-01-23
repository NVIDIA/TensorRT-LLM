# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

import torch
from run import QWenInfer, parse_arguments, vit_process


def make_display(port=8006):
    import cv2
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    def func(image):
        data = cv2.imencode(".jpg", image)[1].tobytes()
        socket.recv()
        socket.send(data)

    return func


def show_pic(image_path, port):
    import cv2
    image = cv2.imread(image_path)
    display_obj = make_display(port)
    display_obj(image)


def show_pic_local(image_path):
    import cv2
    import matplotlib.pyplot as plt
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.pause(0.1)


def cooridinate_extract_show(input, history, tokenizer, local_machine, port):
    pattern = r"\((\d+),(\d+)\)"
    coordinates = re.findall(pattern, input)
    result = "<ref>Box</ref><box>({},{})".format(coordinates[0][0],
                                                 coordinates[0][1])
    result += ",({},{})</box>".format(coordinates[1][0], coordinates[1][1])

    image = tokenizer.draw_bbox_on_latest_picture(result, history)
    if image:
        image.save('1.png')
        if local_machine:
            show_pic_local('1.png')
        else:
            show_pic('1.png', port)
    else:
        print("======No bounding boxes are detected!")


def exist_cooridinate(input):
    pattern = r"\((\d+),(\d+)\)"
    match = re.search(pattern, input)
    if match:
        return True
    else:
        return False


if __name__ == '__main__':
    args = parse_arguments()
    stream = torch.cuda.current_stream().cuda_stream
    image_embeds = vit_process(args.input_dir, args.vit_engine_dir,
                               args.log_level, stream)
    qinfer = QWenInfer(args.tokenizer_dir, args.qwen_engine_dir, args.log_level,
                       args.output_csv, args.output_npy, args.num_beams)
    qinfer.qwen_model_init()

    run_i = 0
    history = []
    if args.display:
        if args.local_machine:
            show_pic_local("./pics/demo.jpeg")
        else:
            show_pic("./pics/demo.jpeg", args.port)

    while True:
        input_text = None
        try:
            input_text = input("Text (or 'q' to quit): ")
        except:
            continue

        if input_text == "clear history":
            history = []
            continue

        if input_text.lower() == 'q':
            break
        print('\n')

        content_list = args.images_path
        content_list.append({'text': input_text})

        if run_i == 0:
            query = qinfer.tokenizer.from_list_format(content_list)
        else:
            query = input_text

        run_i = run_i + 1
        output_text = qinfer.qwen_infer(image_embeds, None, query,
                                        args.max_new_tokens, history)
        if args.display:
            if exist_cooridinate(output_text):
                cooridinate_extract_show(output_text, history, qinfer.tokenizer,
                                         args.local_machine, args.port)
