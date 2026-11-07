// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Run pi-ai 1.4.0's own classifiers over error texts: is it an overflow (compaction) or a
// retryable error (agent-level retry)? usage: PIKIT_PI_DIR=<pi install> node classify.mjs
const piDir = process.env.PIKIT_PI_DIR;
if (!piDir) throw new Error("set PIKIT_PI_DIR to the pi install (the directory holding node_modules)");
const { isContextOverflow, isRetryableAssistantError } = await import(
  `${piDir}/node_modules/@earendil-works/pi-ai/dist/compat.js`);
const texts = {
  "trtllm main, chunked prefill off: max_num_tokens": '400: {"object":"error","message":"The prompt length (40000) should not exceed max_num_tokens (8192)","type":"BadRequestError","param":null,"code":400}',
  "trtllm main, chunked prefill on: default_max_tokens": '400: {"object":"error","message":"`default_max_tokens` (-4096) must be greater than 0, `default_max_tokens` (-4096) = max_seq_len (32768) - `splited_prompt_len` (36864)","type":"BadRequestError","param":null,"code":400}',
  "trtllm C++ maximum input length": "Prompt length (40000) exceeds maximum input length (32767). Set log level to info and check TRTGptModel logs for how maximum input length is set",
  "OpenAI-style (fake overlength_profile)": '400: {"message":"This model\'s maximum context length is 200000 tokens","type":"invalid_request_error","code":"context_length_exceeded"}',
  "Anthropic-style": '400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}',
  "429 (fake http_429)": '429: {"message":"Rate limit reached for requests","type":"requests","param":null,"code":"rate_limit_exceeded"}',
  "400 top-level, 500 inside text": '400: {"object":"error","message":"Internal server error","type":"BadRequestError","param":null,"code":400}',
  "400 plain (fake http_400)": '400: {"message":"Unrecognized request argument supplied: frequency_bias","type":"invalid_request_error","param":null,"code":null}',
};
for (const [name, errorMessage] of Object.entries(texts)) {
  const m = { role: "assistant", stopReason: "error", errorMessage, provider: "trtllm",
              usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const overflow = isContextOverflow(m, 32768);
  const retry = !overflow && isRetryableAssistantError(m);
  console.log(`${name.padEnd(52)} overflow=${overflow} retryable=${retry}`);
}
