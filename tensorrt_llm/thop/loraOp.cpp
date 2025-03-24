mLoraImpl->run(numTokens, numReqs, input.data_ptr(), expandLoraRanks.data(), expandLoraWeightPtrs.data(), weight_index,
    output.data(), workspace.data_ptr(), stream);

sync_check_cuda_error(stream);

TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
return output_torch;
