#include <cstdio>
#include <cuda_runtime.h>

// Example host function
void myHostCallback(void* userData)
{
    printf("Host callback executed: %s\n", static_cast<char*>(userData));
}

int main()
{
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    char message[] = "Hello 0 from host node!";

    cudaStreamCreate(&stream);

    // Begin CUDA stream capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // [Device work would be enqueued here, e.g. kernel launches, memcpys]

    // Schedule host function as a host node in the graph
    cudaLaunchHostFunc(stream, myHostCallback, message);

    // End capture; the graph now includes the host node
    cudaStreamEndCapture(stream, &graph);

    // Instantiate executable graph
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Launch the graph; 'myHostCallback' will execute as a host node
    for (int i = 0; i < 10; i++)
    {
        message[6] = '0' + i;
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
