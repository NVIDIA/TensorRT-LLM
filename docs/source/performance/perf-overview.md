(perf-overview)=

> [!IMPORTANT]
> As of TensorRT-LLM v0.10, these performance benchmarks have changed methodology to utilize in-flight batching and
no longer utilize static benchmarking. These numbers are initial measurements and are expected to improve in future
releases.

# Overview

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp=enable` is enabled). There is also a more
efficient implementation that runs single Matmul + SwiGLU fused kernel for FP8 on Hopper
(when `--use_fused_mlp=enable --gemm_swiglu_plugin fp8` is enabled). The gemm_swiglu_plugin
will support more data types and GPU architectures in the future release.

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput client-server scenario under maximum load.


The performance numbers below were collected using the steps described in this document.

**All data in the table below was generated using version 0.13.0 and presents token throughput in tokens/second.**

|                 |                          |               |                     |                 |                    |                    |                    |          |
| --------------- | ------------------------ | ------------- | ------------------- | --------------- | ------------------ | ------------------ | ------------------ | -------- |
|                 |                          | **GPU**       | **H200 141GB HBM3** | **GH200 120GB** | **H100 80GB HBM3** | **H100 80GB HBM3** | **A100-SXM4-80GB** | **L40S** |
|                 |                          | **Precision** | **FP8**             | **FP8**         | **FP8**            | **FP16**           | **FP16**           | **FP8**  |
| **Model**       | **Input/Output Lengths** | **TP**        |                     |                 |                    |                    |                    |          |
| GPTJ 6B         | 128/128                  | 1             | 24,533.54           | 22,368.50       | 24,318.61          | 12,936.63          | 5,964.19           | 7,688.44 |
|                 | 128/2048                 | 1             | 8,375.67            | 6,588.73        | 7,829.91           | 3,931.61           | 2,215.88           | 1,842.82 |
|                 | 128/4096                 | 1             | 5,048.59            | 3,662.81        | 3,955.28           | 2,041.06           | 1,118.12           | 980.23   |
|                 | 2048/128                 | 1             | 2,770.27            | 2,520.37        | 2,698.08           | 1,479.48           | 650.09             | 746.54   |
|                 | 5000/500                 | 1             | 1,791.39            | 1,449.23        | 1,623.17           | 818.80             | 436.85             | 413.33   |
|                 | 500/2000                 | 1             | 6,770.60            | 5,565.62        | 6,149.65           | 3,030.03           | 1,673.05           | 1,538.45 |
|                 | 1000/1000                | 1             | 6,465.73            | 5,580.37        | 6,078.80           | 2,797.48           | 1,673.45           | 1,531.57 |
|                 | 2048/2048                | 1             | 3,637.42            | 2,998.01        | 3,060.80           | 1,285.08           | 845.83             | 753.55   |
| LLaMA v3.1 8B   | 128/128                  | 1             | 28,125.59           | 26,045.60       | 27,147.22          | 15,647.83          | 6,687.04           | 8,548.90 |
|                 | 128/2048                 | 1             | 22,989.20           | 16,497.79       | 19,221.02          | 8,882.95           | 4,918.53           | 4,988.61 |
|                 | 128/4096                 | 1             | 16,077.62           | 9,637.91        | 11,856.11          | 5,462.96           | 3,054.46           | 2,768.91 |
|                 | 2048/128                 | 1             | 3,625.83            | 3,357.60        | 3,497.30           | 1,859.37           | 796.17             | 1,000.90 |
|                 | 5000/500                 | 1             | 3,823.76            | 3,217.40        | 3,276.69           | 1,687.74           | 788.66             | 872.14   |
|                 | 500/2000                 | 1             | 19,382.37           | 15,128.77       | 13,996.05          | 6,834.76           | 3,929.83           | 3,911.14 |
|                 | 1000/1000                | 1             | 16,435.21           | 12,355.41       | 13,411.43          | 7,160.92           | 3,592.16           | 3,648.21 |
|                 | 2048/2048                | 1             | 11,072.97           | 7,850.75        | 8,851.23           | 4,152.21           | 2,269.78           | 2,055.78 |
|                 | 20000/2000               | 1             | 1,634.98            | 1,200.89        | 1,278.04           | 595.89             | 316.43             | 263.75   |
| LLaMA v3 8B     | 128/128                  | 1             | 27,940.47           | 26,117.13       | 27,156.81          | 15,489.11          | 6,656.98           | 8,734.57 |
|                 | 128/2048                 | 1             | 23,228.98           | 16,417.04       | 19,209.17          | 8,901.43           | 4,967.37           | 5,004.93 |
|                 | 128/4096                 | 1             | 15,980.94           | 9,351.95        | 11,889.67          | 5,455.91           | 3,053.27           | 2,768.15 |
|                 | 2048/128                 | 1             | 3,631.45            | 3,339.90        | 3,476.37           | 1,918.56           | 796.28             | 1,050.68 |
|                 | 5000/500                 | 1             | 3,836.98            | 3,186.22        | 3,279.24           | 1,668.42           | 792.95             | 860.31   |
|                 | 500/2000                 | 1             | 19,725.45           | 15,241.74       | 14,218.30          | 6,816.62           | 3,899.64           | 3,990.73 |
|                 | 1000/1000                | 1             | 16,201.60           | 12,049.81       | 13,371.60          | 7,041.47           | 3,617.10           | 3,679.10 |
|                 | 2048/2048                | 1             | 11,097.69           | 7,255.55        | 8,852.87           | 4,251.45           | 2,269.68           | 2,048.94 |
| LLaMA v2 7B     | 128/128                  | 1             | 19,549.13           | 17,823.45       | 19,298.99          | 11,436.31          | 5,238.68           | 6,396.62 |
|                 | 128/2048                 | 1             | 7,675.14            | 5,438.53        | 6,607.33           | 2,985.61           | 1,807.39           | 1,566.03 |
|                 | 128/4096                 | 1             | 4,397.83            | 3,310.09        | 3,628.46           | 1,575.35           | 957.24             | 821.83   |
|                 | 2048/128                 | 1             | 2,392.31            | 2,064.18        | 2,304.02           | 1,157.55           | 560.35             | 619.83   |
|                 | 5000/500                 | 1             | 1,570.37            | 1,250.11        | 1,419.09           | 624.75             | 366.39             | 347.03   |
|                 | 500/2000                 | 1             | 6,044.15            | 4,717.51        | 5,188.69           | 2,382.75           | 1,408.58           | 1,231.78 |
|                 | 1000/1000                | 1             | 5,896.10            | 4,825.24        | 5,208.97           | 2,462.65           | 1,431.92           | 1,277.79 |
|                 | 2048/2048                | 1             | 3,193.42            | 2,693.21        | 2,792.53           | 1,263.11           | 734.38             | 641.47   |
| Mistral 7B      | 128/128                  | 1             | 30,152.19           | 27,738.08       | 29,672.75          | 16,711.12          | 6,863.59           | 9,676.88 |
|                 | 128/2048                 | 1             | 24,742.09           | 17,528.14       | 20,318.60          | 9,774.11           | 5,321.44           | 5,437.25 |
|                 | 128/4096                 | 1             | 16,905.49           | 10,671.38       | 12,715.46          | 5,740.41           | 3,257.23           | 2,941.08 |
|                 | 2048/128                 | 1             | 3,676.37            | 3,369.77        | 3,502.83           | 1,893.42           | 796.00             | 996.65   |
|                 | 5000/500                 | 1             | 3,890.07            | 3,401.45        | 3,358.65           | 1,740.69           | 807.07             | 904.45   |
|                 | 500/2000                 | 1             | 20,788.70           | 15,035.59       | 15,962.94          | 7,494.80           | 4,168.89           | 4,088.52 |
|                 | 1000/1000                | 1             | 17,620.46           | 13,362.84       | 14,213.48          | 7,281.07           | 3,794.31           | 3,972.63 |
|                 | 2048/2048                | 1             | 11,747.88           | 8,599.03        | 9,200.19           | 4,349.39           | 2,320.50           | 2,170.16 |
|                 | 20000/2000               | 1             | 1,693.41            | 1,271.85        | 1,299.05           | 609.91             | 324.52             | 276.19   |
| LLaMA v3.1 405B | 128/128                  | 8             | 3,734.50            |                 |                    |                    |                    |          |
|                 | 128/2048                 | 8             | 3,039.70            |                 |                    |                    |                    |          |
|                 | 128/4096                 | 8             | 3,144.97            |                 |                    |                    |                    |          |
|                 | 2048/128                 | 8             | 454.17              |                 |                    |                    |                    |          |
|                 | 5000/500                 | 8             | 459.91              |                 |                    |                    |                    |          |
|                 | 500/2000                 | 8             | 2,967.98            |                 |                    |                    |                    |          |
|                 | 1000/1000                | 8             | 2,259.32            |                 |                    |                    |                    |          |
|                 | 2048/2048                | 8             | 2,067.15            |                 |                    |                    |                    |          |
|                 | 20000/2000               | 8             | 447.67              |                 |                    |                    |                    |          |
| LLaMA v3.1 70B  | 128/128                  | 1             | 3,923.61            | 2,998.99        | 2,168.72           |                    |                    |          |
|                 |                          | 2             | 5,358.16            | 1,839.02        | 5,215.12           | 3,156.10           | 1,340.20           |          |
|                 |                          | 4             | 8,969.59            | 8,655.98        | 8,677.59           | 5,845.53           | 2,426.46           | 1,434.63 |
|                 |                          | 8             | 16,449.68           |                 | 15,711.60          | 10,643.75          | 4,491.42           | 1,365.36 |
|                 | 128/2048                 | 1             | 3,503.59            | 1,343.53        | 344.22             |                    |                    |          |
|                 |                          | 2             | 7,068.42            | 1,146.08        | 5,654.43           | 801.82             | 498.44             |          |
|                 |                          | 4             | 12,890.95           | 10,358.10       | 9,377.87           | 4,791.11           | 2,460.91           | 1,748.87 |
|                 |                          | 8             | 19,947.02           |                 | 15,168.97          | 6,892.18           | 4,148.33           | 1,890.62 |
|                 | 128/4096                 | 1             | 2,314.83            |                 |                    |                    |                    |          |
|                 |                          | 2             | 6,227.19            | 896.56          | 3,302.41           | 413.22             | 268.86             |          |
|                 |                          | 4             | 10,059.64           | 6,628.22        | 6,501.69           | 3,056.98           | 1,660.93           | 1,180.87 |
|                 |                          | 8             | 14,393.28           |                 | 9,699.99           | 4,238.15           | 2,705.77           | 1,417.60 |
|                 | 2048/128                 | 1             | 459.73              | 372.44          | 211.51             |                    |                    |          |
|                 |                          | 2             | 689.30              | 280.61          | 690.05             | 323.66             | 143.39             |          |
|                 |                          | 4             | 1,047.96            | 1,015.14        | 1,016.24           | 672.37             | 278.87             | 167.87   |
|                 |                          | 8             | 2,061.19            |                 | 1,964.49           | 1,273.97           | 539.57             | 163.91   |
|                 | 5000/500                 | 1             | 534.79              | 283.19          | 112.21             |                    |                    |          |
|                 |                          | 2             | 943.78              | 337.04          | 897.36             | 224.31             | 115.63             |          |
|                 |                          | 4             | 1,437.45            | 1,383.61        | 1,329.82           | 851.12             | 361.39             | 235.90   |
|                 |                          | 8             | 2,795.95            |                 | 2,472.69           | 1,438.10           | 679.27             | 224.33   |
|                 | 500/2000                 | 1             | 2,758.24            | 1,083.48        |                    |                    |                    |          |
|                 |                          | 2             | 6,063.53            | 851.46          | 4,347.69           | 652.34             | 423.06             |          |
|                 |                          | 4             | 10,061.89           | 9,090.78        | 8,378.16           | 3,441.34           | 2,072.88           | 1,436.41 |
|                 |                          | 8             | 16,139.49           |                 | 10,790.85          | 5,792.17           | 3,115.20           | 1,512.78 |
|                 | 1000/1000                | 1             | 2,539.65            | 728.79          |                    |                    |                    |          |
|                 |                          | 2             | 4,572.03            | 1,223.92        | 3,880.41           | 737.40             | 451.82             |          |
|                 |                          | 4             | 7,612.56            | 6,705.02        | 6,553.00           | 3,655.64           | 1,731.86           | 1,113.18 |
|                 |                          | 8             | 12,660.86           |                 | 11,121.10          | 5,599.45           | 3,013.95           | 1,120.73 |
|                 | 2048/2048                | 1             | 1,753.58            | 611.08          | 161.60             |                    |                    |          |
|                 |                          | 2             | 3,407.26            | 626.26          | 2,432.55           |                    | 108.91             |          |
|                 |                          | 4             | 6,565.77            | 4,864.55        | 4,948.83           | 2,396.06           | 1,220.93           | 855.44   |
|                 |                          | 8             | 9,948.56            |                 | 8,527.52           | 3,819.60           | 2,103.68           | 924.89   |
|                 | 20000/2000               | 1             | 262.82              | 88.89           |                    |                    |                    |          |
|                 |                          | 2             | 598.19              | 177.04          | 414.17             |                    |                    |          |
|                 |                          | 4             | 1,047.27            | 958.88          | 856.31             | 375.85             | 187.42             | 140.73   |
|                 |                          | 8             | 1,793.52            |                 | 1,359.27           | 650.78             | 344.41             | 122.04   |
| LLaMA v3 70B    | 128/128                  | 1             | 3,924.02            | 3,161.73        | 2,177.84           |                    |                    |          |
|                 |                          | 2             | 5,388.22            | 1,551.84        | 5,205.80           | 3,186.61           | 1,321.55           |          |
|                 |                          | 4             | 8,958.95            | 8,618.55        | 8,678.68           | 5,857.16           | 2,424.68           | 1,432.46 |
|                 |                          | 8             | 16,375.41           |                 | 15,703.26          | 10,627.36          | 4,490.19           | 1,333.09 |
|                 | 128/2048                 | 1             | 3,519.24            | 1,346.37        | 353.68             |                    |                    |          |
|                 |                          | 2             | 7,071.54            | 862.54          | 5,878.06           | 802.98             | 512.11             |          |
|                 |                          | 4             | 12,876.38           | 10,015.23       | 8,929.23           | 4,768.27           | 2,458.73           | 1,737.31 |
|                 |                          | 8             | 20,013.92           |                 | 15,171.91          | 6,875.97           | 3,906.35           | 1,892.41 |
|                 | 128/4096                 | 1             | 2,310.85            |                 |                    |                    |                    |          |
|                 |                          | 2             | 6,199.95            | 602.98          | 3,311.05           | 413.29             | 269.02             |          |
|                 |                          | 4             | 9,633.49            | 7,370.19        | 6,489.95           | 3,053.89           | 1,677.51           | 1,199.71 |
|                 |                          | 8             | 14,552.09           |                 | 9,632.02           | 4,259.39           | 2,697.61           | 1,358.34 |
|                 | 2048/128                 | 1             | 458.75              | 371.70          | 210.27             |                    |                    |          |
|                 |                          | 2             | 694.00              | 277.85          | 692.74             | 321.71             | 144.61             |          |
|                 |                          | 4             | 1,048.84            | 1,016.03        | 1,022.77           | 690.10             | 279.06             | 168.52   |
|                 |                          | 8             | 2,072.33            |                 | 1,976.76           | 1,273.41           | 542.93             | 158.63   |
|                 | 5000/500                 | 1             | 533.37              | 303.33          | 112.68             |                    |                    |          |
|                 |                          | 2             | 936.82              | 379.62          | 899.29             | 224.65             | 115.00             |          |
|                 |                          | 4             | 1,442.76            | 1,384.62        | 1,326.95           | 853.73             | 361.06             | 235.19   |
|                 |                          | 8             | 2,797.36            |                 | 2,483.56           | 1,437.15           | 678.70             | 225.15   |
|                 | 500/2000                 | 1             | 2,763.89            | 1,074.62        | 293.47             |                    |                    |          |
|                 |                          | 2             | 6,054.46            | 1,109.13        | 4,356.55           | 683.11             | 423.82             |          |
|                 |                          | 4             | 10,103.08           | 7,325.93        | 8,370.32           | 3,436.29           | 2,064.47           | 1,412.78 |
|                 |                          | 8             | 16,857.45           |                 | 10,760.65          | 5,665.02           | 3,159.89           | 1,517.76 |
|                 | 1000/1000                | 1             | 2,540.45            | 1,164.45        |                    |                    |                    |          |
|                 |                          | 2             | 4,590.38            | 1,040.64        | 3,879.25           | 768.53             | 453.73             |          |
|                 |                          | 4             | 7,606.92            | 6,655.61        | 6,547.23           | 3,655.19           | 1,732.86           | 1,117.53 |
|                 |                          | 8             | 12,660.32           |                 | 11,155.47          | 5,617.24           | 2,894.58           | 1,126.50 |
|                 | 2048/2048                | 1             | 1,746.77            | 610.87          | 162.10             |                    |                    |          |
|                 |                          | 2             | 3,405.72            | 738.51          | 2,548.70           |                    | 108.66             |          |
|                 |                          | 4             | 6,571.34            | 4,880.28        | 5,060.39           | 2,391.55           | 1,222.11           | 854.65   |
|                 |                          | 8             | 9,923.96            |                 | 8,480.48           | 3,826.38           | 2,181.07           | 927.54   |
| LLaMA v2 70B    | 128/128                  | 1             | 3,969.25            | 3,502.35        | 3,413.82           |                    |                    |          |
|                 |                          | 2             | 6,394.64            | 3,252.69        | 6,432.82           | 3,170.28           | 1,336.48           |          |
|                 |                          | 4             | 11,031.42           | 11,126.95       | 10,865.42          | 6,420.88           | 2,766.00           | 1,487.71 |
|                 |                          | 8             | 17,060.04           |                 | 16,384.83          | 11,146.15          | 4,742.74           | 1,404.99 |
|                 | 128/2048                 | 1             | 3,742.99            | 1,660.81        |                    |                    |                    |          |
|                 |                          | 2             | 6,453.25            | 1,335.80        | 5,775.34           | 757.21             | 476.46             |          |
|                 |                          | 4             | 13,869.67           | 11,098.69       | 9,536.82           | 5,274.27           | 2,686.16           | 1,880.22 |
|                 |                          | 8             | 19,220.48           |                 | 17,715.01          | 8,904.94           | 5,520.41           | 2,186.68 |
|                 | 128/4096                 | 1             | 2,459.63            |                 | 446.60             |                    |                    |          |
|                 |                          | 2             | 4,831.03            | 684.68          | 3,354.60           | 385.98             | 235.22             |          |
|                 |                          | 4             | 8,988.84            | 8,397.13        | 7,619.62           | 3,228.36           | 1,941.07           | 1,318.51 |
|                 |                          | 8             | 15,115.41           |                 | 12,506.95          | 5,996.81           | 3,539.36           | 1,782.93 |
|                 | 2048/128                 | 1             | 458.88              | 400.31          | 328.90             |                    |                    |          |
|                 |                          | 2             | 745.71              | 457.57          | 742.17             | 308.02             | 138.81             |          |
|                 |                          | 4             | 1,297.10            | 1,330.90        | 1,270.78           | 755.30             | 321.72             | 171.67   |
|                 |                          | 8             | 2,060.53            |                 | 2,009.57           | 1,348.71           | 561.71             | 160.37   |
|                 | 5000/500                 | 1             | 548.46              | 364.00          | 224.17             |                    |                    |          |
|                 |                          | 2             | 1,020.86            | 335.07          | 885.67             | 212.20             | 112.43             |          |
|                 |                          | 4             | 1,759.69            | 1,683.26        | 1,590.94           | 837.57             | 386.78             | 231.54   |
|                 |                          | 8             | 2,839.69            |                 | 2,546.12           | 1,570.91           | 709.66             | 238.59   |
|                 | 500/2000                 | 1             | 3,019.28            | 1,364.66        | 716.54             |                    |                    |          |
|                 |                          | 2             | 6,402.94            | 1,292.24        | 4,462.98           | 629.21             | 387.61             |          |
|                 |                          | 4             | 12,429.18           | 8,951.07        | 8,753.09           | 4,012.41           | 2,158.17           | 1,517.53 |
|                 |                          | 8             | 16,789.12           |                 | 15,260.29          | 7,384.79           | 4,104.80           | 1,739.28 |
|                 | 1000/1000                | 1             | 2,706.04            | 1,449.83        |                    |                    |                    |          |
|                 |                          | 2             | 4,693.24            | 960.39          | 3,958.45           | 736.68             | 425.70             |          |
|                 |                          | 4             | 8,557.11            | 7,278.64        | 6,817.41           | 3,866.05           | 1,876.40           | 1,188.91 |
|                 |                          | 8             | 13,483.04           |                 | 11,511.74          | 6,543.96           | 3,285.82           | 1,241.42 |
|                 | 2048/2048                | 1             | 1,911.20            | 798.50          | 412.37             |                    |                    |          |
|                 |                          | 2             | 3,408.82            | 767.24          | 2,551.21           | 388.82             | 226.60             |          |
|                 |                          | 4             | 6,702.46            | 5,354.80        | 5,212.02           | 2,512.22           | 1,316.92           | 891.95   |
|                 |                          | 8             | 10,348.65           |                 | 8,016.14           | 4,414.75           | 2,492.09           | 1,083.26 |
| Mixtral 8x7B    | 128/128                  | 2             | 25,135.25           | 8,512.51        | 24,572.90          | 15,395.59          | 5,927.88           |          |
|                 |                          | 4             | 42,394.61           | 40,148.01       | 40,309.25          | 27,747.43          | 11,205.51          | 6,784.44 |
|                 |                          | 8             | 54,648.80           |                 | 51,683.16          | 40,116.51          | 18,496.66          | 6,437.72 |
|                 | 128/2048                 | 2             | 29,412.17           | 3,271.02        | 20,938.80          | 7,391.51           | 4,278.79           |          |
|                 |                          | 4             | 52,603.13           | 43,071.34       | 40,580.94          | 21,332.15          | 10,946.58          | 7,475.05 |
|                 |                          | 8             | 70,427.00           |                 | 64,161.64          | 41,101.18          | 21,235.99          | 9,955.21 |
|                 | 128/4096                 | 2             | 21,312.11           | 2,254.56        |                    | 3,896.02           | 2,388.14           |          |
|                 |                          | 4             | 39,353.01           | 30,065.77       |                    |                    | 7,108.03           | 5,232.44 |
|                 |                          | 8             | 32,992.62           |                 | 47,860.65          | 27,261.67          | 15,943.70          | 8,081.21 |
|                 | 2048/128                 | 2             | 2,946.01            | 921.87          | 2,894.09           | 1,790.49           | 684.71             |          |
|                 |                          | 4             | 5,237.58            | 5,056.60        | 4,988.14           | 3,354.89           | 1,338.54           | 803.50   |
|                 |                          | 8             | 7,053.32            |                 | 6,559.63           | 5,072.46           | 2,244.39           | 753.39   |
|                 | 5000/500                 | 2             | 3,848.10            | 997.06          | 3,630.24           | 1,656.04           | 739.84             |          |
|                 |                          | 4             | 6,877.65            | 6,466.39        | 6,237.22           | 3,607.46           | 1,619.49           | 1,048.60 |
|                 |                          | 8             | 9,531.26            |                 | 8,709.34           | 6,237.96           | 2,927.13           | 1,109.25 |
|                 | 500/2000                 | 2             | 23,539.24           | 2,773.86        | 16,886.30          | 5,773.33           | 3,325.73           |          |
|                 |                          | 4             | 40,035.05           | 33,478.35       | 32,047.73          | 16,897.03          | 8,908.09           | 6,153.32 |
|                 |                          | 8             | 60,572.77           |                 | 41,597.80          | 31,392.32          | 16,954.54          | 7,980.34 |
|                 | 1000/1000                | 2             | 18,644.51           | 4,540.15        | 14,154.95          | 5,826.43           | 3,289.27           |          |
|                 |                          | 4             | 32,709.62           | 29,046.16       | 25,291.30          | 14,307.91          | 7,461.63           | 4,697.19 |
|                 |                          | 8             | 44,072.88           |                 | 40,628.46          | 27,633.48          | 13,741.62          | 5,706.17 |
|                 | 2048/2048                | 2             | 14,017.70           | 2,870.77        | 10,448.79          | 3,535.21           | 1,954.32           |          |
|                 |                          | 4             | 25,550.44           | 21,488.32       | 19,977.11          | 9,620.99           | 5,191.30           | 3,593.18 |
|                 |                          | 8             | 24,999.94           |                 | 31,678.85          | 19,372.52          | 10,572.07          | 4,860.61 |
|                 | 20000/2000               | 2             | 2,195.84            | 367.81          | 1,583.86           | 626.60             | 320.41             |          |
|                 |                          | 4             | 4,086.41            | 3,301.28        | 2,982.42           | 1,586.09           | 807.67             | 579.49   |
|                 |                          | 8             | 5,797.57            |                 | 5,163.91           | 3,106.98           | 1,653.55           | 821.64   |
*TP stands for Tensor Parallelism*

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [Benchmarking Suite README](../../../benchmarks/Suite.md).

### Commands

| Stage | Description | Command |
| :- | - | - |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `python benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file` |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --dataset $dataset_file` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir` |

### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
| `$osl` | Benchmark output sequence length. |
| `$tp_size` | Number of GPUs to run the benchmark with |
| `$engine_dir` | Location to store built engine file (can be deleted after running benchmarks). |
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$num_requests` | The number of requests to generate for dataset generation |
| `$seq_len` | A sequence length of ISL + OSL |

## Preparing a Dataset

In order to prepare a dataset, you can use the provided [script](../../../benchmarks/cpp/prepare_dataset.py).
To generate a synthetic dataset, run the following command:

```shell
python benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file
```

The command will generate a text file located at the path specified `$dataset_file` where all requests are of the same
input/output sequence length combinations. The script works by using the tokenizer to retrieve the vocabulary size and
randomly sample token IDs from it to create entirely random sequences. In the command above, all requests will be uniform
because the standard deviations for both input and output sequences are set to 0.


For each input and output sequence length combination, the table below details the `$num_requests` that were used. For
shorter input and output lengths, a larger number of messages were used to guarantee that the system hit a steady state
because requests enter and exit the system at a much faster rate. For longer input/output sequence lengths, requests
remain in the system longer and therefore require less requests to achieve steady state.


| Input Length | Output Length |  $seq_len  | $num_requests      |
| ------------ | ------------- | ---------- | ------------------ |
| 128          | 128           | 256        | 30000              |
| 128          | 2048          | 2176       | 3000               |
| 128          | 4096          | 4224       | 1500               |
| 2048         | 128           | 2176       | 3000               |
| 2048         | 2048          | 4096       | 1500               |
| 5000         | 500           | 5500       | 1500               |
| 1000         | 1000          | 2000       | 3000               |
| 500          | 2000          | 2500       | 3000               |
| 20000        | 2000          | 22000      | 1000               |

## Engine Building

All engines are built using the `trtllm-bench build` sub-command. The basic command for FP8 quantized engines is as follows:

```
trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --dataset $dataset_file
```

or if you would like to build for a specific sequence length:

```
trtllm-bench --model $model_name build --tp_size $tp_size --quantization FP8 --max_seq_length $seq_len
```

If you would like to build an FP16 engine without any quantization, simply remove the `--quantization FP8` option.

> [!NOTE] If you specify FP8 quantization, the KV cache will automatically be set to FP8 as well!

The `trtllm-bench build` sub-command will output the path where the engine is located upon a successful build. For example,

```shell
===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
===========================================================
```

## Running the Benchmark

To run the benchmark with the generated data set, simply use the `trtllm-bench throughput` sub-command. The benchmarker will
run an offline maximum throughput scenario such that all requests are queued in rapid succession. You simply need to provide
the patch to the engine from the [build](#engine-building) phase and a [generated dataset](#preparing-a-dataset).

```shell
trtllm-bench --model $model_name throughput --dataset $dataset_file --engine_dir $engine_dir
```

The results will be printed to the terminal upon benchmark completion. For example,

```shell
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       2048
Max Sequence Length:    4098

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   99.0%
Issue Rate (req/sec):   3.680275266452667e+18
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  23405.927228471104
Request Throughput (req/sec):   182.8588064724305
Total Latency (seconds):        16.406100739
===========================================================
```

> [!WARNING] In some cases, the benchmarker may not print anything at all. This behavior usually
means that the benchmark has hit an out of memory issue. Try reducing the KV cache percentage
using the `--kv_cache_free_gpu_mem_fraction` option to lower the percentage of used memory.
