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

**All data in the table below was generated using version 0.14.0 and presents token throughput in tokens/second.**

|                 |                          |               |                     |                    |                    |                    |                    |           |
| --------------- | ------------------------ | ------------- | ------------------- | ------------------ | ------------------ | ------------------ | ------------------ | --------- |
|                 |                          | **GPU**       | **H200 141GB HBM3** | **H100 80GB HBM3** | **H100 80GB HBM3** | **A100-SXM4-80GB** | **A100-PCIE-80GB** | **L40S**  |
|                 |                          | **Precision** | **FP8**             | **FP8**            | **FP16**           | **FP16**           | **FP16**           | **FP8**   |
| **Model**       | **Input/Output Lengths** | **TP Size**   |                     |                    |                    |                    |                    |           |
| LLaMA v3 70B    | 1000/1000                | 1             | 2594.2199           | 464.5243           |                    |                    |                    |           |
|                 |                          | 2             | 4574.1197           | 4092.3267          | 776.9965           | 468.5805           | 259.1155           |           |
|                 |                          | 4             | 7612.2487           | 6925.0844          | 3730.2064          | 1765.9123          | 987.1971           | 1159.357  |
|                 |                          | 8             | 13075.5194          | 10733.0804         | 5963.0914          | 3054.8915          | 960.3737           | 1173.3517 |
|                 | 128/128                  | 1             | 3904.1639           | 2551.6384          |                    |                    |                    |           |
|                 |                          | 2             | 5343.8677           | 5191.7428          | 3183.9714          | 1334.903           | 806.1477           |           |
|                 |                          | 4             | 8829.1049           | 8540.5362          | 5837.9598          | 2421.4383          | 1275.5474          | 1427.9115 |
|                 |                          | 8             | 16359.1322          | 15498.2004         | 10597.6556         | 4474.1621          | 1223.1747          | 1377.473  |
|                 | 128/2048                 | 1             | 3613.7474           | 418.3639           |                    |                    |                    |           |
|                 |                          | 2             | 7112.2959           | 5852.0185          | 817.52             | 511.6257           |                    |           |
|                 |                          | 4             | 12772.8148          | 8998.3742          | 5072.0345          | 2484.2018          | 1471.9105          | 1771.4437 |
|                 |                          | 8             | 19722.5974          | 15099.0633         | 7554.2141          | 4463.6602          | 1589.1759          | 1953.7918 |
|                 | 128/4096                 | 1             | 2409.6881           |                    |                    |                    |                    |           |
|                 |                          | 2             | 5687.3482           | 3513.0941          | 413.3767           | 273.5871           |                    |           |
|                 |                          | 4             | 8937.3115           | 6718.5895          | 3093.7358          | 1688.0132          | 1231.8104          | 1279.2496 |
|                 |                          | 8             | 13976.1386          | 9279.1013          | 5001.2743          | 2948.5374          | 1350.794           | 1494.0776 |
|                 | 2048/128                 | 1             | 457.5772            | 241.7561           |                    |                    |                    |           |
|                 |                          | 2             | 699.5582            | 690.9961           | 328.0399           | 145.088            | 91.1746            |           |
|                 |                          | 4             | 1035.6523           | 1008.8318          | 670.6725           | 278.5717           | 150.2619           | 168.7886  |
|                 |                          | 8             | 2055.7245           | 1996.2653          | 1288.7599          | 546.9599           | 140.0144           | 160.2741  |
|                 | 2048/2048                | 1             | 1802.1116           | 204.0931           |                    |                    |                    |           |
|                 |                          | 2             | 3487.2497           | 2444.6903          | 165.6522           | 126.1101           |                    |           |
|                 |                          | 4             | 6126.7196           | 4850.8285          | 2386.6556          | 1230.1833          | 822.2269           | 876.6085  |
|                 |                          | 8             | 9784.0193           | 7432.6659          | 3991.2123          | 2144.3042          | 883.4809           | 994.94    |
|                 | 500/2000                 | 1             | 2822.7846           | 389.8823           |                    |                    |                    |           |
|                 |                          | 2             | 6175.7623           | 4601.857           | 687.5386           | 430.6093           |                    |           |
|                 |                          | 4             | 10783.8925          | 9018.9053          | 3698.3674          | 2113.3936          | 1248.8319          | 1468.7827 |
|                 |                          | 8             | 17631.9756          | 11375.9582         | 6321.3679          | 3673.5693          | 1321.8541          | 1636.4588 |
|                 | 5000/500                 | 1             | 532.2603            | 123.8543           |                    |                    |                    |           |
|                 |                          | 2             | 931.8255            | 897.4263           | 227.9005           | 117.5698           | 75.35              |           |
|                 |                          | 4             | 1399.7865           | 1316.2865          | 831.2804           | 362.3465           | 209.8052           | 234.7343  |
|                 |                          | 8             | 2725.1283           | 2469.5585          | 1446.3508          | 662.5725           | 202.0719           | 231.9027  |
| LLaMA v3.1 405B | 1000/1000                | 8             | 3391.0372           |                    |                    |                    |                    |           |
|                 | 128/128                  | 8             | 3766.2785           |                    |                    |                    |                    |           |
|                 | 128/2048                 | 8             | 5952.1416           |                    |                    |                    |                    |           |
|                 | 128/4096                 | 8             | 3944.117            |                    |                    |                    |                    |           |
|                 | 20000/2000               | 8             | 481.5732            |                    |                    |                    |                    |           |
|                 | 2048/128                 | 8             | 444.5735            |                    |                    |                    |                    |           |
|                 | 2048/2048                | 8             | 2604.8557           |                    |                    |                    |                    |           |
|                 | 500/2000                 | 8             | 4805.86             |                    |                    |                    |                    |           |
|                 | 5000/500                 | 8             | 655.9754            |                    |                    |                    |                    |           |
| LLaMA v3.1 70B  | 1000/1000                | 1             | 2585.0953           | 410.286            |                    |                    |                    |           |
|                 |                          | 2             | 4600.9616           | 4116.4444          | 785.4931           | 468.6383           | 257.972            |           |
|                 |                          | 4             | 7607.5304           | 6932.8808          | 3774.676           | 1762.6831          | 989.4082           | 1161.4814 |
|                 |                          | 8             | 13081.434           | 10730.156          | 5978.4573          | 3190.0211          | 959.8463           | 1188.1193 |
|                 | 128/128                  | 1             | 3897.2623           | 2459.6003          |                    |                    |                    |           |
|                 |                          | 2             | 5357.0227           | 5194.8171          | 3207.2866          | 1346.9692          | 806.7215           |           |
|                 |                          | 4             | 8826.9618           | 8542.3012          | 5846.8413          | 2420.8665          | 1272.6755          | 1438.0446 |
|                 |                          | 8             | 16382.9807          | 15533.1169         | 10649.4968         | 4572.3445          | 1212.0566          | 1381.7051 |
|                 | 128/2048                 | 1             | 3612.2603           | 445.7773           |                    |                    |                    |           |
|                 |                          | 2             | 7054.7235           | 5869.3998          | 822.1912           | 483.1299           |                    |           |
|                 |                          | 4             | 12763.4114          | 9017.4377          | 4982.6225          | 2492.4036          | 1435.236           | 1763.522  |
|                 |                          | 8             | 19266.0398          | 15190.1652         | 7605.5295          | 4254.2871          | 1609.2473          | 1944.1251 |
|                 | 128/4096                 | 1             | 2415.1981           |                    |                    |                    |                    |           |
|                 |                          | 2             | 5671.9561           | 3518.782           | 419.0178           | 272.9137           |                    |           |
|                 |                          | 4             | 8939.8227           | 6431.2702          | 3083.8794          | 1685.9677          | 1212.5416          | 1280.3778 |
|                 |                          | 8             | 13974.2854          | 9168.709           | 4981.9765          | 3067.5452          | 1310.091           | 1499.2441 |
|                 | 20000/2000               | 1             | 240.7202            |                    |                    |                    |                    |           |
|                 |                          | 2             | 614.318             | 397.6801           |                    |                    |                    |           |
|                 |                          | 4             | 1030.9528           | 851.8542           | 369.4269           | 179.5181           | 126.7676           | 140.5565  |
|                 |                          | 8             | 1898.9762           | 1354.5333          |                    | 362.9368           | 156.5767           | 141.1584  |
|                 | 2048/128                 | 1             | 458.1948            | 244.1842           |                    |                    |                    |           |
|                 |                          | 2             | 692.3911            | 697.3907           | 322.7016           | 144.7921           | 95.0306            |           |
|                 |                          | 4             | 1034.5773           | 1001.0771          | 688.0344           | 278.4018           | 150.6795           | 169.0386  |
|                 |                          | 8             | 2070.8157           | 1966.6072          | 1316.3086          | 550.4751           | 142.6166           | 163.6749  |
|                 | 2048/2048                | 1             | 1797.6743           | 209.1707           |                    |                    |                    |           |
|                 |                          | 2             | 3518.0774           | 2445.0093          | 166.792            | 126.1127           |                    |           |
|                 |                          | 4             | 6112.9026           | 4838.5272          | 2393.1359          | 1231.0359          | 823.4777           | 876.2254  |
|                 |                          | 8             | 9716.1934           | 7434.8117          | 4023.6978          | 2171.5323          | 858.6602           | 1001.3649 |
|                 | 500/2000                 | 1             | 2826.6665           |                    |                    |                    |                    |           |
|                 |                          | 2             | 6106.5855           | 4605.9226          | 700.5415           | 430.6129           |                    |           |
|                 |                          | 4             | 10816.8283          | 9205.3766          | 3781.082           | 2096.2441          | 1176.418           | 1470.0826 |
|                 |                          | 8             | 17693.705           | 13109.4437         | 6205.2658          | 3486.7891          | 1306.35            | 1639.2778 |
|                 | 5000/500                 | 1             | 533.6128            | 125.4236           |                    |                    |                    |           |
|                 |                          | 2             | 936.7014            | 886.6758           | 228.874            | 116.9529           | 76.1601            |           |
|                 |                          | 4             | 1386.4827           | 1313.893           | 849.1091           | 362.9361           | 209.2045           | 236.117   |
|                 |                          | 8             | 2711.5057           | 2444.9643          | 1420.5163          | 670.3742           | 203.8008           | 230.3084  |
| LLaMA v3.1 8B   | 1000/1000                | 1             | 16414.6988          | 14108.0361         | 7054.5156          | 3634.3886          | 3165.3542          | 3726.7552 |
|                 | 128/128                  | 1             | 27778.8885          | 26933.1886         | 15571.6549         | 6701.7958          | 5338.0166          | 8639.7933 |
|                 | 128/2048                 | 1             | 22948.5383          | 18995.2523         | 9150.7477          | 4963.4443          | 4250.6391          | 5101.6652 |
|                 | 128/4096                 | 1             | 15583.3035          | 11815.449          | 5368.9227          | 3011.3335          | 2568.5398          | 2774.5363 |
|                 | 20000/2000               | 1             | 1649.5453           | 1301.4754          | 562.8735           | 316.533            | 291.4776           | 270.5404  |
|                 | 2048/128                 | 1             | 3619.4309           | 3460.3545          | 1904.3259          | 795.389            | 611.8446           | 986.9134  |
|                 | 2048/2048                | 1             | 11032.9729          | 8777.6623          | 4159.6857          | 2264.9513          | 2011.1215          | 2018.303  |
|                 | 500/2000                 | 1             | 19510.4015          | 14993.328          | 7498.3331          | 3945.1912          | 3374.7133          | 4065.3921 |
|                 | 5000/500                 | 1             | 3787.6721           | 3258.2001          | 1708.0353          | 790.6631           | 703.56             | 855.9822  |
| Mistral 7B      | 1000/1000                | 1             | 17739.1436          | 14986.7562         | 7697.1418          | 3804.5585          | 3333.4754          | 3981.4799 |
|                 | 128/128                  | 1             | 30094.9137          | 29341.284          | 16238.937          | 6914.2184          | 5491.7418          | 9127.5052 |
|                 | 128/2048                 | 1             | 24671.5477          | 20941.6631         | 9708.1161          | 5303.4318          | 4402.3044          | 5357.3405 |
|                 | 128/4096                 | 1             | 16454.0833          | 12780.3724         | 5800.4957          | 3235.0678          | 2825.7896          | 2879.9833 |
|                 | 20000/2000               | 1             | 1676.0415           | 1317.9654          | 569.7589           | 324.5936           | 281.4751           | 286.353   |
|                 | 2048/128                 | 1             | 3649.1462           | 3492.3042          | 1929.3126          | 800.9286           | 617.0932           | 1019.75   |
|                 | 2048/2048                | 1             | 11403.6968          | 8974.7383          | 4367.8733          | 2331.8112          | 1988.3496          | 2184.3861 |
|                 | 500/2000                 | 1             | 20819.4592          | 15992.3357         | 7947.4257          | 4189.395           | 3603.4489          | 4286.3867 |
|                 | 5000/500                 | 1             | 3840.0108           | 3340.7385          | 1707.2611          | 807.4561           | 722.8385           | 881.7336  |
| Mixtral 8x22B   | 1000/1000                | 8             | 18557.43            | 16918.03           | 9759.888           | 4753.6273          |                    | 2128.4403 |
|                 | 128/128                  | 8             | 25179.4765          | 23729.5293         | 16421.3182         | 6948.5923          |                    | 2488.6297 |
|                 | 128/2048                 | 8             | 27492.4926          | 24556.7807         | 12303.4168         | 7246.7172          |                    | 3540.0067 |
|                 | 128/4096                 | 8             | 19718.8648          | 17755.0018         | 7474.3817          | 4696.6123          |                    | 2568.3114 |
|                 | 20000/2000               | 8             | 2897.182            | 2189.606           | 1118.8294          | 594.8509           |                    | 309.0799  |
|                 | 2048/128                 | 8             | 3093.8418           | 2917.1362          | 1994.0127          | 825.3934           |                    | 294.7706  |
|                 | 2048/2048                | 8             | 13795.9827          | 12487.6502         | 5857.8831          | 3377.8371          |                    | 1694.6176 |
|                 | 500/2000                 | 8             | 24637.473           | 19997.3914         | 10637.6598         | 6007.619           |                    | 2976.9633 |
|                 | 5000/500                 | 8             | 3889.2745           | 3578.4843          | 2211.2377          | 1028.3843          |                    | 420.2156  |
| Mixtral 8x7B    | 1000/1000                | 2             | 18712.2046          | 15931.8663         | 6052.876           | 3276.6186          | 1907.8817          |           |
|                 |                          | 4             | 32834.0923          | 28015.1981         | 15509.1538         | 7357.1613          | 4737.0179          | 5060.8399 |
|                 |                          | 8             | 44410.7533          | 40573.0499         | 27684.9381         | 13948.1533         | 4970.9287          | 5725.9638 |
|                 | 128/128                  | 2             | 24970.5594          | 24321.9927         | 15334.2103         | 5915.3897          | 3810.1846          |           |
|                 |                          | 4             | 42500.5855          | 40182.7271         | 27718.9857         | 11328.7486         | 6026.9206          | 6769.9441 |
|                 |                          | 8             | 54304.0436          | 51030.9048         | 40119.3268         | 17918.1146         | 5573.7682          | 6422.4308 |
|                 | 128/2048                 | 2             | 29314.1475          | 20945.7816         | 7409.9253          | 4284.3035          | 2248.1815          |           |
|                 |                          | 4             | 52680.8353          | 40668.5928         | 21293.1761         | 10929.0182         | 7353.7405          | 7506.7612 |
|                 |                          | 8             | 70409.1968          | 64529.9982         | 40839.3077         | 21058.2144         | 8866.251           | 9907.6896 |
|                 | 128/4096                 | 2             | 21520.4385          | 12070.6724         | 3928.6678          | 2302.964           | 1171.966           |           |
|                 |                          | 4             | 32550.5267          | 29120.2002         | 11678.0071         | 6538.1511          | 5176.9632          | 4958.7004 |
|                 |                          | 8             | 40373.4857          | 36357.7861         | 21628.821          | 13565.7778         | 7209.2336          | 8271.7938 |
|                 | 20000/2000               | 2             | 2204.1378           | 1659.5907          | 622.2717           | 321.9839           | 185.6671           |           |
|                 |                          | 4             | 4047.7473           | 3290.9457          | 1602.0208          | 778.7285           | 572.4282           | 587.1759  |
|                 |                          | 8             | 6561.6849           | 5328.5261          | 3113.2047          | 1645.8114          | 750.5372           | 828.8471  |
|                 | 2048/128                 | 2             | 2958.0873           | 2883.5166          | 1796.5451          | 687.7251           | 465.1585           |           |
|                 |                          | 4             | 5229.8744           | 4972.6818          | 3354.994           | 1351.7191          | 728.4943           | 812.0143  |
|                 |                          | 8             | 7030.9766           | 6532.721           | 5025.3047          | 2248.6418          | 677.9886           | 771.3656  |
|                 | 2048/2048                | 2             | 13842.834           | 9334.0732          | 3503.0218          | 1997.1923          | 1060.8946          |           |
|                 |                          | 4             | 22389.4914          | 20185.8212         | 9143.2741          | 4963.8758          | 3520.3659          | 3453.8076 |
|                 |                          | 8             | 28975.322           | 26176.9163         | 19291.8278         | 10552.9732         | 4590.187           | 4929.7228 |
|                 | 500/2000                 | 2             | 23459.0411          | 18185.6392         | 6023.3308          | 3438.6964          | 1817.11            |           |
|                 |                          | 4             | 39971.0236          | 31693.8787         | 17087.037          | 8930.3495          | 6117.5624          | 6434.9178 |
|                 |                          | 8             | 60721.462           | 48842.8084         | 31358.2791         | 17034.706          | 7118.0767          | 8130.8026 |
|                 | 5000/500                 | 2             | 3742.5293           | 3563.8228          | 1648.9041          | 733.1921           | 448.6716           |           |
|                 |                          | 4             | 6602.3877           | 6020.6267          | 3543.6819          | 1603.8223          | 948.0567           | 1047.3212 |
|                 |                          | 8             | 8862.8164           | 8214.9445          | 5968.7734          | 2813.1531          | 969.817            | 1098.3081 |

*TP stands for Tensor Parallelism*

## Reproducing Benchmarked Results

> [!NOTE] The only models supported in this workflow are those listed in the table above.

The following tables are references for commands that are used as part of the benchmarking process. For a more detailed
description of this benchmarking workflow, see the [benchmarking suite documentation](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html).

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

### Preparing a Dataset

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

### Engine Building

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

### Running the Benchmark

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

## Online Serving Measurements

The [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend) is used to measure the performance of TensorRT-LLM for online serving.

The below table shows the throughput and latency under a serving scenario.

**All data in the table below was generated using version 0.14.0, with 500 requests and BF16 precision.**

|                 |                    |         |         |         |         |                  |                    |                    |                               |                         |
| --------------- | -------------------| --------| --------| --------| --------|------------------| ------------------ | ------------------ | ----------------------------- |------------------------ |
| **Model**       | **GPU**            | **TP**  | **Input Length** | **Output Length** | **QPS** | **Tput(req/s)**  | **Mean TTFT(ms)**  | **Mean ITL(ms)**   | **Total Token Tput (tok/s)**  | **Output Tput (tok/s)** |
|LLaMA 3.1 70B|H100 80GB HBM3|4|467|256|2|2|62|21|1406|498||
||||||4|4|68|24|2750|973|
||||||8|7|92|32|5256|1860|
||||||16|12|175|66|8941|3164|
||||||32|16|1229|86|11537|4083|
||||||INF|16|9123|85|11593|4103|
||||467|16|2|2|53|18|844|28|
||||||4|4|58|20|1908|63|
||||||8|8|71|24|3795|126|
||||||16|16|109|38|7492|248|
||||||32|28|1197|482|13655|452|
||||||INF|28|9126|548|13719|454|
||||202|214|2|2|48|20|780|401|
||||||4|4|51|22|1499|771|
||||||8|7|57|25|2702|1390|
||||||16|11|74|32|4364|2245|
||||||32|14|116|42|5837|3003|
||||||INF|16|4482|50|6725|3459|
|LLaMA 3.1 8B||1|467|256|2|2|23|8|1423|504|
||||||4|4|24|9|2624|929|
||||||8|8|26|9|5535|1959|
||||||16|15|30|11|10636|3765|
||||||32|26|50|19|19138|6774|
||||||INF|37|3335|39|26614|9420|
||||467|16|2|2|19|7|956|32|
||||||4|4|20|7|1910|63|
||||||8|8|22|7|3808|126|
||||||16|16|24|8|7567|251|
||||||32|31|29|10|14894|493|
||||||INF|79|3280|193|38319|1269|
||||202|214|2|2|19|7|809|416|
||||||4|4|20|8|1586|816|
||||||8|7|21|9|3047|1568|
||||||16|13|23|10|5597|2879|
||||||32|23|27|11|9381|4825|
||||||INF|39|1657|21|16117|8291|
|LLaMA 3.1 70B|H200 131GB HBM3|4|467|256|2|2|58|18|1411|499|
||||||4|4|63|20|2770|980|
||||||8|7|84|27|5328|1886|
||||||16|13|165|60|9224|3264|
||||||32|16|1279|83|11800|4176|
||||||INF|16|9222|83|11826|4185|
||||467|16|2|2|50|15|956|32|
||||||4|4|55|16|1909|63|
||||||8|8|67|20|3799|126|
||||||16|16|103|33|7499|248|
||||||32|28|1259|485|13586|450|
||||||INF|29|9074|546|13792|457|
||||202|214|2|2|43|17|793|408|
||||||4|4|46|18|1524|784|
||||||8|7|51|21|2796|1438|
||||||16|11|67|28|4639|2386|
||||||32|15|112|39|6288|3235|
||||||INF|17|4480|48|7230|3719|
|LLaMA 3.1 8B|H200 131GB HBM3|1|467|256|2|2|21|6|1425|504|
||||||4|4|23|7|2828|1001|
||||||8|8|24|7|5567|1971|
||||||16|15|27|9|10761|3809|
||||||32|27|44|16|19848|7025|
||||||INF|40|3237|36|28596|10121|
||||467|16|2|2|18|5|956|32|
||||||4|4|19|6|1910|63|
||||||8|8|20|6|3810|126|
||||||16|16|22|7|7567|250|
||||||32|31|27|9|14927|494|
||||||INF|81|3227|190|39007|1291|
||||202|214|2|2|17|6|812|418|
||||||4|4|18|6|1597|822|
||||||8|7|19|7|3088|1589|
||||||16|14|20|8|5771|2969|
||||||32|24|24|9|9931|5109|
||||||INF|43|1665|19|17861|9189|

*TP stands for Tensor Parallelism*

*TTFT stands for Time To First Token*

*ITL stands for Inter Token Latency*
