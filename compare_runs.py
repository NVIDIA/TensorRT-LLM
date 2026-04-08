#!/usr/bin/env python3
"""Compare two KV perf benchmark runs side by side."""

import csv
import sys

# Run 1 data: (config, concurrency) -> {py_ttft, py_tpot, py_itl, py_e2el, py_tput, cpp_ttft, cpp_tpot, cpp_itl, cpp_e2el, cpp_tput, gap_ttft, gap_tpot, gap_itl, gap_e2el}
run1_raw = """deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_dep16_bs1_eplb0_mtp3-Default	1	16219.19	4.76	365.62	50067.70	142.04	16192.42	4.81	370.33	50415.32	141.06	0.17	-1.04	-1.27	-0.69
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_dep8_bs4_eplb0_mtp2-Default	4	27748.55	6.40	350.67	74805.21	362.21	27409.15	6.77	360.14	77199.19	358.55	1.24	-5.47	-2.63	-3.10
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_tep8_bs1_eplb0_mtp0-Default	1	16257.29	6.89	137.75	65236.21	109.01	16130.56	6.86	137.22	64927.05	109.53	0.79	0.44	0.39	0.48
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_tep8_bs2_eplb0_mtp3-Default	2	20593.69	4.65	254.24	55758.32	263.41	20325.26	5.29	251.11	60462.96	235.71	1.32	-12.10	1.25	-7.78
deepseek-r1-fp4_128k8k_ctx1_pp8_gen11_tep4_bs2_eplb0_mtp0-Default	22	68417.61	8.50	180.16	131511.55	1099.76	83040.12	9.05	182.71	150014.45	1040.85	-17.61	-6.08	-1.40	-12.33
deepseek-r1-fp4_128k8k_ctx1_pp8_gen14_tep4_bs1_eplb0_mtp0-Default	14	57312.30	7.56	151.76	110275.53	847.58	57356.59	7.62	152.33	112466.66	846.85	-0.08	-0.79	-0.37	-1.95
deepseek-r1-fp4_128k8k_ctx1_pp8_gen5_tep8_bs2_eplb0_mtp3-Default	10	46225.28	4.23	254.88	75597.49	877.20	46320.98	4.46	258.64	79972.50	853.72	-0.21	-5.16	-1.45	-5.47
deepseek-r1-fp4_128k8k_ctx1_pp8_gen7_tep4_bs2_eplb0_mtp2-Default	14	55900.08	5.47	292.52	92710.93	1025.94	55963.49	5.48	292.57	98555.04	967.85	-0.11	-0.18	-0.02	-5.93
deepseek-r1-fp4_128k8k_ctx1_pp8_gen7_tep8_bs1_eplb0_mtp0-Default	7	36215.37	6.97	139.64	86317.00	556.79	35694.55	7.02	140.26	86145.87	560.75	1.46	-0.71	-0.44	0.20
deepseek-r1-fp4_128k8k_ctx1_pp8_gen8_tep4_bs4_eplb0_mtp0-Default	32	82918.90	10.43	226.52	164815.63	1220.69	105277.51	11.18	236.01	183385.02	1164.94	-21.24	-6.71	-4.02	-10.13
deepseek-r1-fp4_128k8k_ctx2_pp8_gen1_dep16_bs8_eplb0_mtp0-Default	8	30264.98	13.66	273.30	128668.33	422.21	30289.26	13.63	272.68	128512.98	422.68	-0.08	0.22	0.23	0.12
deepseek-r1-fp4_128k8k_ctx2_pp8_gen1_dep32_bs2_eplb0_mtp0-Default	2	19934.12	13.31	265.70	120765.25	124.59	18958.72	13.28	265.92	119607.39	123.58	5.14	0.23	-0.08	0.97
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep16_bs16_eplb0_mtp0-Default	16	37523.73	14.02	280.28	139622.29	768.40	36313.37	14.03	280.78	138476.67	774.02	3.33	-0.07	-0.18	0.83
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep16_bs8_eplb0_mtp2-Default	8	27276.83	7.27	355.53	80558.47	588.70	28863.21	7.08	351.86	79284.18	596.56	-5.50	2.68	1.04	1.61
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep32_bs4_eplb0_mtp0-Default	4	23198.32	13.67	273.31	125239.50	229.72	22392.17	13.60	272.13	124186.87	232.12	3.60	0.51	0.43	0.85
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	4	139.63	5.55	111.03	5409.29	671.01	133.64	5.54	110.91	5377.41	671.13	4.48	0.18	0.11	0.59
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	8	147.61	6.15	123.15	5847.74	1203.73	141.86	6.06	121.42	5812.87	1228.47	4.05	1.49	1.42	0.60
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	16	176.79	7.27	145.28	6876.34	2085.03	170.60	7.14	142.43	6736.67	2131.33	3.63	1.82	2.00	2.07
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	32	196.86	8.38	167.33	7857.57	3579.64	196.34	8.20	163.29	7681.56	3663.88	0.26	2.20	2.47	2.29
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	64	218.04	9.20	183.68	8757.79	6523.62	208.30	9.27	185.13	8837.94	6497.68	4.68	-0.76	-0.78	-0.91
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	128	243.15	10.47	209.04	9945.10	11352.36	235.36	10.48	209.10	9928.56	11365.23	3.31	-0.10	-0.03	0.17
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	4	160.75	3.37	157.15	3356.69	1061.52	151.34	3.37	156.60	3221.27	1091.34	6.22	0.00	0.35	4.20
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	8	165.62	3.74	176.36	3642.02	1930.78	159.76	3.68	175.47	3529.77	1973.80	3.67	1.63	0.51	3.18
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	16	186.75	4.19	198.27	4019.72	3504.59	195.99	4.27	198.37	4048.19	3472.19	-4.71	-1.87	-0.05	-0.70
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	32	227.43	4.89	225.24	4675.22	5991.56	224.23	4.84	224.90	4610.67	6158.82	1.43	1.03	0.15	1.40
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	64	251.07	5.75	265.62	5523.13	10421.29	227.33	5.73	266.23	5563.54	10438.25	10.44	0.35	-0.23	-0.73
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	128	273.53	6.49	307.14	6231.95	18154.96	261.99	6.53	307.52	6272.18	18168.21	4.40	-0.61	-0.12	-0.64
deepseek-r1-fp4_1k1k_ctx2_gen1_dep16_bs128_eplb0_mtp3_ccb-NIXL	2048	328.34	22.59	1072.94	21296.28	84073.11	359.74	22.51	1069.92	21230.64	84507.64	-8.73	0.36	0.28	0.31
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	3	369.87	3.27	164.89	3370.34	845.86	364.82	3.46	164.41	3695.82	765.66	1.38	-5.49	0.29	-8.81
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	6	436.20	3.88	188.27	4072.08	1320.48	369.36	3.87	187.96	3947.82	1388.49	18.10	0.26	0.16	3.15
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	12	476.99	4.34	209.12	4383.59	2415.25	503.97	4.11	207.82	4130.11	2644.13	-5.35	5.60	0.63	6.14
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	24	641.28	5.11	243.94	5290.26	4025.59	631.47	5.05	244.66	5232.19	4074.81	1.55	1.19	-0.29	1.11
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	48	855.60	6.22	300.29	6625.10	6410.03	781.33	6.10	300.84	6467.20	6598.59	9.51	1.97	-0.18	2.44
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	3	358.66	5.69	113.54	5856.72	485.71	329.03	5.71	114.25	5831.88	486.54	9.01	-0.35	-0.62	0.43
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	6	402.84	6.35	128.11	6333.46	855.07	356.38	6.45	129.76	6388.45	850.14	13.04	-1.55	-1.27	-0.86
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	12	474.46	7.46	149.51	7280.92	1436.74	410.55	7.63	152.69	7504.93	1409.13	15.57	-2.23	-2.08	-2.98
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	24	640.08	8.79	175.68	8700.71	2441.35	572.11	9.00	179.89	8861.75	2399.68	11.88	-2.33	-2.34	-1.82
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	48	709.28	9.93	198.55	9972.71	4204.55	626.91	9.89	197.57	9913.79	4262.35	13.14	0.40	0.50	0.59
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	96	956.26	11.65	232.66	11730.54	6907.61	825.51	11.70	233.67	11654.06	6959.94	15.84	-0.43	-0.43	0.66
deepseek-r1-fp4_8k1k_ctx6_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	1024	788.38	21.13	421.87	20436.36	41599.78	801.86	21.08	421.04	20415.70	41691.29	-1.68	0.24	0.20	0.10
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	512	4457.81	14.86	296.92	18207.62	23604.22	2203.45	15.00	299.63	16130.10	26488.23	102.31	-0.93	-0.90	12.88
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	1024	22631.28	14.86	296.85	36302.34	24730.00	18155.77	15.00	299.15	32698.50	25092.28	24.65	-0.93	-0.77	11.02
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	4	133.24	10.13	202.60	9730.45	373.00	135.63	10.22	204.43	9805.16	369.90	-1.76	-0.88	-0.90	-0.76
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	8	143.53	10.72	213.89	10037.55	695.03	145.03	10.75	214.17	10106.91	693.04	-1.03	-0.28	-0.13	-0.69
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	16	151.56	10.99	219.04	10298.78	1391.62	150.22	11.22	223.98	10516.77	1367.89	0.89	-2.05	-2.21	-2.07
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	32	178.41	11.68	232.49	10866.36	2577.02	172.85	12.08	241.24	11266.68	2496.67	3.22	-3.31	-3.63	-3.55
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	64	214.76	12.17	244.24	11499.57	4925.35	212.96	12.63	254.02	11935.75	4762.27	0.85	-3.64	-3.85	-3.65
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	128	237.68	10.42	207.83	9900.22	11180.15	230.65	10.99	218.53	10404.21	10647.45	3.05	-5.19	-4.90	-4.84
Qwen3-235B-A22B-FP4_8k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	256	26763.45	13.51	269.74	39129.40	5494.55	21593.16	13.54	270.37	33991.09	6292.86	23.94	-0.22	-0.23	15.12
Qwen3-235B-A22B-FP4_8k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	512	68966.60	13.50	269.59	81220.45	5572.53	55566.46	13.54	270.51	67883.66	6576.83	24.12	-0.30	-0.34	19.65"""

run2_raw = """deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_dep16_bs1_eplb0_mtp3-Default	1	16487.36	4.78	365.64	50491.54	2271.6	16161.01	4.78	365.55	50157.27	2286.74
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_dep8_bs4_eplb0_mtp2-Default	4	27673.58	6.79	360.49	79043.86	5805.64	27579.21	6.67	353.89	78009.21	5879.07
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_tep8_bs1_eplb0_mtp0-Default	1	16326.96	6.88	137.34	65220.12	1758.63	16293.05	6.85	136.84	65008.17	1764.36
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_tep8_bs1_eplb0_mtp3-Default	1	16314.51	3.12	205.21	38476.88	2980.88	16135.5	2.92	207.75	36910.05	3107.4
deepseek-r1-fp4_128k8k_ctx1_pp8_gen1_tep8_bs2_eplb0_mtp3-Default	2	20444.67	4.67	253.43	55797.69	4191.56	20428.66	5.12	253.67	59240.85	3956.57
deepseek-r1-fp4_128k8k_ctx1_pp8_gen11_tep4_bs2_eplb0_mtp0-Default	22	63576.29	8.46	168.94	125554.82	19428.29	78608.42	8.98	179.16	144337.49	18453.88
deepseek-r1-fp4_128k8k_ctx1_pp8_gen14_tep4_bs1_eplb0_mtp0-Default	14	51619.01	7.63	152.35	107084.9	14972.7	55664.91	7.62	152.2	111075.42	15030.41
deepseek-r1-fp4_128k8k_ctx1_pp8_gen5_tep8_bs2_eplb0_mtp3-Default	10	44080.68	4.07	249.48	74143.03	15659.29	44893.26	4.22	251.04	76273.03	15013.43
deepseek-r1-fp4_128k8k_ctx1_pp8_gen7_tep4_bs2_eplb0_mtp2-Default	14	51560.94	5.44	277.93	91066.45	17640.57	55560.23	5.54	287.67	95834.16	17423.94
deepseek-r1-fp4_128k8k_ctx1_pp8_gen7_tep8_bs1_eplb0_mtp0-Default	7	36212.93	6.98	139.43	87741.11	9424.54	35887.43	6.97	139.22	87339.07	9473.86
deepseek-r1-fp4_128k8k_ctx1_pp8_gen8_tep4_bs4_eplb0_mtp0-Default	32	78346.66	10.43	208.05	154728.34	20993.9	94408.67	11.45	228.52	178305.44	19918.53
deepseek-r1-fp4_128k8k_ctx2_pp8_gen1_dep16_bs8_eplb0_mtp0-Default	8	31114.46	13.66	272.87	131734.83	7170.06	30700.35	13.71	273.72	131636.43	7174.13
deepseek-r1-fp4_128k8k_ctx2_pp8_gen1_dep32_bs2_eplb0_mtp0-Default	2	19203.15	13.16	262.72	118905.6	1951.44	19552.85	13.34	266.4	120652.37	1957.62
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep16_bs16_eplb0_mtp0-Default	16	35060.45	14.17	282.92	137938.67	13555.91	35203.97	14.17	282.98	138103.76	13533.91
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep16_bs8_eplb0_mtp2-Default	8	27460.5	7.11	357.05	79992.19	10248.78	28371.98	7.11	356.51	80823.97	10177.81
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep32_bs2_eplb0_mtp3-Default	2	21933.5	7.82	381.66	81280.96	2817.71	21587.2	6.74	377.74	72582.48	3090.33
deepseek-r1-fp4_128k8k_ctx3_pp8_gen1_dep32_bs4_eplb0_mtp0-Default	4	22555.01	13.68	273.17	125881.48	3723.05	22846.68	13.62	272.13	125780.41	3803.4
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	4	151.45	5.58	110.15	5335.48	1325.97	158.17	5.58	110.11	5340.07	1326.54
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	8	172.17	6.14	121.39	5864	2410.15	184.62	6.15	121.58	5885.57	2416.38
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	16	202.7	7.21	142.54	6816.48	4227.24	210.41	7.25	143.3	6859.23	4198.52
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	32	227.76	8.33	164.82	7865.52	7212.75	229.95	8.35	165.14	7882.6	7210.4
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	64	277	9.27	183.42	8847.06	12871.83	277.86	9.2	181.89	8776.18	13001.64
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	128	362.39	10.44	206.41	9966.22	22753.91	548.26	10.05	198.71	9793.94	22930.17
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	4	227	3.12	156.38	3115.18	2246.52	206.12	3.4	154.17	3357.04	2022.7
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	8	271.68	3.68	172.33	3686	3856.31	190.52	3.65	172.82	3576.74	3889.08
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	16	298.53	4.1	193.72	4056.34	6840.04	214.42	4.06	195.43	3938.23	7111.37
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	32	333.2	4.62	220.3	4567.06	12037.83	268.1	4.64	221.84	4522.11	12244.4
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	64	333.85	5.45	262.23	5372.55	20730.13	294.71	5.55	264.02	5423.45	20517.58
deepseek-r1-fp4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp3_ccb-NIXL	128	428.06	6.2	297.9	6130.93	36153.5	395.01	6.26	299.48	6151.18	36046.85
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	3	405.42	3.24	164.07	3454.42	6834.82	404.78	3.1	162.89	3323.23	7248.51
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	6	534.73	3.74	181.93	3983.84	11993.31	448.29	3.66	180.87	3816.92	12058.65
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	12	596.15	3.93	201.92	4201.18	22724.54	557.62	3.97	202.67	4197.26	22022.81
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	24	757.78	4.87	238.21	5225.41	35993.97	725.32	4.77	238.86	5103.23	36811
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs16_eplb0_mtp3_ccb-NIXL	48	1097.52	5.81	289.3	6478.95	58017.97	1052.91	5.78	290.62	6402.89	59229.86
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	3	398.73	5.73	113.14	5768.07	4254.73	379.66	5.73	113.15	5749.44	4276.92
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	6	477.02	6.45	127.49	6429.43	7531.53	461.9	6.47	127.9	6433.34	7543.55
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	12	567.85	7.65	151.15	7561.6	12625.4	538.77	7.67	151.66	7556.22	12645.78
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	24	741.15	8.97	177.37	8986.01	21467.76	721.78	9.01	178.04	8998.04	21487.91
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	48	983.19	9.87	195.19	10124.41	37795.9	944.73	9.86	195.08	10081.06	37938.98
deepseek-r1-fp4_8k1k_ctx1_gen3_tep8_bs32_eplb0_mtp0_ccb-NIXL	96	1530.86	11.58	229.03	12174.11	62494.5	1486.68	11.59	229.24	12139.58	62697.6
deepseek-r1-fp4_8k1k_ctx6_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	1024	2093.24	21.02	415.57	21453.18	374161.12	2074.11	21.06	416.46	21475.15	373750.75
deepseek-r1-fp4_8k1k_ctx8_gen1_dep32_bs16_eplb0_mtp3_ccb-NIXL	512	1273.2	8.64	434.93	9224.31	426805.49	1258.81	8.67	435.08	9236.43	425186.59
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	512	4951.13	14.84	293.34	18610	47739.58	2856.12	15.01	296.86	16678.95	53273.62
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	1024	22079.71	14.85	293.66	35761.02	49678.02	17993.57	15.08	298.1	31881.91	55601.5
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	4	151.59	10.25	202.31	9672.95	725.73	146.08	10.22	201.7	9638.82	744.42
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	8	167.55	10.71	211.74	10095.98	1398.74	165.37	10.74	212.33	10121.7	1385.8
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	16	194.36	11.03	217.9	10289.06	2807.49	182.44	11.25	222.24	10478.25	2755.62
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	32	221.58	11.7	231.51	10960.6	5156.02	208.62	12.11	239.53	11319.58	5001.66
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	64	266.36	12.1	239.37	11449.92	9867.05	245.22	12.57	248.7	11864.88	9522.75
Qwen3-235B-A22B-FP4_1k1k_ctx1_gen4_tep8_bs32_eplb0_mtp0_ccb-NIXL	128	339.62	10.4	205.75	9911.98	22431.48	325.54	10.93	216.06	10377.63	21411.82
Qwen3-235B-A22B-FP4_8k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	256	25868.83	13.31	263.11	38112.76	49357.09	25043.35	13.34	263.78	37318.73	50827.74
Qwen3-235B-A22B-FP4_8k1k_ctx1_gen1_dep16_bs64_eplb0_mtp0_ccb-NIXL	512	60573.08	13.29	262.7	72793.39	51342.78	53595.65	13.39	264.77	65912.25	56935.82
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	1	124.59	8.62	170.84	7919.1	232.13	122.53	8.54	169.42	7852.21	234.11
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	2	141.5	9.87	195.47	9279.77	396.3	140.84	9.78	193.6	9191.74	400.09
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	4	153.27	10.57	208.62	9971.4	715.65	153.68	10.55	208.24	9954.03	717.05
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	8	173.91	12.22	241.42	11494.28	1233.5	174.97	12.23	241.66	11506.65	1233.39
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	16	220.17	13.89	274.51	12937.49	2230.45	221.65	13.76	272	12823.11	2250
Qwen3-235B-A22B-FP8_1k1k_ctx1_gen1_tep8_bs32_eplb0_mtp0_ccb-NIXL	36	322.74	16.06	317.64	15092.05	4275.68	329.09	16.06	317.74	15102.76	4274.68"""


def parse_run(raw_text, has_gaps=True):
    """Parse tab-separated run data into dict keyed by (config, concurrency)."""
    data = {}
    for line in raw_text.strip().split('\n'):
        parts = line.split('\t')
        config = parts[0].strip()
        conc = int(parts[1].strip())
        key = (config, conc)

        py_ttft = float(parts[2])
        py_tpot = float(parts[3])
        py_itl = float(parts[4])
        py_e2el = float(parts[5])
        py_tput = float(parts[6])
        cpp_ttft = float(parts[7])
        cpp_tpot = float(parts[8])
        cpp_itl = float(parts[9])
        cpp_e2el = float(parts[10])
        cpp_tput = float(parts[11])

        if has_gaps:
            gap_ttft = float(parts[12])
            gap_tpot = float(parts[13])
            gap_itl = float(parts[14])
            gap_e2el = float(parts[15])
        else:
            # Compute gaps: positive = python slower, negative = cpp slower
            gap_ttft = (py_ttft - cpp_ttft) / cpp_ttft * 100
            gap_tpot = (py_tpot - cpp_tpot) / cpp_tpot * 100
            gap_itl = (py_itl - cpp_itl) / cpp_itl * 100
            gap_e2el = (py_e2el - cpp_e2el) / cpp_e2el * 100

        data[key] = {
            'py_ttft': py_ttft, 'py_tpot': py_tpot, 'py_itl': py_itl, 'py_e2el': py_e2el, 'py_tput': py_tput,
            'cpp_ttft': cpp_ttft, 'cpp_tpot': cpp_tpot, 'cpp_itl': cpp_itl, 'cpp_e2el': cpp_e2el, 'cpp_tput': cpp_tput,
            'gap_ttft': gap_ttft, 'gap_tpot': gap_tpot, 'gap_itl': gap_itl, 'gap_e2el': gap_e2el,
        }
    return data


def pct_diff(v1, v2):
    """Percent difference: (v1 - v2) / avg * 100. Positive = run1 higher."""
    if v1 is None or v2 is None:
        return None
    avg = (v1 + v2) / 2
    if avg == 0:
        return 0.0
    return (v1 - v2) / avg * 100


def main():
    run1 = parse_run(run1_raw, has_gaps=True)
    run2 = parse_run(run2_raw, has_gaps=False)

    # Get all keys
    all_keys = sorted(set(list(run1.keys()) + list(run2.keys())))

    # Output CSV
    writer = csv.writer(sys.stdout)

    # Header
    writer.writerow([
        'config', 'concurrency',
        # Run1 Python
        'R1_py_TTFT', 'R1_py_TPOT', 'R1_py_ITL', 'R1_py_E2EL', 'R1_py_tput',
        # Run2 Python
        'R2_py_TTFT', 'R2_py_TPOT', 'R2_py_ITL', 'R2_py_E2EL', 'R2_py_tput',
        # Python variance (run1 vs run2)
        'py_TTFT_var%', 'py_TPOT_var%', 'py_ITL_var%', 'py_E2EL_var%', 'py_tput_var%',
        # Run1 C++
        'R1_cpp_TTFT', 'R1_cpp_TPOT', 'R1_cpp_ITL', 'R1_cpp_E2EL', 'R1_cpp_tput',
        # Run2 C++
        'R2_cpp_TTFT', 'R2_cpp_TPOT', 'R2_cpp_ITL', 'R2_cpp_E2EL', 'R2_cpp_tput',
        # C++ variance (run1 vs run2)
        'cpp_TTFT_var%', 'cpp_TPOT_var%', 'cpp_ITL_var%', 'cpp_E2EL_var%', 'cpp_tput_var%',
        # Run1 gap (python vs cpp)
        'R1_gap_TTFT%', 'R1_gap_TPOT%', 'R1_gap_ITL%', 'R1_gap_E2EL%',
        # Run2 gap (python vs cpp)
        'R2_gap_TTFT%', 'R2_gap_TPOT%', 'R2_gap_ITL%', 'R2_gap_E2EL%',
        # Gap variance
        'gap_TTFT_diff', 'gap_TPOT_diff', 'gap_ITL_diff', 'gap_E2EL_diff',
        # Presence
        'in_run1', 'in_run2',
    ])

    for key in all_keys:
        config, conc = key
        r1 = run1.get(key)
        r2 = run2.get(key)

        def g(run, field):
            return run[field] if run else None

        def fmt(v):
            if v is None:
                return ''
            return f'{v:.2f}'

        row = [config, conc]

        # Run1 Python
        for f in ['py_ttft', 'py_tpot', 'py_itl', 'py_e2el', 'py_tput']:
            row.append(fmt(g(r1, f)))
        # Run2 Python
        for f in ['py_ttft', 'py_tpot', 'py_itl', 'py_e2el', 'py_tput']:
            row.append(fmt(g(r2, f)))
        # Python variance
        for f in ['py_ttft', 'py_tpot', 'py_itl', 'py_e2el', 'py_tput']:
            row.append(fmt(pct_diff(g(r1, f), g(r2, f))))

        # Run1 C++
        for f in ['cpp_ttft', 'cpp_tpot', 'cpp_itl', 'cpp_e2el', 'cpp_tput']:
            row.append(fmt(g(r1, f)))
        # Run2 C++
        for f in ['cpp_ttft', 'cpp_tpot', 'cpp_itl', 'cpp_e2el', 'cpp_tput']:
            row.append(fmt(g(r2, f)))
        # C++ variance
        for f in ['cpp_ttft', 'cpp_tpot', 'cpp_itl', 'cpp_e2el', 'cpp_tput']:
            row.append(fmt(pct_diff(g(r1, f), g(r2, f))))

        # Run1 gap
        for f in ['gap_ttft', 'gap_tpot', 'gap_itl', 'gap_e2el']:
            row.append(fmt(g(r1, f)))
        # Run2 gap
        for f in ['gap_ttft', 'gap_tpot', 'gap_itl', 'gap_e2el']:
            row.append(fmt(g(r2, f)))
        # Gap difference (R1 gap - R2 gap)
        for f in ['gap_ttft', 'gap_tpot', 'gap_itl', 'gap_e2el']:
            v1 = g(r1, f)
            v2 = g(r2, f)
            if v1 is not None and v2 is not None:
                row.append(fmt(v1 - v2))
            else:
                row.append('')

        # Presence
        row.append('Y' if r1 else 'N')
        row.append('Y' if r2 else 'N')

        writer.writerow(row)


if __name__ == '__main__':
    main()
