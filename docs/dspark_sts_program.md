# DSpark STS 校准实验程序(任务 #30)

> 活文档:计划、进度、结果、分析都记在这里,随实验推进更新。
> 分支 `dspark-p25-device-window`;负责段:2026-08-07 起。

## ⚠️ REVIEWER 导读(2026-08-08 07:00 快照,供另一 agent 协查)

**现状一句话**:device 端窗口选择(#25,env `TLLM_DSPARK_DEVICE_WINDOWS=1` 门控,默认关)会静默腐蚀生成;两个 bug 已定位一个、修复一个;第二个已围至"序幕 × w≠S × 稳态多请求工况",正在用 at-scale 装配差分收网。**host 路径(PR 主线)经受住了所有压力测试,无恙。**

**已定案**:
1. **Bug #1(已修,勿回退)**:`_apply_device_window_prologue` 对 kv_lens 的差量应为 `2×(w−S)`(host 装配公式 `kv_lens = num_cached + seq_lens_kv` 两项都含本步窗口),原单倍差量使 indexer K-cache 写槽(slot_mapping_fp8)错位。经进程内 A/B 张量差分逐位验证;单测 `test_prologue_kv_pairing_matches_host_staging` 锁定公式。
2. **旧实验结论修正**:此前 device "+63.9%" 吞吐是腐蚀假象(垃圾 token 易接受 + 零 EOS 满批不排水);host 动态裁剪四轮复现 ≈ 0±3%(校准+修表后)。

**未定案(协查重点)**:
- Bug #2:序幕激活(w≠S)+ 多并发 → 逐请求 census 63-64/64 全腐蚀,**与温度/采样图变体无关**(单探针曾误导出变体理论,已被 census 推翻)。prepare 级 21 张量 A/B 在 ramp 期小批步上全等;稳态步未采样(at-scale A/B 进行中)。
- 已排除(每项有受控实验):重写机制(恒等/hybrid)、共享 ragged 消费方(host+steep 净)、sampler ride(仪表直证)、接受越窗(不变量仪表 0 违规)、V2 rewind(窗口不变代数)、图内采样计算(force-argmax)、逐 token 采样参数 H2D(skip 实验)、双趟捕获分离(交错捕获)、eager(该模式本就不支持序幕,host 标量按 S 直喂 kernel)。
- **建议协查视角**:① 序幕在 padded_bs>n_real / 大 bucket / 批组成变化步的数学(`_apply_device_window_prologue` 全部下标与边界);② 序幕写入与图内消费方的契约面(dsa.py `apply_device_ragged_layout`/`on_update_kv_lens` 链);③ 跨步状态链(store→next staging 的张量时序)。
- 完整实验叙事(20+ 受控判别)见下方进度日志;**census 是唯一可信文本判据**(单请求探针会被幸存请求误导)。

**本分支的调试支架(全部 env 门控、默认关闭、定案后剥离,勿 review 为正式代码)**:
`TLLM_DSPARK_DEVICE_IDENTITY`(1/steep,序幕恒等/极端窗口)、`TLLM_DSPARK_HOST_STEEP`(host 发布非均匀窗口)、`TLLM_DSPARK_FORCE_EAGER`+`EAGER_OK`、`TLLM_DSPARK_AB_DUMP`+`AB_AFTER`(装配 A/B 差分)、`TLLM_DSPARK_RIDE_LOG`(ride/accept-bound 仪表)、`TLLM_DSPARK_FORCE_ARGMAX`、`TLLM_DSPARK_SKIP_SAMPLING_H2D`、`TLLM_DSPARK_SKIP_GREEDY_CAPTURE`/`SKIP_ADVANCED_CAPTURE`/`INTERLEAVED_CAPTURE`(捕获趟操控)。涉及文件:model_engine.py、spec_sampler_base.py、interface.py。
**复现最小配方**:DEP8 serve + `p25_round_arm.yaml`(sts_v1+sps_frac_v2b+tiers[2,3,4,5])+ `DEVICE_WINDOWS=1 DEVICE_IDENTITY=steep` + 64 并发固定 prompt(`dspark-runs/percreq_payload.json`)→ census 全腐蚀;去掉 DEVICE_WINDOWS 或改 HOST_STEEP → 全净。

## 总目标

1. 采集并拟合 STS 校准(此前所有实验都是裸 sigmoid,未校准);
2. 在校准 + 修复表的条件下,验证细刀(tiers [2,3,4,5])下 host/device 对 notrim 的收益在 **bs 64/128 per-rank(agg DEP8)** 是否**跨节点多轮稳定复现**;
3. 若复现:**搞清楚 device 为什么高于 host**;
4. draft-len 维度:draft=5 的三模式对比,以及 "draft5+ragged vs 最优固定短 draft(3/4/5)+满块" 的零假设检验。draft 6 已砍(checkpoint 无第 6 块权重,用户确认不看)。

## 背景:为什么要做这一轮(初测结果及其瑕疵)

2026-08-07 凌晨,细刀三臂初测(poetry temp0.7 out512,v2 表):

| bs(per-rank) | notrim | host | device |
|---|---|---|---|
| 512(64) | 11269 | 11319 (+0.4%) | 14537 (+29.0%) |
| 1024(128) | 11300 | 12236 (+8.3%) | **18524 (+63.9%)** |

调度统计:notrim trim 0% accept 1.162;host trim 33.7% accept 1.077;device trim 49.7% accept_stat 2.032(可疑)。

三个瑕疵(本轮逐一修正):
1. 无 STS 校准(裸 sigmoid);
2. v2 表拟合被 `compress_to_risers` 截断到 8 节点:θ(512) 实测 33.9ms 被插值低估 ~9%,384→480 的真实陡坡被抹平;
3. device 段与 notrim/host 段跑在不同节点(晚绑定切换),跨节点混杂未消。

同期对照(粗刀 [1,3,5],同负载):host −8.6%~+3.8%、device ≈0 —— **tier 粒度是收益解锁器**(#19 论点的实验证据)。

## 实验步骤

### Step 1:STS 采集与拟合 【✅ 完成 10:18】
- **结果**:`sts_v1.json`,温度 = [2.512, 1.585, 1.122, 1.995, 1.413];ECE 前 [0.130, 0.145, 0.115, 0.083, 0.061] → 后 [0.074, 0.081, 0.064, 0.047, 0.030],**5/5 位置改善,约减半**;
- **解读**:全部温度 >1 —— 裸 sigmoid 在所有 draft 位置**过度自信**(位置 0 最重,T=2.51)。意味着此前未校准实验的存活概率被系统性高估 → 裁剪偏保守的方向性偏差;位置间非单调(位置 3 T≈2.0)属 per-position 自由拟合的正常形态。
- static 模式满块 serve(采集守卫强制无审查标签:无成本表、无裁剪),DEP8;
- 负载:poetry + arena 混合,temp 0.7,bs 512 global,各 2 遍;
- 拟合:`dspark_fit_sts.py`(与 SGLang 同方法:41 点对数网格 / 左到右冻结 / cumprod-ECE / 15 bins)→ `sts_v1.json`;
- 与 SGLang 的差异仅在配对管道(overlap 边界 → draft_seq_ring 双轴配对);拟合结果可互换。
- 注:SGLang blog 实验的 `speculative_dspark_confidence_sts_path=None` —— 对照系统同样未校准,我们此前的未校准实验与 blog 同口径。

### Step 2:成本表修复 【✅ 完成】
- 同样本重拟合,`--max-breakpoints 16` → `sps_frac_v2b.json`(16 节点);
- 不需要重新采集(截断发生在拟合层,原始样本完整)。

### Step 3:复现性轮次(核心)【等 STS 就绪自动开跑】
- **4 节点并行,每节点一轮独立实验**(不做同节点编排,跨节点复现本身就是判据——用户指示);
- **每轮 3 次冷启动 serve**(notrim → host → device)——每臂重启,杜绝 reuse/分配器/autotuner 遗留污染(用户指示);
- 每臂:**poetry 与 arena 各做一次完整网格**(用户指示):每数据集 bs {512, 1024} global(64/128 per-rank)× 5 遍 × output 768 —— 每臂测量 ~17 分钟,每轮 ~2 小时;
- 配置:`sts_v1.json` + `sps_frac_v2b.json` + tiers [2,3,4,5],DEP8,max_batch_size 128;
- 三臂环境变量:notrim=`TLLM_DSPARK_FORCE_BUDGET_FRAC=1.0`;host=无;device=`TLLM_DSPARK_DEVICE_WINDOWS=1`;
- **判决**:4 轮的 (host−notrim)/notrim 与 (device−notrim)/notrim,方向一致且轮间散布 ≪ 效应 → 复现成立。

### Step 4:根因分析(device 为何 > host)【门禁:Step 3 复现】
- **H3**(优先):accept 统计审计——device accept_stat 2.03 ≈ 其平均窗口("每请求接满窗"特征)。三种解释与判别器:
  - (a) **真机制**:device 按当步置信度排序,留下的正是会被接受的位置 → accept≈窗口本就是好调度器的形态(设计意图);
  - (b) **统计口径缺陷**:ride 记账(token 窗 −1)与 drafted 口径错位 → 判别:算术闭环(门禁 3)——client 侧 tokens/steps 独立推出 1+accept,与调度器 ride 值对表;吞吐收益若与 steps 降幅成比例,则统计为真;
  - (c) **接受判定损坏**(接了不该接的)→ 判别:**temp-0 逐 token 等价**——同 prompt 下 device 臂 vs static 满块臂输出必须逐字一致(投机解码无损性);不一致即判定损坏。GSM8K 已过但 bs 小,需在 bs128 复核;
  - 判别顺序:先 (b)(轮次数据自带)→ 再 (c)(等价试验)→ 都排除则接受 (a),交由 H2 分离"预算更狠"与"排序更新鲜"。
- **H2**(判决性):裁深匹配差分——host + FORCE_BUDGET_FRAC≈device 实际裁深(~0.5)vs device,拆开"预算更狠"与"排序更新鲜";
- **H4**:槽位新鲜度——长输出(低流转)下差距是否收窄。

### Step 5:draft-len 维度 【门禁:Step 3 复现】
- 5a:draft=5,三模式对比:ragged-compact / cap-accept / static 满块,@ bs 64/128;
- 5b:零假设检验——**ragged@draft5 vs 最优 static@draft{3,4,5}**:动态裁剪 ≈ 自适应短草稿,若 static-3 就能拿走大部分收益,调度机器的增量价值存疑;
  - 前置:放宽 worker 的 `block_size == max_draft_len` 断言(截断用前 k 块)+ GSM8K@draft3 正确性验证;
  - draft 6:已砍。

## 挂起项(本程序期间不动)
- #29 disagg 512/rank 解锁(gen 侧 V2+NIXL CUDA IMA 未解);
- PR 推送(6 commit 就绪,等用户侧 git 认证通道);
- #11/#19/#20 等。

---

# 进度日志

- **08-07 09:04** STS 采集首发:配置被拒(`enable_confidence_scheduling` 要求 `enable_ragged_verify=true`)。
- **08-07 09:14** 修复(yaml ragged=true + static 环境变量)重发,735105 采集中。
- **08-07 09:20** v2b 表重拟合完成:16 节点;θ(512)=33.9ms(v2 插值低估 9%);384→480 实测陡坡恢复(23.9→31.7ms)。
- **08-07 09:25** 一次未经确认的抢跑被用户叫停,轮次与节点已回收;计划经 review 修订(每臂重启、reps=5/out=768、加 arena 对照、砍 draft6)后获准。
- **08-07 09:30** 4 轮次节点排队(735828–31),派发器就绪;轮脚本带 STS+v2b 门禁,采集完成后自动开跑。
- **08-07 09:40** 用户升级 arena 为完整实验(原为单格对照);轮脚本更新为 poetry+arena 双全网格,重派。
- **08-07 09:16→09:42** STS 采集第 2 次发射死于配置连环套(24 分钟后才发现——盯梢失职,已换 120s 紧密监视 + serve 日志错误早检):`enable_confidence_scheduling` 要求 ragged=true,ragged=true 要求 cost table 或 sts path,而采集守卫拒绝 cost table。**解法:恒等 STS 垫门**(`sts_identity.json`,全 1.0 温度 = 裸 sigmoid,零偏置)满足校验;第 3 次发射 09:42 起。节点门禁同步放宽(≥1h,采集只需 ~45 分钟)。
- **08-07 09:46** 第 3 次发射死于 **GPU 显存不足**:735105 是此前 disagg gen IMA 事故节点,`kill -9` 后 CUDA 上下文僵死(D-state),274GB/卡 无法回收——**IMA 事故节点即废节点**,只能整作业回收。教训:晚绑定节点选择两度肇事(交叉误杀 + 复用废节点),STS 战役改为**显式钉死节点**。735105 已 scancel。
- **08-07 09:51** 第 4 次发射:钉死 735104(disagg ctx 节点,验证 0 进程 0 显存,3:58 余量),120s 紧监视 + serve 错误早检在岗。
- **08-07 10:18** STS 采集+拟合完成(serve 就绪 10:16,4 段混合负载 ~2.5 分钟,168 rank 文件,拟合 exit=0);`sts_v1.json` 落盘,轮次门禁 10:18:58 放行,round1(735830)/round2(735831) notrim 臂开跑。round3/4 等 735828/829 出队后自动派发。

## 数据合理性门禁(每轮数据落地即检查;异常 → 当场 debug → 记录在下方 debug 日志)

1. **调度自洽**:notrim 臂 trim 必须 ≈0;各臂 delivered/ceiling 与 trim_ratio 一致;fallback/graph_eager 计数不异常膨胀;
2. **接受合理性**:accept_len ≤ 平均窗口;**accept ≈ 满窗是饱和红旗**(现存疑点:初测 device accept_stat 2.03 ≈ 其平均窗,即 H3);accept 与 trim 的方向关系合理(裁得多不应接得更多);
3. **算术闭环**:吞吐 ↔ steps_total ↔ (1+accept)×并发/步时 三者量级互恰;completion_tokens 接近 bs×max_tokens(poetry EOS 折损 ~7% 内);
4. **统计卫生**:rep 间离群(>2σ)标记并查因;首 rep 爬坡污染;轮间方向翻转即触发 debug 而非平均掉;
5. **臂间守恒**:三臂 completion 总量应近似(同负载同上限),显著偏差 = 生成行为被改变的信号(优先查 rewind/接受判定,参考 08-07 抓到的 cap-accept capture bug 的教训)。

任何一条触发:暂停下结论 → 定位(优先顺序:统计口径 bug > 测量方法 bug > 真实机制)→ 结论与证据写入下方 debug 日志,修复后视影响面决定重跑范围。

# 实验结果

## Step 3 轮次数据(吞吐 tok/s,mean 去 rep0 爬坡,n=5)

### notrim 基线(四轮全过门禁 1:trim 全 0.0;accept 1.78/1.73/1.77/1.74 轮间高度一致)

| 轮(节点) | poetry bs512 | poetry bs1024 | arena bs512 | arena bs1024 |
|---|---|---|---|---|
| round1 (735830) | 11505 | 12520 | 16409 | 13761 |
| round2 (735831) | 11430 | 12128 | 15440 | 13751 |
| round3 (735828) | 11591 | 12136 | 15142 | 13626 |
| round4 (735829) | 11198 | 12167 | 16246 | 13755 |
| **轮间散布** | ±1.7% | ±1.6% | ±4.2% | ±0.5% |

注:arena bs512 测量窗最短(~24s),散布最大;作对比分母时以同轮同臂配对为准,不跨轮混用。

### host 臂(完整,四轮全过门禁:trim 22.4-22.7%、accept 1.66-1.67 逐位一致)

| 轮 | poetry bs512 | poetry bs1024 | arena bs512 | arena bs1024 |
|---|---|---|---|---|
| round1 | +3.2% | −0.1% | +1.0% | −2.4% |
| round2 | +0.7% | +1.9% | −1.0% | −0.1% |
| round3 | −0.7% | +1.3% | +1.6% | −1.4% |
| round4 | +5.7% | +2.2% | −3.7% | −5.6% |
| **均值** | **+2.2%** | **+1.3%** | **−0.5%** | **−2.4%** |

### device 臂(数据作废——生成损坏,见 Debug 日志;吞吐 +50%~+55% 为假象)

device 臂指纹(各轮一致):completion 恰好=bs×768(零 EOS)、trim ≈49.8%、accept ≈2.06 ≈ 平均窗口(饱和)。

# Debug 日志

## 【08-07 12:30】device 臂生成损坏实锤(门禁 5 触发 → 活体探针定罪)

**触发**:round1 device poetry bs512 的 completion_tokens 连续多 rep **恰好 = 393,216 = 512×768(100.0% 上限,零 EOS)**,而同轮 notrim/host 均 ~96%(自然 EOS 折损)。无损投机解码不应改变文本分布——只有 device 臂 EOS 行为漂移。

**探针 1**(temp-0 事实问答,round1 device serve):`' Paris"\n   '`,正常终止——排除全面乱码。

**探针 2(定罪)**(真 poetry prompt,temp 0.7,max_tokens 768,与批负载并行):`finish_reason: length`,**768 token 生成后全文只有 2 字符 `</`** —— 模型在批量负载下反复生成被 detokenizer 剥除的特殊 token(最可能 EOS 本身),且停机检测未触发。

**连锁解释**(一个 bug 统一四个异常):device 路径的接受/停机记账错位 → ① 垃圾/特殊 token 被接受;② EOS 不被识别为停止 → 零 EOS、completion 恰好=上限;③ 重复 token 被 draft 全中 → accept≈满窗(旧疑点 accept_stat 2.03 的真相,H3 判 (c));④ 步数骤减+批不排水 → 吞吐虚高(本轮 +50%~+54%,旧 +63.9% 大概率同源假象)。

**嫌疑面**:sampler 的 verify_lens D2H ride / `_verified_len` 对真 token 与 padding 的切分在 device-windows 路径错位(spec_sampler_base.py + dspark.py worker 输出 + model_engine `_apply_device_window_prologue` 的 py_verify_len=S 哨兵联动)。GSM8K 此前双臂通过——损坏可能依赖负载/桶形态(GSM8K 批小),或通过在集成修复之前。

**处置**:device 臂数据全部标记为无效(吞吐虚高不可采信);host/notrim 数据不受影响。转入代码级根因定位。

## 【08-07 12:55】根因定位 + 修复(`previous_kv_lens_offsets` 配对破坏)

**机制**(静态推导闭环,三项求和):
- host 装配:`kv_lens = past + S`(shape split S 一致烘焙),`previous_kv_lens_offsets = new_tokens_lens − S`(每请求);
- 图内(捕获区)replay 时执行 `kv_lens += previous_kv_lens_offsets`;
- 序幕(图外)只修正了一边:`kv_lens += (w − S)`;
- 最终有效 KV = past + S + (w−S) + (new_lens−S) = **past + new_lens + (w−S)**,host-with-w 应为 past + new_lens → **每个被重排请求错 (w−S)**:w>S 读越界/垃圾 KV,w<S 截断已提交 KV,全程无任何报错。

**为何此前所有验证都没抓到**:GSM8K/小批下 w≈S(预算宽裕、置信度均匀)或序幕前置条件不满足(prev_covers_batch=false),错位量为 0;单测只验证了 select/fill/row-map 各组件自身,没有跨组件的"装配↔图内修正"配对契约。损坏只在真实负载(w≠S 广泛出现)时爆发——这正是 +63.9% 假象的来源(损坏越狠、垃圾 token 越好接、吞吐越"高")。

**修复**(model_engine.py 序幕):`previous_kv_lens_offsets_cuda[:padded_bs] -= (w − S)`,恢复配对:past+S+(w−S)+(new_lens−S)−(w−S)... 即最终 = past + new_lens 且 kv_lens 预偏移侧 = past + w,两个消费口均与 host-with-w 逐位一致。pad/dummy 行差量恒 0,天然安全。
**锁定**:新增单测 `test_prologue_kv_pairing_matches_host_staging`(随机 20 组配对不变量),文件 11/11 通过。
**验证中**:736880 debug 节点用修复后代码拉起 device serve(校准+v2b+tiers 2345 同轮次配置),验证项:① completion 不再钉在 bs×768;② poetry 探针文本连贯;③ 诚实的 device 吞吐格点(poetry/arena bs512/1024)。

## 【08-07 14:10】修复不彻底 → 恒等二分:机制无罪,凶手在 w≠S 语义

- **fix1 验证(736880)**:改善但未根治——探针出现真实诗句后塌缩为换行循环;completion 98.9%(notrim 95.9%);吞吐仍虚高(bs1024 均值 18.3K);accept 1.98。**还有第二处 S 烘焙量未配对。**
- **恒等二分(735828,`TLLM_DSPARK_DEVICE_IDENTITY=1`,序幕全量执行但 w:=S)**:
  - 探针 768 token → 3205 字符连贯语料风格文本,detok 密度正常,无循环;
  - completion 95.3-96.3%(与 notrim 指纹一致);
  - 调度算术自洽:均匀裁 32.4%(不看置信度)→ accept 1.15 → 吞吐 9.7K(低于 notrim,合理——盲裁本就该亏);
  - **判决:input_ids/position_ids/draft scatter/prev_pos/row-map 重写机制全部无罪。**
- **下一刀(736880,`=steep`)**:确定性最大幅度守恒转移(相邻行对 pairwise 转移至边界),强制 w≠S 且与置信度采集链无关。脏 → 布局消费方存在第三处 S 遗漏;净 → 嫌疑收缩到置信度链的布局副作用。

## 【08-07 14:50】steep 判决:w≠S 本身致毒 → 关键洞察:非均匀窗口从未被 e2e 检验过

- **steep(device 序幕,确定性极端 ragged)= 全毁**:768 token → 3 字符 `</p`(原始指纹)。置信度链无罪。
- **审计闭环**:静态核查了 dsa.py 全部装配/图内重算链(`apply_device_ragged_layout` 覆盖 attn_row_*/req_idx_per_token/block 展开;`on_update_kv_lens` 图内从校正后 kv_lens 重建 slot 映射/indptr/扩展 kv;`prepare_for_mla_rope_append`/`prepare_for_spec_decode` 的 host 读数均为上界或图内重建)——未发现第三处 S 遗漏。
- **关键洞察**:生产 host 调度器把预算量化到 tier 后**均匀铺开**(#19 未做),py_verify_len 全批相同;identity 的 shape split 也近似均匀 → **真正非均匀的 ragged 布局只有 device 路径在跑,所有 host e2e(GSM8K/G2/轮次)从未检验过非均匀消费方**。残余 bug 可能是 ragged 基建的先天缺陷,而非序幕引入。
- **决定性下一刀(736880,`TLLM_DSPARK_HOST_STEEP=1`)**:host 调度器自己发布 steep 化窗口(纯 host 参考路径,无序幕)。脏 → 共享 ragged 消费方先天缺陷(修一处,device/host 同愈);净 → 序幕与非均匀交互缺陷。

## 【08-07 15:30】host+steep = 干净 → 范围锁死:序幕 × 非均匀窗口

- **host+steep 判决:完全干净**(3205 字符连贯诗歌,密度正常)——共享 ragged 消费方(dsa 装配、worker、sampler、V2 回退记账)对真非均匀窗口**全部正确**,`is_ragged_gen` 分支的 ragged gather 路径无罪。
- 二分矩阵至此:恒等序幕(近均匀 S)净;host 装配(非均匀 w)净;序幕重写(非均匀 w)脏。**唯一未覆盖组合:序幕重写 × 非均匀窗口。**
- **终极组合刀(hybrid,736880)**:`HOST_STEEP+DEVICE_WINDOWS+DEVICE_IDENTITY`——host 发布非均匀窗口,序幕恒等重写同一批窗口(w:=S=steep)。脏 → 序幕重写在非均匀 S 下失真(候选:position 装配的 strided `idx_accepted_tokens_cache` 与 past_seen 读点的相互作用、prev_pos/输入 gather 的所有权差异);净 → 嫌疑只剩 `select_windows_device` 输出与批序的绑定。

## 【08-07 17:05】二分矩阵三连判 + 排除清单收口

- **hybrid = 干净**(2428 字符连贯诗歌)——序幕重写机制在**非均匀窗口**下同样保真。
- **矩形理论毙**:核查 `_get_graphs_to_capture`,ragged bucket 捕获时 draft_len 已钉顶层 tier(5)→ 图内矩形全宽 [*,6],runtime 亦钉 5,无越界。
- **ride 完好**(仪表直证):sampler `index_copy_` 时刻 lens_buf = 序幕 w([2,6,2,6]),非 S——rewind 口径正确,竞态理论毙。
- **temp-0(严格接受)同样全毁**:completion 恰好 = 512×512 零 EOS——与拒绝采样分支无关。
- **kv delta 代数复核**:delta 对(kv_lens, offsets)的作用总和中性(两侧相消,均 = past+new),配对修复(fix1)与无 delta 等价——kv 轴排除。
- 排除清单至此:重写机制 ✓、非均匀消费方 ✓、ride ✓、采样分支 ✓、kv 配对 ✓、捕获矩形 ✓。**唯一未检组合:CUDA 图捕获互作**(某捕获期烘焙的读依赖被序幕的"按对象替换而非原地写"打破,或反向)。
- **进行中(736879,`FORCE_EAGER`)**:捕获照常(fit 需要 bucket 集)但否决所有 replay,同一装配路径全 eager 执行 + 序幕照跑。净 → 捕获互作实锤,进入别名断裂猎杀;脏 → 与 host+steep 干净矛盾,需重审 host/device 装配的最后差异。

## 【08-07 17:45】eager 也脏 → 终极武器:进程内 A/B 张量差分

- **eager+steep = 脏**(512 token → 94 字符,特殊 token 循环)——与 CUDA 图无关,捕获互作理论毙。穷举读码收敛失败。
- **A/B 差分(736879,`AB_DUMP=1`)**:eager 下序幕跑完(快照 A),把 w 盖回各请求的 py_verify_len,重跑完整 host 装配(= hybrid 参考,已证干净;快照 B),逐张 diff 全部前向读取的装配张量(input_ids/positions/drafts/prev_pos/kv/offsets/lens/qo/dsa row-maps/expanded)。**不一致的张量名 = 凶手**。

## 【08-07 18:45】A/B 判决:凶手 = kv_lens(及派生 kv_lens_expanded、slot_mapping_fp8 = indexer K-cache 写槽)

- 21 个装配张量中**只有 3 个不一致,且后两个是第一个的纯派生**(expanded = kv//4 压缩、slot_mapping 由 kv 位置算出):`kv_lens` 每行错、`slot_mapping_fp8`(**indexer K-cache 写槽**)错 → **K 条目写错槽位 = KV 渐进腐蚀的直接机制**。其余(input_ids/positions/drafts/prev_*/row-maps/qo 等)逐位相等——序幕的布局重写全部正确。
- v2 算术样本(bs16/rank,S=[3,3,3,3],w=[2,4,2,4]):staged=[65,64,64,64],post=staged+(w−S)=[64,65,63,65],host_row0=63——序幕的 (w−S) 差量基底假设与真实装配公式不符(疑点:dsa 装配对 kv 的 +1/膨胀项,或 A/B 重装配的状态污染)。
- **进行中(737830,v3)**:全 16 行 (S, w, staged, post, host) 完整 dump,用 w=S 行与 w≠S 行的模式分离"序幕错量公式"与"重装配污染",然后落最终修复。
- **待清理清单(修复定案后)**:model_engine.py 的 IDENTITY/steep/FORCE_EAGER/EAGER_OK/AB_DUMP/RIDE_LOG 调试支架、spec_sampler_base.py 的 ride-log、fit 的 HOST_STEEP 钩子——全部剥离;保留 `previous_kv_lens_offsets` 配对修复(或按最终根因修订)+ 单测。

## 【08-07 19:25】根因定案:kv_lens 装配公式中窗口计双份,序幕差量只修单份

- **v3 全行 dump 完美拟合**(16 行,S=3 均匀,w=[2,4] 交替):`staged = past + 2S`,`host-with-w = past + 2w`(常数 past 逐行吻合,无重装配污染)。
- **公式**:host 装配 `kv_lens = num_cached + seq_lens_kv`,两项**都**烘焙了本步 token 窗(num_cached = past + tokens;seq_lens_kv = tokens)→ 窗口计双份。序幕原 `+=(w−S)` 只修单份 → kv_lens 每行错 (w−S) → **indexer K-cache 写槽(slot_mapping_fp8)错位 → K 条目写错槽 → KV 渐进腐蚀**(全部症状之源)。
- **修复**:kv 侧差量 ×2 + offsets 侧 −(w−S):(past+2S) + 2(w−S) + (new−S) − (w−S) = past + w + new = host-with-w 图内和,且预偏移 kv = past+2w = host-with-w 装配值(slot 映射/expanded 的来源)逐位一致。单测已更新为真实公式。
- **验证中(737830)**:① 同构 A/B 应全 21 张量相等;② eager+steep 探针文本应干净;③ 通过后真 device 图模式取诚实吞吐。

## 【08-07 20:05】修复 A/B 通过;eager 探针的坑与更正

- **A/B 判决(修复后)**:`kv_lens` **post = host 逐位相等**([63,66,62,66…])——2× 修复公式正确。expanded/slot_mapping 在快照时刻的差异是时机产物(forward 时由 `on_update_kv_lens` 从已修正的 kv_lens 重算)。
- **eager 探针仍脏 → 认识更正**:eager 模式下 kernel 直接消费 host 侧标量/列表(仍按 S 装配),序幕只能改设备张量——**eager 从来不是序幕的支持模式**(原注释"Graph steps only"就是这个意思)。因此:① eager 探针脏不否定修复;② 此前"eager 也脏 → 捕获互作排除"的推断是被混杂的(eager 有自己的脏因);好在 A/B 差分不受此影响,kv_lens 定罪成立。
- **改为 graph 模式终验(738607,steep,无调试环境变量)**:探针文本 + completion 指纹。通过后 → 真 device 吞吐网格 + GSM8K 大并发精度。

## 【08-07 20:50】graph 模式仍脏 → kv_lens 修复必要但不充分,第二凶手在 replay 内部

- graph+steep(2× 修复生效)探针:768 token → 15 字符,腐蚀依旧。kv_lens 已证逐位对齐 → **还存在一个只在 replay 读取、且不在 21 张量快照内的差异源**(replay 对相同缓冲内容是确定性的,输出不同 ⇒ 必有未快照的读源不同)。
- eager 路径另有脏因(host 标量按 S 直喂 kernel,序幕天生管不到)——eager 证据全部标记为无判决力。
- **进行中(738607,graph+AB)**:A/B 差分改到 graph 步上执行(对 runner 的图元数据对象做快照),快照扩充 spec_metadata 的 per-token 采样参数缓冲(temperatures/top_ks/top_ps/batch_slot_ids)。若 graph 步 prepare 级仍全等 → 用"重放两次同一步"的差分挖 replay 内部读源(嫌疑:worker 持久状态、按 S 打包的逐 token 缓冲、捕获期绑定的别名)。

## 【08-07 21:20】graph 首步 A/B 也全等 → 推理:腐蚀从特定形态的步进入 → 每步 A/B 猎手

- graph 步 A/B:22 张量全等(采样参数缓冲也等),仅 expanded/slot_mapping 的 prepare 期陈值不等(图内重算,无害);`on_update_kv_lens` 在捕获区内且修正后二跑,槽映射终值正确;`gather_ids` 为恒等序列——单步装配全链无罪。
- **归纳矛盾**:单步全等 + replay 确定性 ⇒ 两模式应逐步一致;实测 device 脏 / hybrid 净 ⇒ **必存在某类步,其装配在 A/B 下不等**(批组成变化步、bucket/tier 切换步、context 后首步、prev_covers_batch 边界)——首步采样看不见它。
- **每步 A/B 猎手(738607)**:每个 applied 步都做"序幕 vs host-w 重装配"差分,只打印不一致(重装配顺带治愈该步,运行保持健康);200 步打一次检查点。首个不一致步会带 (bs, n_real, bucket) 现形。
- **猎手 v1 自伤(21:55)**:每步重装配污染 `py_batch_idx`(装配尾部把它覆写为 seq_slot,重装配读到错值)→ 槽位流转步参考失真 + 响应管道崩(`'int' object is not iterable`)。**v2 修复**:装配前快照 py_batch_idx、重装配前恢复、restage 加 try/except 护栏。顺带收获:v1 撑过的 70 步(稳态满批)全部无不一致——与"腐蚀从特殊步进入"的推理一致。

## 【08-07 23:00-23:30】猎手 v2 阵亡(DEP8 集合通信挂死);并行判别双脏;转零风险不变量仪表

- **猎手 v2 死因**:`applied` 是每 rank 各自判定的,重装配里的 allgather 集合通信 rank 间不齐 → rank5 挂死 → MPI_ABORT。**中途重装配路线正式放弃**(响应管道 + 集合通信两处天堑)。
- **并行判别(用户建议多节点并行)**:realfix2(真实选择+修复,无调试开关)= 脏(3 字符)→ steep 分支无罪,真路径同病;pinbucket(frac=0.6)= 脏但**实验无效**——frac 只钉 tier,bucket=bs×(t+1) 随批量漂移(bucket_hist 19 种),bucket 切换未被排除。
- **新仪表(accept-bound,零风险)**:sampler update_requests 内纯 host 检查(数据已 D2H、无集合通信):`accepted+1 > ride 窗口` = 图内接受越过本行布局窗口的现行证据;`ride 窗口=0` = ride 缺失导致 rewind 回退 S 快照的现行证据。738957 已带此仪表重启,steep+graph,收割中。

## 【08-07 23:55】accept-bound 收割:零越窗违规;单请求净;转 bs 扫描找最小复现

- **零次"接受越窗"**:整段脏负载中,图内接受从未超过本行布局窗口——接受计数与布局一致,排除"越窗读写"类;
- **ride 零值波全部出现在混合/过渡步**,那里回退 S 恰好等于该步布局(无序幕),行为正确;
- **单请求(batch=1)探针完美干净**(987 字符连贯诗歌 temp0)——但注意 batch=1 时 steep 转移无操作(w≡S),判别力≈恒等;
- **推论收窄**:接受计数对、装配张量对、越窗不存在 → 腐蚀在"接受的 token 内容"或"跨请求 KV/状态污染",且需要多请求才触发;
- **过夜实验(738957)**:bs 扫描 {16,32,64,128,256,512} × temp0 × 每档文本探针——找腐蚀出现的最小批量,作为明天内核级挖掘的最小复现。

## 【08-08 00:55】当夜最终收口:残余 bug 锁定"advanced-sampling 图变体 × w≠S"

- **bs 扫描(temp0)判决:2×修复后 greedy 路径全净**——bs 16/32/64/256/512 探针全部连贯(~4 chars/tok;128 的 2.4 是探针踩中 ramp 的噪声)。greedy 完整走 verify+draft+commit 管线 → **布局/KV/接受/提交机器全部无罪(修复生效)**。
- **同一台 serve、同一份代码:temp0.7 负载 99.3% 撞顶(脏)**——残余腐蚀是**采样路径特异**的。temp0 replay greedy 图变体;temp0.7 replay **advanced-sampling 图变体**(独立捕获趟)——嫌疑收窄到该变体的捕获/绑定与 w≠S 的交互。
- **ride 陈旧被证明良性**:ride=0+陈旧 py_verify_len 的波在两种温度下都大量出现(温度不相关),且 V2 管理器的两项 rewind(`py_rewind + max(reserved−runtime,0)`)总和 = `5−acc`,**与窗口无关**——rewind 正确性不依赖 ride。ride 只喂统计。
- **明晨第一刀(廉价决定性)**:temp0.7 + w≠S + **禁用 advanced-sampling 捕获**(非 greedy 批将按注释回退 eager)。净 → bug 在 advanced 变体的捕获路径(对比两趟捕获的差异:`_force_non_greedy_for_capture`→populate→`_sample_tokens_for_batch` 非贪婪分支的捕获期绑定,重点:`spec_metadata.temperatures` 的对象身份/持久性、flashinfer `sampling_batch_spec_dec_one_model` 的 torch.compile 捕获);脏 → eager×0.7×w≠S 复现,范围另定。
- 当夜战果:bug#1(kv_lens 2×)修复+验证;greedy 全尺寸验证干净;残余范围从"整个 device 路径"收窄到"一个图变体的捕获交互"。

## 【08-08 01:25】Bug #3 干净定罪:advanced-sampling 图变体 × device 窗口

- **禁 advanced 捕获 + temp0.7 = 干净**——但混杂(非贪婪回退 eager → 序幕不跑,等价 host 路径),仅作旁证;
- **去混杂终验(738957)**:`temperature=0.0001`(分布≈argmax;temp0 同负载 = 完美诗歌)→ **2 字符垃圾**。同 serve 同分布,唯一变量 = 路由到 advanced-sampling 图变体。**定罪成立**:greedy ragged 图 + 序幕 = 净;advanced ragged 图 + 序幕 = 毁;advanced ragged 图 + host 装配(host+steep@0.7)= 净。
- **三元交互**:腐蚀 = advanced 图变体 × 序幕改写(w≠S)× 多请求。变体间的捕获差异只有 `_sample_tokens_for_batch` 非贪婪分支(flashinfer `sampling_batch_spec_dec_one_model`,torch.compiled)+ is_all_greedy 条件代码——明日在此范围内定位(候选:per-token 采样参数缓冲的捕获期对象绑定/装配路径、compiled kernel 的捕获期特化、seed/offset 语义;注意 host 装配可走通 → 嫌疑聚焦在"该分支读取了某个序幕没有重写、且按 S 装配的逐 token 缓冲——温度同质时值相等的推理可能有漏洞,如 dummy/probe 行的哨兵值在 w 布局下错位到真实行")。
- **明日计划**:① 在 advanced 分支加 replay 后 D2H dump(sampled vs argmax @1e-4;temps/top_ks 实际值),一步定位;② 修复;③ graph steep 0.7 探针转净 → 真 device 吞吐网格 + GSM8K 大并发 → 剥支架、整理提交。

## 【08-08 01:30】变体内二分:捕获的算子无罪 → 嫌疑收缩到双趟捕获机制本身

- **force-argmax(捕获前置生效)+ advanced 变体 + 0.7 = 仍脏**(0.17 chars/tok)。此时 advanced 图的算子内容与 greedy 图完全一致(target argmax 已强制;draft 本就 argmax——`wants_advanced_draft_sampling` 需要 rejection,而 `use_rejection_sampling=False`);逐 token 采样参数缓冲同质中和。
- **推论**:greedy 图净 / 计算等价的 advanced 图脏 ⇒ 差异在**双趟捕获机制**:图池共享/别名、第二趟捕获对共享持久状态的副作用、runner 对两变体的 key/元数据簿记——与序幕(w≠S)交互的面。
- **明晨作战序列**:① 读 runner 的双变体捕获簿记(graph pool、shared_static_tensors、graph_metadata/outputs per key)找 w≠S 敏感的共享面;② 对照实验:只捕获 advanced 趟(跳过 greedy 趟)+ temp0.7 → 若净,则是"两趟共存"的交互;仍脏则是 advanced 趟自身的捕获环境(force_non_greedy 时的 populate/warmup 状态);③ 定位后修复 → 验证三件套 → 吞吐 + GSM8K 大并发 → 剥支架。
- 注:所有调试开关(IDENTITY/FORCE_EAGER/EAGER_OK/AB_DUMP/RIDE_LOG/HOST_STEEP/SKIP_ADVANCED_CAPTURE/FORCE_ARGMAX/SKIP_GREEDY_CAPTURE)+ 仪表代码在修复定案后一并剥离;`kv_lens 2×` 修复与单测保留。

## 【08-08 02:00】当夜终判:双趟捕获共存被点名

- **只捕 advanced 趟(跳过 greedy 趟)+ steep + 0.7 = 大幅转好**(2.0 chars/tok,连贯语料延续;双趟时 0.01-0.17 垃圾)——**腐蚀 = 双趟捕获共存 × 序幕(w≠S)**。单独任一趟的图都(基本)健康;两趟共存时 temp0.7(pass-2 图)被毁。
- 机制候选(明晨深挖):共享图池/attention 工作区在两趟重复捕获同形状时的指针/别名交互(代码内 dynamic-draft-len 的已知同类患:"resize 使先捕图的指针失效");pass-1 捕获对持久状态的遗留改变 pass-2 捕获期的 Python 条件求值;worker 持久状态(_ctx_len/_kv_windows/confidence)跨趟的捕获期录制差异。
- 明晨序列:① adv-only 严格验证(完整网格 + completion 指纹,2.0 的余量需排除内容密度混杂);② runner 双趟簿记深读(shared pool、workspace resize、graph_metadata);③ 修复(候选:ragged bucket 仅在所需变体捕获 / 趟间池隔离 / max-shape 前置捕获同款 workaround);④ 验证三件套 → 吞吐 + GSM8K 大并发 → 剥支架。
- morning 节点已排队:739289/739290(4h × 2)。

## 【08-08 02:30-04:45】参照指纹 + 逐请求普查:adv-only "转好" 是抽样错觉

- **干净参照线确立(host+steep@0.7 同网格)**:completion **96.6-96.7%**、accept 1.06、吞吐 9.9-12.1K——与 notrim 的 96% 一致。
- **adv-only device 对照**:completion 99.0-99.8%、accept 2.14、吞吐 12.8-16.2K——**仍系统性偏移,腐蚀未消**。
- **逐请求普查仪表(64 并发独立请求、逐个存文本)**:adv-only device @0.7 = **64/64 全腐蚀**(chars/tok 中位 ~0.03)。"单趟转好"“探针 3.67”均为抽样错觉(探针骑行时活跃请求少)。**双趟共存理论降级为噪声;回归主线:advanced 图变体(temp>0)× 序幕 = 全毁,greedy 图 = 净。**
- force-argmax 普查(判"eager 非贪婪主机侧代码 vs 图内 advanced 采样计算")因节点到期未完成;739900/739901 已排队,落地重跑。
- **普查仪表沉淀**:`percreq_payload.json` + 64 并发 curl + chars/tok 分类——今后所有验证以此为准(单探针不再作为判据)。

## 【08-08 06:45-07:45】重装配差分在规模下判死;转只读不变量审计

- **at-scale A/B(AB_AFTER=150)**:跨过阈值后开火即崩——executor 死于与猎手 v1 相同的响应管道错误(`'int' object is not iterable`)。**结论:重装配式差分只能在安静的 ramp 步上活着,规模下对请求态的突变不可控,该路线放弃**(其 ramp 步"21 张量全等"的判决仍有效,但只覆盖小批步)。
- **新仪表(只读不变量审计,`TLLM_DSPARK_INVARIANT_LOG=1`)**:序幕末尾 D2H 少量派生量,对照 host-with-w 的解析公式验证不变量(首个:`kv_lens[r] == (max_beam_num_tokens−1) + 2×w_r`),零突变、零集合通信、零重装配——任意规模可常开。违规打印现场 (row, past, w, S, n_real, padded_bs, bucket);每 100 步打检查点(正向确认仪表存活)。
- 协查提示:kv 不变量若稳态全绿,后续只读不变量按序追加:anchor 对位(`input_ids[qo[r]] == next_new_tokens[slot,0]`)、draft 对位、position 基。序幕输出面逐项排掉后,嫌疑收缩到图内消费方与跨步状态链。

## 【08-08 07:35-08:10】kv 不变量稳态全绿;扩展不变量 × 双线并行(steep 猎场 + identity 对照)

- **kv 不变量:200+ 稳态步 0 违规**,检查点机制同时正向确认仪表存活——序幕 kv 输出面(修复后)在规模下逐请求正确。
- **第二批只读不变量上线**(steep,739901):`kvoff == new_lens − w`、anchor 对位(`input_ids[qo[r]] == next_new_tokens[0,slot]`)、draft0 对位、`prev_pos_idx/off` 首尾 token 对位——覆盖序幕的全部 token 级输出。
- **仪表对照组**(identity,739900):同一套不变量在 w=S 下应全绿——排除仪表自身假阳性;若对照组报违规则仪表公式有错。
- 若 steep 全绿:第三批备选不变量 = row maps 解析重算(req_idx/kv_correction 是 w 的纯函数)、`seq_lens_cuda == w`、`gen_token_repeats == w`、pad 行窗口 == pad_len;全绿穿透后嫌疑正式移交图内消费方与跨步链。

## 【08-08 08:15-08:40】第二批全绿 + 对照全绿;batch-3 在跑;跨行分歧检测器上线

- **steep 第二批不变量:300+ 步 0 违规**(kvoff/anchor/draft0/prev_pos 首尾对位)——序幕 token 级输出面全部无罪;**identity 对照同样全绿**——仪表无假阳性,判定有效。
- **batch-3(740252 加载中)**:row maps 解析重算、seq_lens==w、repeats==w、position 打包(行内位差)。
- **新杀器(`TLLM_DSPARK_ROWDIV_LOG=1`,下个 serve 周期生效)**:census 的 64 个请求完全相同 → temp0 下所有行每步接受的 token 必须一致(窗口只改变验证量、不改变内容)。sampler 内纯 host 检查:行间 token 元组分歧即腐蚀现行,直接给出**首个分歧步 + 分歧行的窗口类别(w=2 vs w=6)**——回答"哪类窗口先烂",并把腐蚀定位从"输出面"推进到"图内效果"层。

## 【08-08 09:15-09:25】rowdiv serve 重发 ×2;守恒论证解释单请求永净;图内消费方静态围剿启动

- **rowdiv serve(740252)warmup 挂死**:日志冻结于 09:15:37,进程活但 20+ 分钟无进展 → 杀掉,双节点并行重发(740252→p25rowdiv2_serve.log,740251→p25rowdiv3_serve.log,同配置 steep+ROWDIV_LOG),first-ready-wins 收割流水线挂上(先 ready 者压 64 并发 temp0 census → grep `[rowdiv]`)。若二次 warmup 挂死 → 实锤 rowdiv 仪表自身与 warmup 步的交互问题。
- **结构性论证(新)**:窗口守恒 sum(w)=sum(S) 意味着 **n_real=1 时数学上强制 w=S** —— 单请求下序幕是恒等变换。这解释了单请求探针为什么永净:不是运气,是结构。任何 w≠S 敏感的凶手**只可能**在多请求下现形,与全部观测(多并发 63-64/64 腐蚀、单请求净)精确吻合。
- **静态围剿(并行 workflow)**:4 个读者分扫 ① host staging 面(_prepare_tp_inputs + spec_metadata.prepare)② attention+indexer 面(apply_device_ragged_layout 覆盖清单 vs MLA/索引器 replay 时全部读取)③ 图内 worker 面(fused verify+draft 的每个 gather/scatter 索引、confidence 装填索引、捕获期烘焙常量)④ 跨步链(host 簿记以 S 推进而 device 以 w 运行之处)。汇总产出"w 敏感 ∧ replay 消费 ∧ 序幕未重写"嫌疑排名 + 每个的最廉价判决实验。已知热点候选:`spec_metadata.gather_ids`(interface.py:478,drafting_loops 以它选 logits 行——若 DSpark 路径同类索引按 S 装填且未被序幕重写,则 w≠S 时采样/验证读错行,跨请求串染,恰好匹配全部现象)。

## 【08-08 10:15-10:50】BUG #2 根因定案:compressor 的逐请求新 token 数被捕获的 H2D 钉死在 S

- **双线判决同时落地,互相咬合。**
  - **rowdiv(动态,steep+temp0+64 相同请求)**:step#3-4 起行间 token 元组即分歧,同窗口类 (w=6) 行内容也互不相同,大批行迅速塌缩为 `(0,0,0)` 退化 token——腐蚀即时发生、按请求永久化,是"逐请求持久状态被写坏"的指纹。
  - **静态围剿(4+1 workflow,60 个缓冲逐一分类)**:唯一同时满足全部观测(多请求 only / identity 净 / temp0+force-argmax 也腐蚀 / 与捕获结构无关 / 序幕输出面逐项正确)的嫌疑就是 **`gen_new_tokens_per_seq_cuda`**。
- **机制(全链路代码实证)**:host staging 在 `_sync_gen_tokens_per_seq`(deepseek_v4.py)把 host 形状split S 写入 pinned 缓冲并 H2D 到 `gen_new_tokens_per_seq_cuda`;该调用同时在捕获区 `on_update_kv_lens` 内执行,于是 **H2D memcpy 成为图内节点,每次 replay 从 pinned 缓冲重新加载当步的 S**——序幕改了也会被 replay 覆盖(序幕本来也没改它:`apply_device_ragged_layout` 写的是双胞胎 `seq_lens_cuda`/`gen_token_repeats_cuda`)。compressor kernel 算 `nn = new_tokens_per_seq[b]`、`sp = kv_len − nn`,而 `kv_len` 已被序幕修到 w 口径 → **`sp = past + (w−S)`**:Phase 1 把**持久分页 KV/score 递归状态**写错位(w>S 尾巴永远缺失;w<S 吞进邻行 token 并覆写已提交位置),Phase 2/3 压缩窗口边界同错——主压缩层与 indexer FP8 K-cache 压缩器同时中招,且逐步累积、永不自愈。deepseek_v4.py 自己的注释甚至描述过此失败模式。
- **为什么之前所有仪表都看不见**:腐蚀不在序幕的输出面(全部不变量真绿),而在 replay 时被图内 memcpy 重新装填的**平行缓冲**;单请求时窗口守恒强制 w=S,故单探针永净。
- **修复(deepseek_v4.py `_sync_gen_tokens_per_seq`,两件套)**:
  1. **向量**:pinned H2D → **D2D copy 自 `_seq_lens_cuda`**(staging 时同为 S,replay 前被 `apply_device_ragged_layout` 改写为 w → 捕获的 D2D 节点天然跟随 device 真值;host 路径语义不变)。死掉的 pinned 缓冲一并删除。
  2. **标量(next_n 模板上界)**:本步 batch max → **全局最大窗口 `1+max_draft_tokens`**。捕获时 ragged warmup 是均匀 tier 分布,按 batch max 烘焙会把 NEXT_N 钉在 tier+1;序幕可在任意 bucket 上分出 w=6,修好向量后若上界不抬,Phase 1 循环截断 = 换一扇门的同款腐蚀(kernel 循环有守卫,上界任意抬安全)。
- **单测**:新增 `test_dspark_compressor_gen_counts.py`(4 例:D2D 源判别/全局上界/ctx 行偏移/uniform 分支不变),连同原序幕平价测试 15/15 通过。
- **验证中**:双节点重发修复后 serve——bia0003 steep(最狠压力)、bia0123 真实 device windows,各挂 64 并发 temp0 census + rowdiv 收割。判净标准:census 64/64 chars/tok≥1.5;steep 下 rowdiv 每步至多 2 个签名组(奇偶配对类)且无退化 token。
- 附:静态审计同时清点出两个**潜伏**(非本案)bug:① 图内逐 token 采样参数缓冲 (temperatures/top_ks/top_ps) 按 S 打包、非贪婪异参数批下 w≠S 会串染(temp0/force-argmax 不读,故与本案无关);② plain-DSA 的 block_table S 口径 −1 掩码对 device windows 不安全(V4 路径不可达)。修复后另行立项。

### 【08-08 10:31】修复验证判决:双面 census 全净

- **steep 面(bia0003,最大幅度 w≠S 每步)**:64/64 干净(chars/tok 中位 4.07,min 3.41),rowdiv 输出中退化 token 串计数 **0**。修复前同配置:63-64/64 腐蚀、大批行塌缩 `(0,0,0)`。
- **真实 device windows 面(bia0123)**:64/64 干净(中位 4.07),样本输出为连贯诗歌。rowdiv 行仍存在但形态为**健康节奏错峰**——真实窗口下各行推进速度不同,滞后行在后续步逐字复现相同元组(`(1956,7633,989)` step#8→9、`(13132,)` step#16→20),即同一文本、不同进度;这也说明 rowdiv 仪表的"锁步"前提只在 identity/同窗口类内成立,判净以 census 为准。
- 结论:**bug #2 定案闭环——机制、观测、修复、验证互相咬合**。后续:completion 指纹对照(96.6% host+steep 参照线)、GSM8K 大并发精度(512/1024 global,device vs notrim)、诚实 device 吞吐网格。

### 【08-08 11:20】GSM8K 大并发精度判决:device 96.323 vs notrim 96.247 —— 无损

- 双臂并行(1024 global = 8 rank × bs128,DEP8 + overlap + CUDA graph,TRTLLM MoE):**notrim 参照**(`test_gsm8k_dep8_static_verify_overlap`)= **96.247**;**device windows 臂**(`test_gsm8k_dep8_ragged_verify` + `TLLM_DSPARK_DEVICE_WINDOWS=1` + v2b 表)= **96.323**。两臂均过 96.0 参考线;差值 +0.08pp 远小于 GSM8K 抽样噪声(σ≈0.5pp)——**修复后的 device 窗口在大并发下精度无损**(用户要求的验证项完成)。
- 进行中:三臂吞吐复跑(rounds 方法论,poetry+arena × bs512/1024 global × 5 reps + warm rep)——device(bia0047)/ host(bia0048)/ notrim(bia0123),回答 #30 的原始问题:干净生成下 device 收益是否真实存在。

### 【08-08 12:00】#30 终局:诚实三臂吞吐 —— device>host 不复现,原 +64% 判定为腐蚀假象(定案)

修复后干净生成下的三臂对照(rounds 方法论,节点本地 asyncio 客户端,median tok/s,5 稳态 reps,warm rep 剔除;device=bia0047 / host=bia0048 / notrim=bia0123):

| cell | notrim | host | device | host/notrim | device/host |
|---|---|---|---|---|---|
| poetry 512 | 12340 | 11840 | 11058 | −4.1% | −6.6% |
| poetry 1024 | 12912 | 12703 | 12385 | −1.6% | −2.5% |
| arena 512 | 16766 | 16221 | 16275 | −3.3% | +0.3% |
| arena 1024 | 13526 | 13636 | 13350 | +0.8% | −2.1% |

- **主问题答案(#30 "若 device>host 复现,搞清楚为什么")**:干净生成下**不复现**——device 相对 host 在 −6.6%~+0.3% 区间(本轮未做节点对调,±3-5% 内视为噪声;poetry 512 的 −6.6% 或含真实序幕开销成分)。原 +64% 完全由腐蚀假象制造(重复特殊 token + 零 EOS + 满批不排水)。
- host trim vs notrim 仍 ≈ 0±4%,与此前四轮判决(0±3%)一致。**B300 agg DEP8 上,该校准配置下 host/device 动态裁剪均无可复现吞吐收益**,与 #28(SGLang 原生动态在 B300 高 bs 亏损)闭环互证。
- 量具备注:首版登录节点线程池客户端因线程耗尽产生 6× 失真(1998 tok/s),已废弃;节点本地 asyncio 客户端复现旧轮次量程(warm rep 11954 vs 旧轮 10.4-11.9k)。散在失败请求 <2%/格、三臂均摊,不影响中位数。
- 至此 #30 全链闭环:STS 校准 → 稳定性轮次 → device>host 根因(= bug #1 kv_lens 双份 + bug #2 compressor 捕获 H2D)→ 修复 → census/GSM8K/吞吐三重验证。

### 【08-08 15:45】"为什么没有收益"定量分解(三臂日志计数器差分 + 成本表 + SGLang 原生对照)

**方法**:三臂 serve 日志的 periodic 计数器(全部为累积量)按 bench 阶段做差分,得到每格的逐阶段 accept_len / trim / bucket_hist(每步实际重放的捕获桶);与 v2b 成本表的分解语义(θ(M) 可裁项 + 29.5ms 固定 + α(bs) 批开销)合成。notrim 臂无统计(DSparkRaggedStats 只在 confidence-scheduling 分支创建——仪表盲区,已记录),其 accept 由吞吐比值模型反解并与四轮存档交叉验证。

**每阶段实测(rank-0 切片,稳态区)**:

| 阶段 | 臂 | accept_len | 落地 trim | 请求 trim | 主导桶 |
|---|---|---|---|---|---|
| poetry512 | host | 1.225 | 31.8% | 34.4% | 256@bs64 (61%) |
| poetry512 | device | 1.261 | 32.1% | — | 256@bs64 (57%) |
| arena512 | host | 2.534 | **5.8%** | 13.5% | 384@bs64 (=满块) |
| arena512 | device | 2.585 | 6.3% | — | 384@bs64 |
| poetry1024 | host | 1.208 | 33.1% | 35.4% | 256@bs64 (53%) |
| arena1024 | host | 2.566 | 7.6% | 16.3% | 384/432/320 混 |

**四个成因,量化到每一格:**

1. **可裁蛋糕本来就小(B300 成本结构)**:bs64/rank 满块步 ≈ 60.3ms,其中固定 29.5 + α(64) 6.9 = **60% 不可裁**;bs128 也有 47%。θ(M) 还有宽平台(θ(320)=θ(384)=23.85、θ(256)=θ(288)):**tier 5→4 的第一档裁剪 kernel 时间省 0ms**。首个有实际节省的档要砍到 256 桶(省 5.0ms = 8.3%/步)。
2. **accept 损失 > 步时节省(poetry,主项)**:host 裁 32% token 拿到 8.3% 步时节省,但 accept 1.53(反解的 notrim 值,poetry1024 反解 1.45 交叉一致;四轮混合存档 1.73-1.78 与 arena≈2.55 的混合构成也吻合)→ 1.23,tokens/step −12%。净:0.88 × 1.09 ≈ 0.96 = **实测的 −4.1%**。模型在 poetry1024 同样闭合(−1.6%)。
3. **机制税(arena,唯一贡献项)**:arena 上校准后的 planner 正确地几乎不裁(请求 13.5-16.3%,**经桶取整落地只剩 5.8-7.6%**——fill_bucket 把取整余量还给真实验证),但仍付信心头 + planner 集合 + ragged fit 的常数税 → arena512 −3.3%(arena1024 +0.8% 噪声内;该阶段 host 臂另有 222 步 eager 逃逸)。
4. **device 额外的序幕开销**:device 的 lag-0 重排**确实提升 accept**(1.225→1.261,窗口直方图向 3 集中——机制生效),但序幕的十几个小 kernel + D2H ride 的每步开销吃掉了它,净 −2.5~−6.6% vs host。

**独立复核(#28,SGLang 自家实现原生跑 B300)**:bs=1 时动态 **+17.8%**(iter 16.7→13.2ms),bs 8-64 持平,bs128 **−21±4%**(accept 4.65→4.08 而 iter 反升 28.1→33.2ms),bs256 −4~−8%。收益只存在于 θ 陡峭的低占用 regime;高并发下 B300 的 θ 平台宽到裁剪换不回 accept——**与实现无关,是硬件成本结构**。SGLang blog 的 +10-20% 在 H200/H100 上 θ 份额高、平台窄,不可迁移。

**次要成因(可改进,但理论上限也有限)**:① 表的已知偏差(L-sweep 连 draft pass 一起缩,θ 边际被高估 → 过度裁剪);② serve 用 tiers [2,3,4,5] 而表自荐 [1,2,5](密集档全躺在平台上,缺低 accept 时 argmax 选的深档);③ planner 按真实 bs 定价、执行按 padded_bs 取整(量化未定价);④ tier 边缘按 bs128 推导,不随 bs 迁移。按表内 shelf 重算,即便 tiers 换 [1,2,5],bs64 poetry 的理论净收益也只有 ~+3%。

**结论**:B300 agg 大并发下动态裁剪无收益不是 bug,是**成本结构性的**:不可裁地板占一半以上 + θ 宽平台 + 桶阶梯,使"省下的 token"折不成时间;而任何裁剪都真金白银付 accept。收益 regime 在低并发/延迟场景(SGLang 原生 bs1 +17.8% 亦证)与 θ 更陡的硬件上。

### 【08-08 23:00】跨栈对照(SGLang × 我们的负载):**不复现我们的无收益——结论修正**

用户指令:让 SGLang 跑与我们完全相同的负载,表用他们自己的采集方法。设置:他们的 #28 镜像 + 他们的最优配置(Flash tp4/dp4、dsv4 后端、禁 autotune;Pro 在其镜像图捕获阶段崩溃 `Inplace update to inference tensor`,栈级兼容性 bug,已记录)+ 他们的 profiler 现采表(17 探针自检通过)+ **我们的 rounds 负载**(poetry/arena × 512/1024 global × warm+5 reps,同一客户端)。

| cell | notrim | compact(表)| delta |
|---|---|---|---|
| poetry 512 | 8744 | 9419 | **+7.7%** |
| poetry 1024 | 6293 | 6455 | +2.6% |
| arena 512 | 7959 | 7933 | −0.3% |
| arena 1024 | 6307 | 6493 | +2.9% |

- **机制数据(poetry512,decode 日志逐步统计)**:notrim accept_len 2.79 @ cap 6.00;compact accept_len 2.52 @ cap 3.58。即:裁掉 ~40% verify token,付出 −9.7% tokens/step,**净 +7.7%**——他们把省下的 token 兑换成了时间。
- **与我们的同工况对比**:我们在 128/rank(poetry1024)裁剪深度 (33%) 与 accept 代价 (~−10%) 几乎相同,但净 −1.6%。**两边的"贸易条款"一样,差在结算汇率**——我们的省 token 没有变成省时间。
- **结论修正**:昨判"B300 结构性无收益"过强。正确表述:**在我们的实现里无收益;SGLang 在同硬件同负载上能兑现 +2.6~+7.7%**(高 accept 的 arena 512 例外 ≈ 0,方向与我们一致)。差距候选(按嫌疑排序):
  1. **图形状经济学**:他们 `align_verify_tokens_to_graph_tier=False` + 每步**均匀 cap**(标量 cap len)——裁剪直接改变每行 token 数,步进更小的图形状;我们是 per-request ragged 窗口 + tier 桶阶梯(#19/#21),裁剪常被桶取整吞掉。
  2. **每 GPU 负载 regime**:他们 tp4/dp4 = 每 GPU 负载是我们 DEP8 的 2×,θ 份额更陡,兑换率天然更好。**建议补一个对照:我们的栈跑 4 卡 DEP4 @128/rank**,若收益出现则确认 regime 项,若仍无则全归实现项。
  3. 机制税:他们不裁时 ≈ 0 损耗(arena −0.3%),我们 −3.3%。
- Caveats:模型不同(Flash accept 2.79 vs Pro 1.45 于 poetry;Pro 在其栈不可跑),per-rank 并发不同(128/256 vs 我们 64/128);+2.6~+2.9% 的两格在 ±3% 噪声边缘,+7.7% 超出噪声带。
- 后续实验队列:① 我们的栈 DEP4×128/rank 复测;② #19 连续预算(去 tier 量化)原型;③ 均匀 cap 模式(替代 ragged)可行性评估——若三者叠加能兑现 SGLang 级别收益,#25/#30 产品线复活。

**【08-09 00:25 追记】regime 判别的结论(实验 ① 改为解析判定)**:
- DEP4-Pro 实测尝试三次均 OOM(Pro 832GB 权重 tp4 = 208GB/卡,图捕获期差 1-2GB;KV 0.32 + 捕获集裁到 5 尺寸仍不够)——Pro 在 4 卡上不可行,与 SGLang 侧 Pro 崩溃互为印证:**跨栈同模型比较做不到,模型差异是本对照的固有 caveat**。
- 但 regime 问题不需要 DEP4:**per-GPU 负载对齐的格子已经存在**。他们 poetry512(tp4/dp4:每 GPU 128 请求、~128k token MoE 负载)↔ 我们 poetry1024(DEP8:每 GPU 同样 128 请求、~128k token)。同等每 GPU 压力:**他们 +7.7% vs 我们 −1.6%**。→ **差距主体是实现侧**(图形状经济学 + 机制税),不是负载 regime;残余混杂仅剩 EP 宽度(EP8 vs EP4 的 a2a)与模型(Flash vs Pro 的 accept 分布)。
- 优先级更新:#19 连续预算原型 + 均匀 cap 模式评估升为首位修复路线;DEP4 复测取消。

### 【08-09 01:30】SGLang 深挖:兑换率数学闭合,第一嫌疑修正为**批碎片化**

双方各自的成本表 × 各自实测的裁剪深度/accept 代价:

| | 步时 | accept | 预测净 | 实测 |
|---|---|---|---|---|
| SGLang poetry512(128/rank 真实步)| 44.9→35.4ms(−21%)| −9.7% | +14.7% | +7.7% |
| 我们 @128 行/步(假想满行)| 92.2→77.2ms(−16%)| −9.8% | **+7.7%** | — |
| 我们 @64 行/步(实际形态)| 60.3→55.3ms(−8%)| −9.8% | **−1.6%** | **−1.6%** |

- **我们自己的 v2b 表预测:真 128 行/步下裁剪净赚 +7.7%**(与 SGLang 实测同值)。但 poetry1024 的 padded_bs 直方图 = {64: 2364, 72: 443, **128: 仅 134**}/4448 步——**执行器在 1024-global 下多数步只装 64 行**;64 行 regime 地板占 60%,预测 −1.6% = 实测 −1.6%,逐位闭合。
- **结论**:我们的裁剪机制、决策、成本表全都没大错——**是执行器的批装配把每一步滑进了裁剪必亏的 regime**。SGLang 同压力下每步稳定 126-170 行(decode 日志)。修好碎片化,poetry1024 按自家表应翻到 ~+7.7%,毋须改裁剪本身。
- **新任务(第一优先)**:定位 1024-global 下 rank 步只有 ~64 行的机制——候选:serve 准入节流、调度器容量交互、DP 路由不均、ragged fit 大批回退。判据:修复后 padded_bs 直方图模式移到 128,poetry1024 转正。
- 次级(降级但成立):64/rank 低压场景救不回(地板 60%,SGLang 也是靠 128/rank 赚);我们 θ 有平台他们没有(tp4 大形状);地板 29.5 vs 13ms(层数归一后仍 ~1.6×);机制税。SGLang 画像:高 accept 的 arena 他们也只打平(cap 4.01,−0.3%/+2.9%);超出 graph 覆盖(>192/rank)收益缩水到 +2.6%。

### 【08-09 01:35】cap-accept 判决(#20):被裁 token 只有 6.2%/16.1% 本会被接受——决策层无罪

cap-accept 模式(同一 planner 决策,提交完整块全量验证、接受端封顶,逐请求精确计数反事实)跑同负载:

| | poetry | arena |
|---|---|---|
| 被裁位置数/请求/步 | 2.08 | 0.71 |
| 其中本会被接受 | **0.130(6.2%)** | **0.114(16.1%)** |
| 占无裁剪接受量 | 9.8% | 4.3% |
| trim_regret_rate | 22.7% | 34.3% |

- **决策层定案无罪**:poetry 被裁位置 93.8% 反正会被拒,置信筛选有效;arena 上 planner 少裁(0.71 位)且被裁位置存活率高 2.6 倍——行为与风险结构一致。~10% 的 accept 代价与 compact 臂吞吐分解、SGLang 对照(9.7%)三方吻合。
- **责任链闭合**:决策层 ✅(本节)→ 经济层 ✅(v2b 表:真 128 行/步预测 +7.7%)→ **执行层 ❌(批碎片化,64 行/步 → 地板 60% → −1.6%)**。修复目标唯一:批装配。
- 更正记录:SGLang profiler 支持双模式(`--fracs` 给定 → 2D `SpsAdditiveCostTable`,离对角 frac 扫描,与我们 v2b 同构且为 #26 的方法来源;省略 → 对角 static → 1D `SpsCostTable`)。#28 与本次跨栈对照用的均为一维模式(manifest fracs=null)——他们拿一维粗表仍 +7.7%,进一步支持"表精度非胜负手"。

### 【08-09 03:00-08:00】同台对照终判(parity):裁剪收益 = 慢执行的补偿;我们的栈无肉可裁

用户指令的实验:**同模型(Flash-DSpark)、同 4 卡 B300、同 per-rank 负载、同客户端,各自用各自的 profiler 采表**,对照 lbs128/256 下两栈行为。四节点七臂(双方 notrim/trim/cap-accept + 双表)。

**① 绝对吞吐(同一切条件,notrim)**:

| cell | TRT-LLM | SGLang | 比值 |
|---|---|---|---|
| poetry 512 | 20,840 | 8,794 | 2.4× |
| poetry 1024 (lbs256) | 21,838 | 6,324 | **3.5×** |
| arena 512 | 26,279 | 7,941 | 3.3× |
| arena 1024 | 20,041 | 6,397 | 3.1× |

**② 裁剪收益**:SGLang compact +5.9%(poetry512)/ +0.2~+0.8%(其余);TRT:**profiler 判自家新表 inert(argmax 在 bs32-256×接受率 0.3-0.95 全网格上零次选择裁剪)拒绝落盘**;强制按 SGLang 深度(frac 0.4)裁 → **−12~−22% 全线亏损**,表的判决被实证。

**③ 成本结构(双方 profiler 同台采集,机理定案)**:TRT 步时 ≈ 行数的函数——bs128 时 L=2 与 L=5 步时同为 39.2ms(token 翻倍零增长),fixed 20.1ms + α∝行数,**θ≈0,地板 93-95%**;SGLang 步时 ≈ token 的函数(13→45ms 随 M 涨 3.5×,θ 份额 ~70%)。**同一块 GPU、同一个模型,成本结构完全相反——动态裁剪是 token 成本主导执行的补偿机制,在行数成本主导的执行上无标的**。

**④ 决策/排序质量(cap-accept 对称测量)**:被裁位置存活率 TRT 15.2%(1.65 位/请求,裸 sigmoid,125 万请求)≈ SGLang 11.4-14.5%(差分估计)≈ Pro+STS 6.2-16.1%(昨日精确计数)——**双方筛子同级,排序不是胜负手**。

**⑤ 工程包线**:SGLang decode 图上限实测 192 行(256 行 cublas 捕获崩溃、profiler 自我拒绝),Pro 模型在其栈图捕获崩溃;其每 rank 在飞请求恒饱和 ~110-115(与需求无关)。TRT 256 行图一次捕获成功,notrim 吞吐 512→1024 global 仍在上升。

**⑥ 附带修正**:SGLang profiler 亦有二维 additive 表能力(`--fracs` 离对角扫描,#26 方法之源),canonical/blog 流程用一维 static 模式(两次对照均如此);其 STS 机制存在但 canonical 不启用(本对照两侧均未用)。

**产品线结论**:B300 + 高效执行(本仓)+ agg 场景下,**动态 verify 裁剪没有经济基础**——不是决策不准(④)、不是表不准(③已双向验证)、而是没有可回收的时间。#19(连续预算)在 θ≈0 下亦无标的。**保留价值**:①(2.4-3.5×)与 lbs256 扩展性本身;#31(Pro@DEP8 批碎片化)仍值得修——Pro 的 θ 份额(40-53%)非零,v2b 预测真 128 行步 +7.7% 未被 Flash 判决否定,须修完碎片化后用 Pro 复测定夺;#25 device 序幕的机制正确性已证(census/GSM8K),其激活条件同上。

### 【08-09 08:50-10:15】iter-log 地面真值:θ 平坦结论被推翻,真凶 = 短窗桶 kernel 病灶(#32)

用户质疑 profiler 可信度(stats 开销污染?pin 是否生效?)→ 重构为纯 `print_iter_log` 的五阶段隔离矩阵(Flash DEP4,逐迭代 `host_step_time`/`prev_device_step_time`/`num_generation_tokens` 地面真值,单 serve 上 pin 换档零重启):

| 阶段(同负载)| tok/行 | ~128 行 device ms | ~256 行 device ms |
|---|---|---|---|
| U6 notrim(满块均匀图)| 6.30 | 46.4 | 72.3 |
| R5 pin 满窗(ragged 机制,同 token)| 6.30 | 47.9 | 79.8 |
| R5+`enable_iter_perf_stats` | 6.30 | 48.0 | 81.6 |
| **R3 pin 窗口4(token −33%)** | 4.20 | **53.7** | **107.0** |
| R2 pin 窗口3(token −50%)| 3.15 | 50.3 | 102.8 |
| FRAC 0.4(自然抖动)| 4.20 | 53.7 | 102.4 |

**分解全部闭合**:① pin 真实生效(token 计数逐档精确变化)→ profiler 钉窗机制无罪;② `enable_iter_perf_stats` 税 0-2% → 污染假设基本排除(但采表默认关掉它,已列方法论改进);③ ragged 机制税小(+1.5ms@128 行 / +7.5ms@256 行);④ 桶抖动 ≈ 0(FRAC ≡ pin4);⑤ **真凶:短窗桶形状本身**——token 砍 1/3,步时反涨 **+12%/+34%**,且非单调(窗口 4 比 3 更糟)。**θ 在 serving 真值下不是 ~0,是负的。**

- **跨证据链**:GB300 Pro frac 表的 w=4 异常(bs64: 62.7 > w5 的 60.8)、我方 Flash profiler 的 42.09 离群点(bs128 L=4)、SGLang #28 bs128 裁剪后 iter 28→33ms——同一病灶跨模型、跨节点、跨栈复现。嫌疑:attention/indexer/DeepGEMM 在非满宽 token 形状下的 tile/算法选型退化。已立 **#32**,下一步 nsys 对拍 pin5 vs pin3 单步 kernel 时间线点名退化 kernel。
- **对既有结论的修订**:昨夜"θ≈0 → 裁剪无标的"修正为"**满窗形状高度优化、短窗形状病态劣化 → 当前裁剪不但无收益还倒贴**";profiler 引擎模式的平坦行是合成负载(input_len 1024)遮蔽了病灶,采表流程需改用真实负载形态或 serving 日志建表(`--from-iter-log`)。
- **产品线含义**:#31(碎片化)+ **#32(短窗 kernel)** 是两把钥匙——#32 修复后 θ 恢复正斜率,Pro 的悬崖交换率(GB300 表:w5→w3 省 34% 步时/40% token)将**优于**当前一切估计;裁剪经济学的最终判决推迟到两者修复后的 Pro 复测。

### 备选修复方案评估(供评审;当前实施的是 A,标记为暂定)

- **A(已实施,Python-only)**:捕获拷贝换 D2D 源(`_seq_lens_cuda`)+ NEXT_N 抬全局上界。最小爆炸半径,当场可在活挂载容器里验证;把双胞胎缓冲降级为 canonical 缓冲的衍生物。host 路径装填时刻数值逐位等价,uniform/eager 分支未触碰。
- **B(否)**:序幕直写该缓冲 + 把 H2D 挪出捕获区。可行但延续"平行缓冲靠写集纪律同步"的模式——本 bug 正是写集漏项造成,管理双胞胎不如消灭双胞胎。
- **C(架构终态,需 C++ 重编译,建议后续立项)**:改 kernel 删掉 `new_tokens_per_seq` 输入,`nn` 由 kernel 内从 `cu_seq_lens[b+1]−cu_seq_lens[b]`(或 `sp` 直接取图内重算的 `cached_tokens`)导出——冗余输入消失,永不可能再失同步。与 A 不冲突,A 先落地验证机制,C 作 PR 收尾清理候选。
- **D(否)**:钳制序幕 w ≤ bucket tier。功能倒退且不修主 bug。
- **E(否)**:仅 device-windows 模式走 D2D。零收益模式分叉,制造未来漂移面。
- NEXT_N 抬全局上界同时排掉一个 host 侧潜伏雷:#19(取消 tier 阶梯量化)实施后 host 也可能在小 tier bucket 发超 tier 窗口,同样会截断 Phase 1。

### #32 nsys 取证:四次报告丢失的根因与 v6 修正(08-09 00:20)

四轮 nsys 采集(SIGINT 版 / TERM 版 / 手动收尾版 / v5 stop-shutdown 版)全部丢失报告,根因**不是收尾方式,而是采集从未开始**:v5 用 `--capture-range-end=stop-shutdown` 时应用理应在 iter 2050 被自动关停,实测 pin3 应用活过 iter 2110 → `cudaProfilerStart` 触发器从未被 nsys 看见。

- **代码侧验证(py_executor.py)**:触发门只有 `iter_counter ∈ profile_start_iters and not is_warmup`(L1739-1748),打印的 `iter = N` 与触发比较用的是**同一个计数器**(L1725),三条执行循环(pp/普通/overlap)都套着 `profile_step`——TRT-LLM 侧路径无问题,且触发时会打 `"Profiling started at iteration N."` 日志(此后可用该行判别触发与注入哪个失效)。
- **真凶(官方配方对照,perf-analysis.md)**:`trtllm-serve` 的 worker 进程是 **fork-before-exec** 派生的,nsys 缺 `--trace-fork-before-exec=true` 时不会向 fork 出的子进程注入 → worker 里的 `cudaProfilerStart` 对 nsys 不可见,capture 永远不开始,报告永远不落盘。nsys 退出时抱怨的 "re-parented processes" 即其指纹。
- **v6 修正**:`--trace-fork-before-exec=true` + `--cuda-graph-trace node`(后者让 CUDA graph 内的 kernel 逐个展开——我们全部计算都在图里,做 pin5 vs pin3 kernel 对拍必须要 node 粒度)。两轮已重发(pin5@743961 / pin3@743962)。

**v6 判决(08-09 22:10)**:fork 追踪使触发器**开火了**(两轮 serve log 均出现 `"Profiling started at iteration 2000."`,这是 TRT-LLM 侧打的,证明 env 到位、门条件满足),但 nsys 退出时自供 `Processing events... → No reports were generated`——**容器内 nsys 2026.2.1 仍未注入到 worker 进程**,cudaProfilerStart 调了但没人接。此前"没看到 Profiling stopped"是预期行为(该日志只在 torch trace 开启时打,L1666-1671)。另收获:stop-shutdown 关不掉 re-parented worker,`--wait=primary` 让 nsys 干等,需手动 TERM app(留 nsys)才收尾。

**v7(执行中)——采用同事已验证配方**:peihengh 的 dsv4 nsys 战役(同集群同拓扑 B300 + trtllm-llmapi-launch + trtllm-serve)成功出过 `.nsys-rep`,其 start_worker.sh 注释直言 *"Lingjie's hang-avoidance: NSYS_BIN=extracted 2025.5.1"*——与用户"nsys 得用 2025.5.x"的提示互相印证:**容器自带 2026.2.1 有问题,须用 lustre 上提取版 2025.5.1.121**。v7 变更:① nsys 换 `peihengh/dsv4_pipe_clean_20260723/tools/nsys-2025.5.1.121`(容器内烟测可运行);② `--capture-range-end=stop`(cudaProfilerStop 时就地落盘,app 继续跑,绕开 shutdown 死结);③ 照抄其余旗标(`-t cuda,nvtx --cuda-graph-trace node --gpu-metrics-devices=none`);④ **窗口对准**:pin3 v6 的窗口正好砸进 burst 排水缝(iter 2000-2025 只有 1 个请求;pin5 幸运落在稳态 120 请求)——burst 跨度 ≈ max_tokens/accept 与批大小无关(pin5 ~470/pin3 ~515 iter,确定性强),v7 round 先放校准 burst 实测跨度,再用小 filler burst(64 req × 按缺口折算的 max_tokens)把 2000-2050 顶进整 burst 的稳态中段。

### #32 nsys 取证 v7:报告到手 + 初步 kernel 对拍(08-09 22:50)

**五连丢终结**:v7(提取版 nsys 2025.5.1.121 + `--capture-range-end=stop` + 窗口对准)两轮均落盘 27MB 报告(`nsys/pin5_rank0.nsys-rep` / `pin3_rank0.nsys-rep`),窗口 iter 2000-2050,graph 内 kernel node 粒度。对准偏差:两窗都落在 burst 的 ~65% 位置(排水段),pin5 占用 72→23 请求、pin3 57→6——pin3 没够到 padded_bs=128 档,已再发 v7b 两轮(对准收紧到 burst 20-35% 段)补清洁样本。

**混形状粗对拍(初步,待 v7b 清洁窗验证)**:
| 信号 | pin5(6 tok/req) | pin3(4 tok/req) | 读法 |
|---|---|---|---|
| MoE FC1 大 tile `t128x256x256`(swiGlu) | 17.0%,中位 178µs | 3.2%(几乎退场) | 大桶专属,效率最高 |
| MoE FC1 中 tile `t128x128x256u2` | 1.7% | **8.0%,中位 110µs** | 小桶主力:token 减半,GEMM 只省 38% |
| MoE FC1 小 tile `t128x64x256u2` | 0.1%,中位 69µs | **5.7%,中位 94µs(m3/m5=1.35)** | 同名 kernel 在 pin3 上下文里慢 35% |
| FMHA sparse / A2A / topK / 元素级 | — | 比值 0.6-0.8 | 正常随 token 缩放 |

**初步指认**:短窗桶的病灶集中在 **MoE grouped GEMM 的 tile 阶梯**——token 桶变小 → 换小 tile → 每 token 效率近乎腰斩,GEMM 时间不随 token 线性下降,形成 #32 的"常数惩罚";与 GB300 Pro frac 表的平台+悬崖楼梯(MoE 更重 → 惩罚更大)跨模型同构。另见 pin3 独有的 FMHA 变体(1.0%)待查。**定论等 v7b 的 (128,768) vs (128,384) 同批次干净对拍。**

### #32 定论:短窗惩罚 = `_prepare_inputs` 的 host 同步链,不是 kernel 病态(08-09 23:40,v7b 清洁窗)

v7b 两轮对准命中:pin5b(122±3 请求,桶 768)/ pin3b(122±2 请求,桶 512),窗口 iter 2000-2050 全程均质满批,**同 padded_bs=128 的干净对拍**。步时(窗内 `prev_device_step_time` 均值):pin5b **51.39ms**,pin3b **58.25ms**——token −33%,步时 **+13.3%**,在 nsys 在场下复现 iter-log 矩阵的短窗惩罚。

**三层分解(每 50-iter 窗口,rank0):**

| 层 | pin5b | pin3b | 判读 |
|---|---|---|---|
| GPU kernel 总时间 | 2638.7ms | 2281.9ms(0.865) | kernel 变**快**了,不是 kernel 病态 |
| GPU 空转间隙(>merge)| 162ms(~1 个大隙/步)| **844ms(~7 个/步,单隙 6-8.7ms)** | 慢的全部来自空隙 |
| `_prepare_inputs` NVTX 相位 | 1253.7ms(25.1ms/步)| **1977.4ms(39.5ms/步)** | 惩罚源头相位 |
| 其中 cudaEventSynchronize | 100 次 / 0.9ms(免费)| **300 次 / 1078.2ms(21.6ms/步)** | **6 次/步阻塞 D2H,平均 3.6ms/次** |

**机理**:短窗(w<max)激活压缩器 ragged 元数据路径(`deepseek_v4.py` 的 `_compute_gen_compressed_position_ids` / `_compute_compressed_mask` 等,inductor 编成 `triton_poi_fused_arange_clamp_index_lt_searchsorted_*`);其数据依赖形状迫使调用方按 compress_ratio 把 0-dim CUDA 张量读回 host(代码注释自供:`gen_output_offsets` "pre-extracted by the caller to avoid .item() inside compiled code"、`ctx_output_sizes` "0-dim-CUDA fallback costs two implicit D2H syncs per ratio")。每步 6 次阻塞读 → 排空 launch 流水线 → GPU 干等 host 跑 python(空隙期间 host 不在任何 CUDA API 里)→ 每步净损 ~14ms。满窗(w=max)时形状与静态捕获一致走快路径,2 次/步事件同步且零耗时——这就是为什么 iter-log 矩阵里"ragged 机制税"在满窗下只有 +1.5ms,而 w4/w3 一压就 +12%/+34%。

**修订与含义**:
- 首轮(排水窗)对拍的"MoE tile 换挡"故事在同 bs 下**不成立**(大 tile 两侧都是 2300 实例、0.94)——那是批大小混杂的假象,撤回。
- kernel 家族缩放(次要但真实):MoE grouped GEMM 0.94(权重搬运主导,token 不敏感)、nvjet 1.00、elementwise 0.88、FMHA 0.74、topK 0.69——Flash 的 θ_model 本来就小,**该省的 33% token 本来也只值 ~13% kernel 时间**;但 prep 同步链把它反噬成 +13% 步时。
- **裁剪经济学翻案路径**:修掉 prep 同步链(方向:元数据形状改由 host 端窗口计划推导,或直接用 padded 均匀形状——文档注释明言 reserved slots 会在 postprocess 被 mask,按桶静态化本就是容忍的设计),短窗步时应回到 ≤ 满窗水平,届时 Pro@DEP8 复测才是 trim 产品线的真判决。#11(ragged 路径无额外同步验证)就此有了答案:**有,6 次/步,已点名**。
- 跨栈注记:GB300 Pro 的 w4 异常应为同一 prep 路径(同代码);SGLang 的 28→33ms 是另一栈,不并案。

### #32 修复落地并验证:双缓冲 ping-pong 暂存(08-10 00:15)

**同步点最终点名**:6 次/步 = 基线 2 次 + `_pinned_host` 的 4 个短窗 key(`accept_caps` / `ragged_verify_lens` / `gather_rows` / `gather_cols`,model_engine.py)。其 WAR 守卫在复用暂存缓冲前 `evt.synchronize()` 等**上一步**从同一缓冲发出的 H2D 被流消费——overlap 模式下即把第 N+1 步的 prepare 串行到第 N 步的流位置上。

**修复(b87ea28bf3,已提交)**:每 key 双槽 ping-pong——守卫改等**两步前**同槽的拷贝(稳态必已完成,等待≈0);WAR 保证原样保留(槽自身上次 H2D 完成后才重写),与 confidence relay 两深环同一设计语言,23 行。

**验证(122±1 请求/rank,DEP4 Flash,无 nsys,窗口 iter 2000-2050 全稳态)**:

| 臂 | 修复前 | 修复后 | 判读 |
|---|---|---|---|
| pin5(满窗,对照)| 47.9ms(iter-log 矩阵)| **48.55 ± 0.54ms** | 无回归 ✓ |
| pin3(4 tok/req,桶 512)| 53.7ms | **42.20 ± 0.76ms(−21%)** | **短窗首次快过满窗** |
| 步时比 42.20/48.55 | — | **0.869** | 与 nsys kernel 总时间比 0.865 重合:步时回归纯 GPU 决定 |

**含义:θ 由负转正**——w5→w3 现在省 13% 步时(此前反贴 12%)。此前所有"裁剪无收益/负收益"的吞吐结论(强裁臂 −12~−22%、host 动态 ≈0)都建立在这条被串行化拖垮的短窗路径上,需要在修复后重测;Pro@DEP8 复测(#31 修复后)仍是产品线最终判决。单测:hw_agnostic 套件 DSpark 相关 296/296 通过(`test_user_provided.py` 的 33 个失败为 llama 权重环境性失败,与本补丁无关;测试需 `--mpi=pmix` 起容器,否则 mpi4py 初始化即中止)。

**FRAC0.4 混合窗验证(00:25)**:**42.23 ± 0.78ms @ 121 请求**——与 pin3 uniform(42.20)逐位一致,真 ragged 混合窗(gather_rows/gather_cols 全覆盖)同样 −21%。修复前该臂 53.7ms。**三臂验证闭合,#32 关闭。** 下一步:修复后的三臂吞吐重测(scheduled vs off vs forced,Flash DEP4)——此前"裁剪无收益"的结论全部失效待重验。

### #33 修复后吞吐重测(08-10 02:00,首批;两节点臂序对调,poetry cell 全齐 n=5)

强裁臂(FORCE_BUDGET_FRAC=0.4)vs 满窗基线,Flash DEP4,修复 b87ea28bf3 之后:

| 载荷 | 全局 bs | frac04 中位 | notrim 中位 | Δ | 修复前同臂 |
|---|---|---|---|---|---|
| poetry | 512 | 21,369 | 20,292 | **+5.3%** | −12~−22% |
| poetry | 1024 | 23,540 | 21,876 | **+7.6%** | 同上 |
| arena | 512 | 25,854 | 25,863 | −0.0% | — |
| arena | 1024 | 20,197 | 19,903 | +1.5%(终值,n=10) | — |

**判读**:①裁剪的步时节省首次真实转化为吞吐收益(poetry +5~8%,与 SGLang 原生动态在 B300 的 +5.9% 同量级);②两节点交叉复现(poetry 各 cell 双节点中位差 <2%);③arena ≈ 0 提示机械 40% 裁剪在多样负载上把接受损失顶回去了——**这正是置信度调度(按请求裁)该赢过机械裁的场景**;④scheduled 臂待重采表:旧表(v2b/parity)嵌入了修复前的序列化惩罚,会误导 planner 拒绝裁剪,须在修复后代码上重采(server-sweep pin 阶梯 + --from-iter-log)。

### 修复后成本表(08-10 02:45,--from-iter-log,今晚全部 serving 日志)

`postfix_flash_table.json`(bs 32-128;bs 1/8 排水 cell 因网格连通性剔除):**θ(M) 单调且有真实质量**——θ(96)=3.1 → θ(512)=19.6 → θ(768)=25.9ms;固定 18.4ms + α(32/64/128)=0/1.3/3.9ms。验算与实测严丝合缝:bs128 满窗预测 48.2(实测 48.55)、bs128 L3 预测 41.9(实测 42.20)。**bs128 步时的 54% 是 token 比例项**(修复前 parity 表 θ 全程 0.9-3.1ms"惰性"、serving 真值为负)——置信度调度首次拥有真实杠杆。方法论注:旧表(v2b/parity/steep)全部作废,嵌入了 #32 的序列化惩罚。

scheduled 臂(新表 + tiers 2-5)已在两节点开跑,基线 notrim(今晚同节点 n=10)可直接配对。

### #33 终局:三臂全齐,动态裁剪在我方栈上首次全面转正(08-10 03:20)

Flash DEP4,修复 b87ea28bf3 + 新表 `postfix_flash_table.json`,每 cell n=10(双节点臂序对调):

| 载荷 | 全局 bs | notrim | frac04 | **scheduled** | frac Δ | **sched Δ** |
|---|---|---|---|---|---|---|
| poetry | 512 | 20,292 | 21,369 | 21,502 | +5.3% | **+6.0%** |
| poetry | 1024 | 21,876 | 23,540 | 23,114 | +7.6% | **+5.7%** |
| arena | 512 | 25,863 | 25,854 | 26,125 | −0.0% | **+1.0%** |
| arena | 1024 | 19,903 | 20,197 | 20,222 | +1.5% | **+1.6%** |

**判决**:①scheduled 四个 cell 全部为正,无一倒贴——修复前(host 动态 ≈0±3%、强裁 −12~−22%)的"无收益"结论整体翻案,根因就是 #32 的 WAR 守卫串行化;②在机械裁剪打平的 arena 上,置信度调度仍 +1.0~+1.6%——"按请求该裁才裁"的选择性兑现;③poetry +5.7~+6.0% 与 SGLang 博客量级(H200 +10-20%、其 B300 原生复现 +5.9%)对齐;④master arc "定位为什么没有收益"就此闭环:收益一直存在,被我们自己 4 个暂存 key 的同步链吃掉了。

**余留**:Pro@DEP8 复测(依赖 #31 碎片化修复)= 产品线最终判决;#19(连续预算)在新表 θ 单调后价值上升;GSM8K 精度回归(新表只改窗口选择不改接受判定,风险低,例行补测)。

### #31 碎片化:复现 + 排除链 + 工作假设(08-10 05:50,进行中)

**复现(Pro@DEP8 单 8-GPU 节点,poetry 1024 全局,notrim + iter log)**:rank0 每步调度请求数众数 **60-62**(上尾 75,偶发 65 个 iter@128),远低于公平份额 128;累计分配 639/5120 ≈ 完全均衡 → **路由公平,并发准入被卡 ~60/rank**。一个 1024-burst 要 ~4480 iter 才消化完(等效并发 ~60)。

**排除链**:① KV 容量——`free_gpu_memory_fraction` 0.5→0.85 后最终配额 35.4→60.1GiB(定容管线无恙;注意 v2 有两条 quota 日志,第一条是估算期临时缓存,勿误读),直方图不变;KV 需求(~830 token/req × 8438 B/token)远低于配额,排除。② 客户端连接池——nsys_burst2 的 `TCPConnector(limit=600)` 曾是干扰源(600/8=75 恰是上尾!),提到 1300 后众数仍 60-62,排除(**但注意:此前所有用该客户端打 >600 全局并发的实验都被客户端限流污染,幸而 p25fix_bench 是 1100,bs1024 吞吐数据勉强幸免**)。③ ADP per-rank cap(nvbug-6133201 派生上限)——serve log 无 capping 警告,4096/6=682 不咬,排除。

**工作假设(判别中)**:调度器步 token 预算按 **KV block 粒度**给 gen 请求计费——Pro block=64 → 4096/64 = **64 请求/步**(众数 60-62 恰在其下,且与最初"64 行步"观察逐位吻合);Flash block=32 → 4096/32 = 128(DEP4 从未撞上,v7b 实测 122 通行)。判别:Pro `max_num_tokens` 4096→8192,若调度跳 ~128 即定案;修复方向 = gen 请求按实际步 token(6)计费,或部署侧提高 max_num_tokens。

**⚠ 实验作废公告(08-10 06:35)**:上述 KV-fraction 排除与 max_num_tokens=8192 判别**全部作废**——744499 是 2 节点作业,而各轮的 serve/客户端/pkill srun(-N1)随机落节点,导致**僵尸 serve 跨轮累积**(清点:bia0038 上 12 个、bia0042 上 2 个 trtllm 进程),burst 常打到旧 serve(带统计客户端坐实:1024 全 OK 而新 serve 日志 0 流量)。可信的仅有首轮复现(60-62 众数,fresh serve)。已全节点清理 + 所有 srun 钉死 `-w bia0038`,判别链从头重跑。教训入册:**多节点分配上做单节点实验,每个 srun 必须显式 -w**。

**干净判决链(08-10 08:30,单节点钉死后全部重跑)**:
- 复现坐实:1024 全局真实在飞(带统计客户端 1024 全 OK 且与服务端 200 计数一致),rank0 持有满份额 128 活跃请求(`currank_total_requests = 128/1024`),ctx 在前 3 个 iter 内全部完成(准入一次到位),此后**每步只调度 60-63 个**,kv_cache_util 仅 0.014。
- 干净排除:① KV 配额(35.4 vs 60.1GiB,直方图不变);② max_num_tokens(4096 vs 8192,不变)→ block 粒度计费假设也随之排除;③ 客户端;④ 准入速率。
- 新观测:步时 ~290ms @ 62 行(缺健康基线,eager 嫌疑未证);SWA 固定成本 8.1GiB = max_batch_size(128) × 63.6MB/请求,定容公式按满批预留(sizing 侧无恙)。
- 进行中的二分:v1 KV 管理器(`use_kv_cache_manager_v2: false`)vs python 调度器(`use_python_scheduler: true`)各跑一轮——若 v1 恢复 128 → 病灶在 v2 管理器的调度集成;若 python 调度器恢复 → 病灶在 C++ 容量调度器与 v2 的适配。


## Step 3 复现性判决(08-07 12:40,校准 + v2b 表条件下)

1. **host 未复现**:初测 +8.3%(poetry 128/rank)在四轮中缩为 [0.0, +1.9, +1.3, +2.2]%;arena 侧 [−2.4, 0.0, −1.4, −5.6]%。host 整体 ≈ 0±3%,poetry 微正、arena 微负。host 调度行为本身跨节点完全可复现(trim 22.4-22.7%,accept 1.66-1.67,四轮逐位一致)——**是收益不存在,不是实验不稳**。
2. **device "+63.9%" 判定为假象**:device 臂生成损坏(见 Debug 日志 12:30 条),吞吐虚高由重复特殊 token + 零 EOS + 满批不排水制造。本轮 device 吞吐(+50%~+55%)全部作废。**H3 判 (c):接受判定/停机记账损坏**,旧 accept_stat 2.03 即其指纹。
3. 待办:device 路径根因定位与修复(736879/736880 debug 节点已排队);修复后 device 收益需重测才有结论。H2/H4 挂起,等 device 修复。
4. 联动结论:#28(SGLang 原生动态在 B300 高 bs 亏损)+ 本轮 host≈0 → 细刀 tier + 校准 + 修表后,**host 动态裁剪在 B300 agg 上没有可复现的吞吐收益**;与 SGLang blog 的 +10-20%(H200/H100)相比,B300 的 θ 平坦区更宽,裁剪节省的 token 成本换不回接受长度损失。
