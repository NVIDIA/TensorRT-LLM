# Target: deepseek-r1-0528-nvfp4 / sm_103 / dep4

## Identity

| | |
|---|---|
| Checkpoint | DeepSeek-R1-0528, modelopt NVFP4 export (**v1**, `DeepSeek-R1-0528-FP4`): 61 layers, hidden 7168, 128 query heads, `q_lora_rank` 1536, vocab 129280, untied embeddings; MLA attention in bf16; layers 0-2 dense (intermediate 18432), layers 3-60 MoE with 256 routed experts at top-8 (group-limited: `n_group` 8, `topk_group` 4) plus one shared expert (intermediate 2048); YaRN rope, factor 40 over an original 4096-position window; **NVFP4 MLP weights with an fp8-e4m3 KV cache** (`hf_quant_config.json`: `quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`), attention / router / embedding / lm_head bf16. The checkpoint also ships a bf16 MTP module at layer 61 (`num_nextn_predict_layers: 1`), which the identity config does not load and `configs/mtp{1,2,3}.yaml` do |
| GPU arch | sm_103 (GB300) |
| Parallel | dep4 — `tensor_parallel_size: 4` + `moe_expert_parallel_size: 4` + `enable_attention_dp: true`; world size 4, one rank per GPU. The engine builds `tp_size=4`, `moe_ep_size=4`, `moe_tp_size=1`, `pp_size=1`, `enable_attention_dp=True`, and construction asserts every one of those |
| Registered class | `StaircaseDeepseekR10528Nvfp4Sm103Dep4` — a synthetic architecture name no checkpoint declares. `models/deepseek_v3/routing.py` rewrites `DeepseekV3ForCausalLM` into it when the config shape, SM and topology all match; the checkpoint is read unpatched. Per-target names mean one process can hold every target at once |

> **NO GATE RECORD HOLDS FOR THIS TARGET.** Every result below — boot,
> gsm8k, the MTP acceptance comparison, the whole *Performance* section —
> was measured on **sm_100 (B200)** through the pre-move standalone harness.
> This target is **sm_103 (GB300)**, and certification is per architecture.
> The numbers are kept as provenance, but this target is **ungated** until
> they are reproduced on GB300. Read every "passed" below as "passed, on
> sm_100, before the move". The MTP variant needs all three of its gates
> repeated, acceptance included — see *The MTP variant*.

Checkpoint sha256 — the checkpoint directory passed to `--model` should
resolve to files with these digests. **Routing does not check them**: it
fingerprints the config's shape
(`num_hidden_layers`, `hidden_size`, `n_routed_experts`, `q_lora_rank`), so
a fine-tune or a re-export of this checkpoint routes here silently. That is
the deliberate trade in `models/deepseek_v3/routing.py`, and it changes what
a gate record means — not "this target passed" but "this modeling code
passed *on the checkpoint with these digests*". Run it on another one and
the result is ungated (recorded at:
`umbriel-b200-027:/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4`).
The source `config.json` this stub was patched from hashes
`a80de10ccf70e7e98dcc6730b45872298937b232d3e8eb0321cfd2bce4cb40e0`; it is
the one checkpoint file that is copied rather than linked.

<details>
<summary>169 linked files (163 safetensors shards + index + 5 aux)</summary>

```
c139e0cd9aa4c418ebd38ebae9aff73b4eaacc7723e25d2490dfe41c02bb8708  configuration_deepseek.py
0ef9febae6b6087f4822b02bc9a1c03a83263dabaa931fb9155d061c8951ca07  generation_config.json
36dc07886afcd1679ebf5328a325f9f9629d7a0ddfa644ee6427b10874b2c295  hf_quant_config.json
991aaf2f2c7a5b21f6c6708bd736285640ebec92790d7882958dcc26b0c9f6f6  model.safetensors.index.json
ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d  tokenizer.json
a58700120f68f96faf27a8921876e5e3dbb95f66355c6b331312354cbbf792f9  tokenizer_config.json
bbf04b49f4cdb3c2cb139d84e9701089cff14d08b95b969379aa5b57a63a9242  model-00001-of-000163.safetensors
ac6cb6c4a30d18ee8d24c5237c1b0f9cb30ab24aa9f4e4758a8beb5125360720  model-00002-of-000163.safetensors
c280f0fc3241c685b033c26c92fd4c3fca34285f288922e3d8119ba7d779aa32  model-00003-of-000163.safetensors
ca5f80f8d62248366271495d0a9a95953dd5fb06237231a3f8465207d475edd9  model-00004-of-000163.safetensors
5a454d4577381ec2fb9511015c9a06024fe592949badd45d36d2efe24dc1a0f2  model-00005-of-000163.safetensors
9bb8873103ba89dd16efad52e7846abe7a0ee40af47d4739f3503a82160b1001  model-00006-of-000163.safetensors
48a5a4d26db3e4a11d1bacdfd5c94e27bfc90b14750c165ee7ebca2d81b924e4  model-00007-of-000163.safetensors
b52d09d1147f1704bb39096221afbd992c6a1fe1114358975ddca79ea214361e  model-00008-of-000163.safetensors
0f0274233de53fe22c50262c36d6154579af3cc29b83d3b2d601761b0475c810  model-00009-of-000163.safetensors
2658cef0619d4912da43964d312fa1448eac38f5e44e0313218367b888f8e7ae  model-00010-of-000163.safetensors
360189a2c32647c16efbf465a9f2bb3f607e47f0848c6dbfca1eae455d8fa6dd  model-00011-of-000163.safetensors
8ae7554ea1a6d65c9e194dd99254bc0797c9fd848a7be23fca94e2defc56ce9d  model-00012-of-000163.safetensors
dc6ed60abb1be1223979884405aad63f35e5eec42c28b73773bd19f834c90065  model-00013-of-000163.safetensors
7d359af3e3ccf70794d83235e5d022392cb76ade1a34d4fd8797c588c212056a  model-00014-of-000163.safetensors
afbdb1f5da08dc38ecaf900ae8866e4687a3ec9e2b21b16d0cba7fa3d529b74a  model-00015-of-000163.safetensors
b0ed219a52ec2d617ecc750963519df23d5e27160621d2fd964b506ba017a5bc  model-00016-of-000163.safetensors
bbe5cb864267a27a55add115d27f9fdec6d47405f626de9774e0f79ed271de8e  model-00017-of-000163.safetensors
ca0e49de986c19b866bf2c330ca91f8c0a092ee56e19ce832267f375174cc496  model-00018-of-000163.safetensors
0b1db3ad16aea8afdcc667011dabc11addf41c20b4c10234508117578278ba21  model-00019-of-000163.safetensors
c016097de09ca54826380158d531ef7121ccc6f5aa44dbbe255dd4c80e817e7d  model-00020-of-000163.safetensors
4929c6acd11fa3f87465aef13e5ce7e3d24c83a1babac6d1c99c0201bba7985b  model-00021-of-000163.safetensors
05f55fd3fa47fcebc0c4910204672fb602b30f364cf088544c6ec1d67f7d33ba  model-00022-of-000163.safetensors
8f861d34dfd9299c043ba0c351505bef3a107d8f95ea8a46ceda8a3ebf6f2625  model-00023-of-000163.safetensors
95362bd84e97a23460933f630b69d9ff720b7d1da15c544b97ade3481bcca444  model-00024-of-000163.safetensors
c1db194b1d6a0f2c6a1935e5105622174007db387c79e6c1b5430bc3273c778b  model-00025-of-000163.safetensors
d181f80f2aaf9cb7bdf68e421b950d64a583a52420ce5fd0d3b411ed33c3ea66  model-00026-of-000163.safetensors
04ee66ce0f3e51f407d5a68e4b29c0b63e9bda0a5e88896c39a097aaba3a2dd2  model-00027-of-000163.safetensors
9167d91670023079e629de55d02be50a06a720b125da70b9556481b9f23e49d3  model-00028-of-000163.safetensors
ecf62e772486af899afb440540803550480343b66e5dc096e1e4a8dea6a44154  model-00029-of-000163.safetensors
18dfa7d3ed08561b4109e83e49fe4a073ce5b1a23d374bce98dc5bab832f4948  model-00030-of-000163.safetensors
70e471763de9f5ad8b12cc26be6f44d48c8ebcbb7f1c4a57f4eaabf112f7f57f  model-00031-of-000163.safetensors
fd86c19b1c61cd171d5f573cd62a85d09baae34b54b678ceed921c522412a406  model-00032-of-000163.safetensors
e4a2457ca0b979830ba5bdff1b9e17be8ec227c866a8adeeb455abc87a780761  model-00033-of-000163.safetensors
a69fad78c8fce390011d98b140789010e2fcb83129e7ca17ce1bce64bfd2b759  model-00034-of-000163.safetensors
a3898dfd91f59bec65c82ae04b707575501e32f43d2f1dfb26174912f01d0fd5  model-00035-of-000163.safetensors
2e0fe82025efc32f27a290438b2e607c89ac394716e97b3c9b6cfa27e2480623  model-00036-of-000163.safetensors
77afb4330256683ef7962bf43cae8762f80fb0edf889be55073f421c3ad24930  model-00037-of-000163.safetensors
8e2498c311a925d54212cee7d41ce4f98c345d788adcddcc749053292511e4bd  model-00038-of-000163.safetensors
0ee6089eb7bfef59dde3b154b1e133f0034347b391b7ed3c903d6e6525b20ea9  model-00039-of-000163.safetensors
54bee64d4e307768da9a6bf437e00eec91e3a4986a2358d2d5eb41d915f4dd0e  model-00040-of-000163.safetensors
e8fc96ae38097b90d1b4424f7610186ff7102cfa67930823a5500906f0398b7f  model-00041-of-000163.safetensors
2adea35e960e5e35f95c62c0bc97d883248aa79d186da815535336fb0ea914c3  model-00042-of-000163.safetensors
bad155fdce361b40b328edbe35e3995cf6e149e1f798eba415db58ff998880fb  model-00043-of-000163.safetensors
17e353f93d66c07708586a8374012a8cb68939a831931bb215508553adcdbeb4  model-00044-of-000163.safetensors
215c3ee090030d9f6a12cac238e1cdc14c1bf7d33a45eb5a84114986a65e988c  model-00045-of-000163.safetensors
ba7377f46d611140338dd6efde6ffbb47fce763a70ffe2d0746d5175ad799eea  model-00046-of-000163.safetensors
d1ba05c9c00dffcfe86e08ff0109d83d68694179ef45dbcc3766c32ab0316922  model-00047-of-000163.safetensors
171a534fa7797a90cfcfc5b340a25013359750ef37f637b89d7ce5665b13d191  model-00048-of-000163.safetensors
d4c05d603481809dd5a6910461e08ed2387c6b7459438e93f201d00c23222690  model-00049-of-000163.safetensors
558e86f0977914e39665e2a4756082cc5a9ea1d2d99ef4a863c69adc84febf66  model-00050-of-000163.safetensors
5e07cadc4d02a7783e0ea6edfeeff88bbcf02b78679ad4efe32592c54d154208  model-00051-of-000163.safetensors
83c7f0c70e1bb1b0ff6e0915f40d1ca91d98c7c63aebeae50265fa214b8005ad  model-00052-of-000163.safetensors
96fa3cc110f476e1414000da8431ddac767c694143a20f07cfb14de62bdbe419  model-00053-of-000163.safetensors
7b170e72c34df97980ed4988a261f1732f2b573244f3345456fefde9874609bb  model-00054-of-000163.safetensors
a5b7b9001f1ad50039e63989eb90be1f1e5d9dc0654544a87073627b73289eae  model-00055-of-000163.safetensors
98d711b4214ef67598ac60887c12b1f2186e7aa3ec97d95858389c799f8b46f7  model-00056-of-000163.safetensors
3ec5916a49938b634be79548cbe6944660fdcacbdb255a6393932fd404e796ee  model-00057-of-000163.safetensors
0d60c7c3f78cd237164c0ed466bed5e73e53328a1db0605aabfa430920210003  model-00058-of-000163.safetensors
0eb7d1c45c8444dbcfc152b44280357543fc1b798f9dc38cd7cb555204f55d99  model-00059-of-000163.safetensors
92bc2a13fd190ecbc25dfab34ceb210026a6674b984609e5642c6c152bb3c6ba  model-00060-of-000163.safetensors
4d00fd183118a24cbd5ebb357238505fc252b117e0cbe4e469605639d10b0b3f  model-00061-of-000163.safetensors
14ac6babc7b5ec03944a0253aa3276d114e07e36221d797fca50292ab4f371da  model-00062-of-000163.safetensors
5d3a9d7c49fe341f8b7f5e21c3ae63a0718f1a3227b18dac393e2565d3154e09  model-00063-of-000163.safetensors
49a1ddf499f4129111ffb50edac498fc23312f98ebc1b69e8841e853ac609355  model-00064-of-000163.safetensors
5341f93faa5c992d684f234c3a479aab87eb3b4576b1f978eb119476131a8ef2  model-00065-of-000163.safetensors
72d183baf3c7c2c7b816e286867e4f1e8398685b6c76110f778dbda4d85ce6fc  model-00066-of-000163.safetensors
5ec23f22dcae3d5d883dc925ce70d463a6c9bba3c1f9b5150d447ccfee451e77  model-00067-of-000163.safetensors
205d88536e4a52d83ebaa62005665d531e6f0f8440c280772e7ae61cad23d89f  model-00068-of-000163.safetensors
dde4b037231af225ed529148ce14e6dcf10ea3bcb5f79c47aa5c7fde35770d9e  model-00069-of-000163.safetensors
7f7ee508718627657ef87966a885a9956f3734674cb988fcb0bbb9e22de005d1  model-00070-of-000163.safetensors
1824edecf1bbce9246038693adf729d0484a7148aff8ce7c4c2f302184874cf1  model-00071-of-000163.safetensors
633dcca915de5aa3629f2b53345a86c5eb4b4bcd4731aa3fbf5e3b5ee4c25bef  model-00072-of-000163.safetensors
b53bc434459993f27ce202bd913640f64129c6d91cea6445fddaf6869ea3e618  model-00073-of-000163.safetensors
86680218bc3719271bc7c47f74810b469fbdeb7cb4370433121cc2c8e1416dca  model-00074-of-000163.safetensors
0eac1ff5b435996ca17677429909307081eced38c7aa8543f54569e509a18b7c  model-00075-of-000163.safetensors
6b49ec351aa02e4d152f95d2d7fb25fefd537a48ab12941ef1ede7b8f2879601  model-00076-of-000163.safetensors
36d338fba2c4a1543a691c72ed2297aad49c46230219c3e570c41a30b6430671  model-00077-of-000163.safetensors
1611afd72d0e702e4edd0e519525f0fd70763039b18db539d1d1a465d1262732  model-00078-of-000163.safetensors
c2cbd1c7ac0762ae69a453fdca414ff3b41d6a831e36f33a0e365eeb3a4f7796  model-00079-of-000163.safetensors
5968523bd66250a4250e41fa01fd1868d558efc4df271d17e44d8a3931bb9f4f  model-00080-of-000163.safetensors
9f1bf3ae7b459e3471ead763cacddaf77b030c1bd153ae7c6c330fbe9309a75a  model-00081-of-000163.safetensors
b6e47522fcbcd15ee2d4c8a7921954d48403b3daaf45d1fe0b49f3ab6e7c7bd7  model-00082-of-000163.safetensors
a641e0c5932548bc1f93d94bdcf435898656123298425c04c5cfa7f17fc756d3  model-00083-of-000163.safetensors
f43e8dbbfbdbf808a11d8c42668aed50855b668e0f3c1a3cbd007e6b03652e78  model-00084-of-000163.safetensors
894fe387005e9b73fbe8483c5f4948f127527fd90dcf07bfb1dbdedf74357b3e  model-00085-of-000163.safetensors
217e2a040495a5a3825c308638806b750508d23ce78f996d53459d83c848d50e  model-00086-of-000163.safetensors
9eb75ba7b09bffab0a4d45fe739ab339db47352c227ca0ee15beed80cc1e7da5  model-00087-of-000163.safetensors
f4497ac9f44495704458aa0542395dc29abf97d8a3abbc16e36b4ef522068725  model-00088-of-000163.safetensors
1cde75047119b80d9bb27333d9d52340b56c5c3879cb326f7418dd589dfd73d8  model-00089-of-000163.safetensors
e3b9df1600cabbb33a0963412a8a919fdeb75472938a46346fb9207876940da9  model-00090-of-000163.safetensors
2aa0a431652ac12273e3a0d38870baf9810f5d748be16e1ca37cccf1381a817f  model-00091-of-000163.safetensors
a1c78001c5e2ac6ae5a8212e87440091dd56142c0f8829da3c463e0f717725bb  model-00092-of-000163.safetensors
9bec57d3138b23e9c9e5dc53e9b6ac63adfb27e35584232b88ab4bcea0653dfd  model-00093-of-000163.safetensors
b1bcc0b137739cacff96dd6c2fe0861af0102244be4865ddd50ff9fe71bc5c0a  model-00094-of-000163.safetensors
76c031439199fdba5c01bc8ad571a0f8655c3ce1a04bb466338c25ca5256cd6c  model-00095-of-000163.safetensors
2f10fce8ee1187960f53cf5bbc5750d4033688f0e6bb24fb92447dd1bffb9ac4  model-00096-of-000163.safetensors
ceebab7ff4022aaa21e15ff05e6a3ebd4a334e10049bf721d3e2519b98514737  model-00097-of-000163.safetensors
87ef9077d9212db15cb8fe89342501c9cea08f3c5c96a31235db683d8751221f  model-00098-of-000163.safetensors
cac29cbf8e08520c89bcf27a08a77343a5f6c2aea1771ee7a1bbdf64474e90d1  model-00099-of-000163.safetensors
5ee6438bc3414412a7ab279b3e0b72d631245a7e27975a895d0e94f5aeef90d8  model-00100-of-000163.safetensors
d99c26d1874005fee504b44ac96c496a496ff25cb87ff755832e926541a48492  model-00101-of-000163.safetensors
dff5cc926dc3c1684fbec174508b5ba5abc67c65e1acf6f20354c3acce74b05a  model-00102-of-000163.safetensors
babcf9affea89d453f2023379a34b32d0ecc4f9b2787255c0fcb8889cabc093a  model-00103-of-000163.safetensors
45e2f2dae475a13c6d10d7a2bfde28b8d65765d421e73d08c527d8c1f5b9392a  model-00104-of-000163.safetensors
1b276c9186ee2f335e0930840d406a59776d695433081bbabf4c5ba25e79bd05  model-00105-of-000163.safetensors
a9a5527aede2cb7f292671d86f2864d6fc843396466d16f60e74743d7fc0a6ee  model-00106-of-000163.safetensors
a950aedf442134db181e4ac440c9007178c36b0ef57245f7f49a3ee5bf50540f  model-00107-of-000163.safetensors
e7a7bf9ae8f6ea8ecc8750cd44ccb3814eafbb2f0b88fc7482e6ee39e9d8dc1b  model-00108-of-000163.safetensors
cb9ac1d529250b0524bd86864725144d45d5e91c70e77375ff0bf30434633250  model-00109-of-000163.safetensors
a6999a13f265c55995ba5c1e0260d64469dc714949a4df300b37af6ae9a1e61a  model-00110-of-000163.safetensors
7ada750ccc74bcc4c283332ce7ca573c74254ed61931737b1d739e9b633d0e78  model-00111-of-000163.safetensors
3fbfd40aa863cb1ae17eb175958b95e168325c1cae9c8affe639d88c09e1960a  model-00112-of-000163.safetensors
61dea7ce4e66bf886e2ede5ff2a9df65660b7492eb820116b4bac9d7419b1e85  model-00113-of-000163.safetensors
12fa6385c3932d28de5300199e53aadacc8a0d3638d03c42f5cbac71f2d8574b  model-00114-of-000163.safetensors
22e0a03279a7d2ebe6b0bfb6d5e2ce98b5da39f518cb9a82c04b4d5b52096fa1  model-00115-of-000163.safetensors
c1c9fc85b7cd9d403a2b73098e1d81a10e12fdffd9549c669a2ae515d851052a  model-00116-of-000163.safetensors
4dc24b7fcbc6517c62222ca7da18bc664d5f1da5e2efcc8bcb8d5fd590652b78  model-00117-of-000163.safetensors
794d6d03b6447e7f31882ab070bbcede362becd464139cbcca5970d2f41f12ab  model-00118-of-000163.safetensors
dae3a9aceb361372feef019635c17a1bda7c826dba8cacbcb3881ea7de2139db  model-00119-of-000163.safetensors
93d8435bb4af72314030be201f0f5e8f431fe6e48f79e7918e46d7ef92f2cb59  model-00120-of-000163.safetensors
9aa69352d752be9438a99776732b4e0b62ff589151bcf8d8cfb5e9dfecc45a0a  model-00121-of-000163.safetensors
8f7bc0398f6417d9d60567b0687406a951a6190bc8df46be675b2a90c69aa4cf  model-00122-of-000163.safetensors
1aa442507c1ee8a26afef5950aa87e6834724231f0c2852b642254c3afb79eaf  model-00123-of-000163.safetensors
6e4185179e96b94dbe61de9db3fc68386f524bd5119ef5025c67608f852479be  model-00124-of-000163.safetensors
e03dae8708f37980fea4471fe9cf88d9ac703fc2db7b128985fe759cd9922fa8  model-00125-of-000163.safetensors
7c910c972cbfa6f8afcd6cd55e97a90688429037e429b0c67b048fb48bff53c7  model-00126-of-000163.safetensors
bae19ab02d0fb95a32fa08867741e437592c9857e39b4cdb9279da627948a647  model-00127-of-000163.safetensors
fa3c946d0c051bfd44077f497971dc21b93750a3412f541bc34b742a0b504398  model-00128-of-000163.safetensors
8d1afa3ef67315e898c3fcca58e522b278c52ee6db14e3beac8d3f3da9aace23  model-00129-of-000163.safetensors
4a0db8ab07b13372e598790ccb5222e0ebc345e514575169c98ce30bf1b54f8d  model-00130-of-000163.safetensors
95e59b883e43b7259e2f061c5ae47071076f8d9ea26d025d12e86c3979f15007  model-00131-of-000163.safetensors
e7310c0a55bd9126ae844d8b1c9074c476d5ff31b9d8902f27114dbde16fce23  model-00132-of-000163.safetensors
184d0abd9be8e5b966cbbf17264b204872450c0fae4c29a82083e39e273432cb  model-00133-of-000163.safetensors
4f0d6beb59421c0bc1560495dc89324de1f2e7a794b65ffb854534a21b17b50a  model-00134-of-000163.safetensors
ec9fc4a523c9aabe818356a72a8f032540ebd22273a03976bfa8f9af11a5d8b6  model-00135-of-000163.safetensors
9862e7589f2288d86e45d9377203935621b982a0e376101996196fdfc4fd8cad  model-00136-of-000163.safetensors
395a89c3c93071004c8d73810f73c23954449acc03efbe220a6b4cf576afdb60  model-00137-of-000163.safetensors
a747eb61204cb7d1f78acfcae182c75781a7839ca185132a0421acd72668eafc  model-00138-of-000163.safetensors
27651e88d2fbfc582e0de685ec318c6e9af6e8ad7c13233a2bb3c33ea14ca634  model-00139-of-000163.safetensors
eaa5f027070aed1ad02c4295218c99f3c91c1ad067c4bf60f56ec69219fe0949  model-00140-of-000163.safetensors
6c924b6d4b1fc90bd0aa18a9421125a6c54d317516fe58ad8b8d07d08f12c7b7  model-00141-of-000163.safetensors
4e3cd496cb1af08ea801b831982573883b158bb60a1a9f3576f3986a4cff3467  model-00142-of-000163.safetensors
37716bff9e48a87fd4801e644527b2fd31468c4f6866bda70ff9db409902c593  model-00143-of-000163.safetensors
760f7034122d6ef1738bb74fcf4c9230e5a7ee9024b499931e05760c66aff3ac  model-00144-of-000163.safetensors
fa1203747f79017c0324b5564ea70dd611b237928054dbe3a8f9cf4217cea66a  model-00145-of-000163.safetensors
c05235067f189ade00c6d0ce7faf2ba728012ed78863352cc9d613da2cf959ba  model-00146-of-000163.safetensors
248b214fbd8455664ca1183cbc8f1ccd8f6fc3a9e3b2898798b5b1a716225811  model-00147-of-000163.safetensors
738de1a2b1844982c6f29317e508a9c8b57c218dd4aac058ca113e90f265ba96  model-00148-of-000163.safetensors
c35cd109b7895cf7a74b6f561f47145a99daccdead56fce6b256a0914aceaf25  model-00149-of-000163.safetensors
a08febee45f832a99e4029bc73c0093a40b4f3d7ef2f1690088982ba65fbe00c  model-00150-of-000163.safetensors
eb659a6d3d6dcce0c0a051ea5b2184fc7c190c6037859d197637abd6ee305f6c  model-00151-of-000163.safetensors
d19f2846145b89a456e71ef2e6e3f00a4ede37c29f8e4ba6cef55159a3230922  model-00152-of-000163.safetensors
801cc06d203ea5e4cbe9097622e7b971174249c5b1b0298b3560cb831946dfb6  model-00153-of-000163.safetensors
42e9ce66a658f447f16c3bca1cb9d843a66df989201ef7fe2696138309d2ef25  model-00154-of-000163.safetensors
ed61b343d9f632a15d1b8b98a103e7ca9077a8e57c92c798f82472d4f7e9776c  model-00155-of-000163.safetensors
8d2689d216da51e5a3d507a9a1eec347200fc1146a9c84ebd1e30d9d93ec779a  model-00156-of-000163.safetensors
4413b79d84ec94cb627440e096b36012184cce599151355691213348b6e4b795  model-00157-of-000163.safetensors
fe09a5c54db66630ceea47091df108d8235736abe679700786bcf7294f0e9cbc  model-00158-of-000163.safetensors
79b6b4bb774569c433cf9b4ac391b4fae11a6ebcb235b213e4c23b2b66d3c477  model-00159-of-000163.safetensors
243cb503cc88e16170bad7a216742da5262032bc640730fc0b3e348a7b14f451  model-00160-of-000163.safetensors
4981af366b2bedb6ec8d6cd2eeae8bd2c027c3ccaf001ab54af207883da0cdfd  model-00161-of-000163.safetensors
783d3e45422f562d9dc7b11495dc7a44834f0cfab59e0235432d1188170565f2  model-00162-of-000163.safetensors
8ad1e6011ac0ebd811cbf00edb62ad095be42b01266cf4638492fecc1e9020fa  model-00163-of-000163.safetensors
```

</details>

`model_dir/config.json` is the checkpoint's own config with two deliberate
divergences: `architectures` patched to `["StaircaseForCausalLM"]`, and a
`dtype: "bfloat16"` field added beside the checkpoint's `torch_dtype`
(transformers 5.x renamed the field, and the shell materializes its own
`lm_head` from whichever surface resolves — both were observed present and
`torch.bfloat16` on this build). `hf_quant_config.json` is linked because the
engine reads it: it sets `quant_algo=NVFP4` **and**
`kv_cache_quant_algo=FP8`, and the target asserts the latter — the fp8 latent
pool is what selects `quant_mode` on every MLA call, and a checkpoint
declaring no KV quantization is a different assembly. Nothing in the stub
carries the topology; that lives in `llm_args.yaml`.

## Version

| | |
|---|---|
| tensorrt_llm | in-tree — the target moves with the trunk, so there is no version to pin and none is asserted. What *is* asserted at construction is the SM version (`_SM = (10, 3)`), which the pin used to stand in for. The gate records below name the commit they were taken at |
| torch | 2.11.0+cu130 |
| transformers | 5.5.4 (the config surface the engine hands the target) |
| Attention metadata fact source | `TrtllmAttentionMetadata` (TRTLLM backend) |

## Vocabulary

**Two audit roots.** This target carries two forward paths: the trunk's, which
runs on every engine step, and the MTP layer's, which runs `max_draft_len`
times per step under a `configs/mtp*.yaml` variant and not at all without one.
Auditing only the first would leave the second free to use any op unseen.

### Trunk — the `StaircaseCore.forward` closure

`forward` plus the private methods it reaches: `_dp_rows`, `_dense_mlp`,
`_check_step_contract`, `_build_step_args`, `_moe_chunk_sizes`, and — through
one first-forward branch only — `_rope_tables`.

| call | catalog entry |
|---|---|
| `embedding` | `torch/embedding.py` |
| `flashinfer_rmsnorm` | `norm/flashinfer_rmsnorm.py` |
| `flashinfer_fused_add_rmsnorm` | `norm/flashinfer_fused_add_rmsnorm.py` |
| `cublas_mm` | `gemm/cublas_mm.py` |
| `bmm_out` | `gemm/bmm_out.py` |
| `mla_rope_append_paged_kv_assign_q` | `attention/mla_rope_append_paged_kv_assign_q.py` |
| `load_paged_kv_cache_for_mla` | `attention/load_paged_kv_cache_for_mla.py` |
| `mla_rope_generation` | `attention/mla_rope_generation.py` |
| `thop_attention` | `attention/thop_attention.py` |
| `allgather` | `comm/allgather.py` |
| `reducescatter` | `comm/reducescatter.py` |
| `fp4_quantize` | `quantization/fp4_quantize.py` |
| `nvfp4_gemm` | `gemm/nvfp4_gemm.py` |
| `flashinfer_silu_and_mul` | `activation/flashinfer_silu_and_mul.py` |
| `noaux_tc_op` | `moe/noaux_tc_op.py` |
| `fp4_block_scale_moe_runner` | `moe/fp4_block_scale_moe_runner.py` |
| `empty`, `reshape`, `split`, `concat`, `copy_`, `expand`, `transpose`, `view_dtype`, `add`, `pad` | `torch/*.py` |

`_rope_tables` enters the closure through exactly one branch, taken on the
first forward and never again: when the engine's admitted `max_seq_len`
exceeds the config's `max_position_embeddings` — which a `speculative_config`
causes, measured **163840 -> 163848** at `max_draft_len: 3` — the constant rope
table is rebuilt at the larger row count. Its `torch.arange` / `.cos()` /
`.sin()` / `torch.empty` are host-side table construction, not step math: they
create no activation, run once before any CUDA-graph capture, and every table
row depends only on its own position, so the rows the identity config uses are
bit-identical whether the table was built at 163840 or 163848. Recorded here
rather than left for a closure scan to trip over.

Everything else in the closure is a builtin or a metadata read (`getattr`,
`hasattr`, `isinstance`, `int`, `bool`, `all`, `len`, `max`, `range`,
`sorted`, `divmod`, `dict`, `kwargs.get`, `list.append`,
`host_kv_cache_pool_mapping.tolist()`).

### MTP layer — the `MTPLayer.forward` closure, plus `.shared_head`

`forward` plus `_shared_mlp`, `_routed_experts`, `_mtp_dp_rows`,
`_build_step_args`, `_moe_chunk_sizes`; and `shared_head`, which the runtime
calls separately for the draft logits.

| call | catalog entry |
|---|---|
| `embedding` | `torch/embedding.py` |
| `flashinfer_rmsnorm` | `norm/flashinfer_rmsnorm.py` |
| `flashinfer_fused_add_rmsnorm` | `norm/flashinfer_fused_add_rmsnorm.py` |
| `cublas_mm` | `gemm/cublas_mm.py` |
| `bmm_out` | `gemm/bmm_out.py` |
| `mla_rope_append_paged_kv_assign_q` | `attention/mla_rope_append_paged_kv_assign_q.py` |
| `load_paged_kv_cache_for_mla` | `attention/load_paged_kv_cache_for_mla.py` |
| `mla_rope_generation` | `attention/mla_rope_generation.py` |
| `thop_attention` | `attention/thop_attention.py` |
| `allgather` | `comm/allgather.py` |
| `reducescatter` | `comm/reducescatter.py` |
| `flashinfer_silu_and_mul` | `activation/flashinfer_silu_and_mul.py` |
| `noaux_tc_op` | `moe/noaux_tc_op.py` |
| `fused_moe` | `moe/fused_moe.py` |
| `empty`, `reshape`, `split`, `concat`, `copy_`, `expand`, `transpose`, `add`, `pad` | `torch/*.py` |

The two tables differ in exactly one place, and it is a dtype consequence: the
checkpoint excludes `model.layers.61*` from NVFP4 wholesale, so this module is
bf16 throughout. It therefore uses **`moe/fused_moe.py`**, the unquantized
grouped-expert runner, where the trunk uses `fp4_block_scale_moe_runner` — and
correspondingly the trunk's `fp4_quantize`, `nvfp4_gemm` and `view_dtype` do
not appear here at all. Everything else is the same vocabulary.

`shared_head` adds `flashinfer_rmsnorm` and one `logits_processor.forward`.

### The shell's speculative branch

`StaircaseForCausalLM.forward` is a plain delegation to the inherited base
when there is no spec worker. With one it contains exactly one tensor
expression — the row gather of the trunk's hidden states at
`spec_metadata.gather_ids`, which is **`torch/embedding.py`**
(`torch.nn.functional.embedding` is a row lookup, used here as one) — plus a
`logits_processor.forward`. That projection is the inherited shell's own
logits path, the same one the non-speculative forward runs internally, so it
is runtime rather than modeling; it is named here rather than left implicit.

**No catalog entry was added by this target**, torch mirror included. The
trunk's entry set is exactly the `deepseek-v3-lite-nvfp4/sm_100/dep4`
sibling's; the MTP increment consumed one further **existing** entry,
`moe/fused_moe.py`, which the catalog owner certified at this checkpoint's MTP
routed geometry rather than this target adding anything.

Load time (outside the closed-vocabulary rule, `weights.py`):
`torch.ops.trtllm.block_scale_interleave` for the 128x4 scale swizzle, plus
`torch.cat` / `torch.index_select` for the `[up; gate]` concat and the
interleave + 32-row block shuffle of the expert stacks, and the `kv_b_proj`
row regrouping. The MTP module adds only plain `torch.cat` (its `[up; gate]`
FC1 stack and its shared-expert `[gate; up]` pair) — no swizzle, no shuffle,
because none of it is quantized. `derive_after_load` builds the YaRN rope
table, the `.t()` GEMM views, the two MLA absorption operands, every NVFP4
call scalar, and the expert-window slices of the three MoE scale scalars — and
asserts the checkpoint's fp8 KV scales are all exactly 1.0 (122 tensors, 124
with the MTP module loaded).

Audit is mechanical: collect the calls in each root and the private methods it
reaches, then match against `catalog/index.yaml`.

## Verification

**The gate records below are the identity config's.** A `configs/mtp*.yaml`
variant is a different forward and a different weight load, so nothing here
speaks for it; its own records are in *The MTP variant* at the end of this
section.

Both gates run under the target's identity config (`llm_args.yaml` =
`tensor_parallel_size: 4` + `moe_expert_parallel_size: 4` +
`enable_attention_dp: true`, everything else trtllm defaults — block reuse
and CUDA graphs on, page size 32), 2026-08-01, **GPUs 3-6** of
`umbriel-b200-027` (8x NVIDIA B200, driver 595.58.03, 224-core),
`source scripts/env.sh` then `CUDA_VISIBLE_DEVICES=3,4,5,6`.

### Required on sm_103 — not yet run

Every row needs 4 GB300 GPUs and `trtllm-llmapi-launch` over a 4-task srun
allocation. `CFG=` this target's `configs/` directory: each file there is a
complete `--extra_llm_api_options`, carrying the dep4 topology that selects
this target plus the knobs of its own variant.

Records below that speak of "smoke" were measured with a per-target
`smoke.py` — a bespoke CLI that asserted a keyword in each of ten greedy
continuations. It was removed once this package moved in-tree: the generic
script in the `boot` row below starts the engine the same way and prints
the same continuations, and
unlike a module nothing ran, the accuracy rows below are wired into CI.

| Gate | Command | Result |
|---|---|---|
| boot | `TRTLLM_STAIRCASE=require trtllm-llmapi-launch python examples/llm-api/quickstart_advanced.py --model_dir <ckpt> --tp_size 4 --moe_ep_size 4 --enable_attention_dp --max_tokens 16 --prompt "The capital of France is" "The chemical symbol for gold is" "1, 2, 3, 4, 5,"` | **passed, 10/10** greedy keyword asserts, 2026-09-10, 4x GB300 on nvl72d199-T07, trtllm 1.3.0rc26, 5m18s. Run through `trtllm-llmapi-launch` over a 4-task srun allocation; the engine built `tensor_parallel_size=4`, `moe_expert_parallel_size=4`, `enable_attention_dp=True` with `TRTLLM_STAIRCASE=require` exported |

**The identity assembly and the MTP variant are both gated on sm_103.** Everything up to the
weight load is driven by the checkpoint's config alone, and that part
was exercised on a GB300 against the config *as published* (no
target-owned stub): routing resolved this target, the module imported,
`StaircaseCore.__init__` passed every geometry, topology and dtype
assert, and 1980 parameters declared, 61 layers, hidden 7168. `lm_head.weight`
came out **bfloat16**, which is the specific thing removing the stub
put at risk -- the shell sizes it from the pretrained dtype, and a
regression there materializes fp32 two layers from its cause.

The weight path is verified by the two gates above, over four ranks:
the manifest load fills every declared parameter, the post-load
derivations run, and the forward is exercised through prefill, the
CUDA-graph decode path, and the expert-parallel MoE round trip.

The checkpoint they ran on is the one this file records. Every digest
that can be checked was re-verified after download -- `config.json`,
`generation_config.json`, `hf_quant_config.json`, `tokenizer_config.json`
and `tokenizer.json` by hand, the 163 safetensors shards by git-lfs
whose object id *is* the sha256 -- so these records and the sm_100 ones
below were measured on byte-identical weights.

`configs/mtp3.yaml` carries its own three gate records above -- it selects a
second forward path *and* a second weight-loading path, so the identity
records do not speak for it. `mtp1.yaml` and `mtp2.yaml` remain **ungated**;
they are the dominated low end of the measured draft-length axis, and
`mtp3.yaml` is the one to serve.

| boot, MTP | as above plus `--spec_decode_algo MTP --spec_decode_max_draft_len 3 --kv_cache_fraction 0.75`, i.e. what `$CFG/mtp3.yaml` declares | **passed, 10/10**, 2026-09-10, 4x GB300, 6m16s. `MTPDecodingConfig(max_draft_len=3)` in the run's LLM Args and 124.65 GiB of weights loaded against the identity path's ~118.8 GiB -- the layer-61 bf16 MTP module, i.e. the variant's second weight-loading path, really ran |
| gsm8k full | `TRTLLM_STAIRCASE=require trtllm-llmapi-launch trtllm-eval --model <ckpt> --extra_llm_api_options $CFG/identity.yaml gsm8k --output_path <dir>` | **passed, 95.0720** (`exact_match,flexible-extract`, +-0.5962, full 1319 questions) against threshold **89.9962** (anchor `deepseek-ai/DeepSeek-R1-0528` = 94.9962, tol 5.0) -- pass by 5.08 points. 2026-09-10, 4x GB300, 6m47s. `strict-match` on the same run: 94.7688 |
| gsm8k, **paired in one session** | `$CFG/identity.yaml` and `$CFG/mtp3.yaml`, back to back in one allocation; in CI as `accuracy/test_staircase.py::TestStaircaseDeepseekR10528Nvfp4Sm103Dep4::test_gsm8k_identity_vs_mtp3` | **passed.** identity **94.9962** (+-0.6005) / strict 94.7688; mtp3 **94.6171** (+-0.6216) / strict 94.4655. **delta = -0.3791 flexible, -0.3033 strict** against a `|delta| < 1.2` criterion (2 sigma at sigma = 0.60) -- 0.63 sigma, five questions of 1319, and both filters move the same way |
| acceptance vs stock | `trtllm-bench throughput` twice in one allocation over one fixed-seed dataset, `TRTLLM_STAIRCASE=require` against `=off`; in CI as the two `accuracy/test_staircase.py::TestStaircaseDeepseekR10528Nvfp4Sm103Dep4::test_mtp3_acceptance[...]` legs | **passed.** `acceptance_length` **3.3514** (ours) against **3.2752** (stock) -- ratio **1.023**; draft acceptance 78.38% against 75.84%. 2026-09-10, 4x GB300, 64 requests at ISL=OSL=1024, concurrency 32 |

`TRTLLM_STAIRCASE=require` is what makes these gates at all: under `auto` a
configuration that missed this target would measure the built-in DeepseekV3
implementation and report it as this target's score. The old `--trtllm` flag
of the perf harness, which worked by unsetting an environment variable, is
now just `staircase: off` — one config key selecting between the two
systems.



#### The MTP variant is gated, and why the third gate was the one that mattered

Rejection sampling holds the emitted distribution to the target model's, so a
*miscomputed* draft layer produces correct text more slowly rather than wrong
text: the boot gate passes, accuracy passes, and only speed moves. `acceptance_length`
is the sole detector, and it is only readable against a reference -- 1.0 would
mean every draft rejected, and "clearly above 1.0 but below the reference"
would mean subtly wrong.

Measured **3.3514** against stock's **3.2752** on the same 64-request
fixed-seed dataset in the same allocation, a ratio of **1.023**, against a
ceiling of 4.0 at `max_draft_len: 3`. The sm_100 record below measured ratios
of 1.0215 and 1.0175 at concurrency 1 and 32, so this sits in the same family.
The draft path computes what the checkpoint says.

Two things about the comparison worth stating rather than leaving implicit.
`TRTLLM_CAN_USE_DEEP_EP=0` was exported for **both** sides: stock cannot boot
this checkpoint at dep4 with MTP without it (the MoE communication factory
lands on DeepEPLowLatency, whose dispatch takes only NVFP4 uint8 hidden states,
and the MTP layer is bf16), and it is inert for the staircase target, which
implements the all-gather/reduce-scatter round trip by hand rather than
through that factory. So it makes the two systems more comparable, not less.
And the throughput on the same runs -- 2955.2 against 2732.7 output tok/s -- is
**recorded, not claimed**: it is one concurrency of one synthetic workload, and
this project measures perf rather than gating it.

#### On the flexible-extract number landing exactly on the sm_100 one

This run measured **95.0720 (+-0.5962)**, and the sm_100 identity run
recorded below measured **95.0720 (+-0.5962)** -- the same 1254 of 1319
questions. That is agreement at the score level, not proof of identical
generations: `strict-match` differs on the same pair of runs (94.7688 here
against 94.9204 there), so the underlying text does move, as it should
across two architectures whose MoE epilogues use different block-scale
recipes. Read the flexible-extract match as a strong reproduction, and not
as evidence that the two forwards are bit-identical -- they are not.

### Prior record — sm_100 (B200), pre-move harness, does not gate this target

| Gate | Result |
|---|---|
| smoke — `uv run targets/deepseek-r1-0528-nvfp4/sm_100/dep4/smoke.py` | **passed, 10/10** greedy keyword asserts, first run, no iteration. Keywords were authored provisionally and frozen against the continuations this model actually produced |
| gsm8k full — `uv run bench/accuracy.py --target targets/deepseek-r1-0528-nvfp4/sm_100/dep4` | **passed, 94.9962** (`exact_match,flexible-extract`, 1319 questions, 5-shot completion, no chat template, 256 output tokens) against `reference: deepseek-ai/DeepSeek-R1-0528 = 94.24 (trtllm); gate: accuracy >= 89.24`. First run, stock defaults, no instrumentation; TRTLLM execution 37.695 s, engine init 197.663 s |

Both lm-eval filters of the passing run: `flexible-extract` **94.9962**
(±0.6005) and `strict-match` **94.6171** (±0.6216). The gap is 5 questions of
1319 — under few-shot completion this checkpoint mostly answers in the strict
`#### N` form, and flexible extraction finds a few more.

**Read the +0.76 over the anchor as agreement, not as an improvement.** The
anchor was measured on a **different NVFP4 export of the same base model**
(`DeepSeek-R1-0528-FP4-v2`, whose modelopt `exclude_modules` list differs
substantially from this v1 export's), which is what `tol: 5.0` absorbs; and
one filter's stderr alone is ±0.60. Nothing at this distance is a result in
either direction.

The reference entry is still at its external anchor (`source: trtllm`). Its
header asks for a write-back after a target's first passing run; `bench/` is
read-only for the assembler, so that edit is left to whoever owns the file.

### The MTP variant

`configs/mtp{1,2,3}.yaml`, 2026-08-02, same host and same **GPUs 3-6**.

| Gate | Result |
|---|---|
| identity smoke, re-run after the MTP increment — `uv run .../smoke.py` | **passed, 10/10**, and all 10 continuations **byte-identical** to the pre-increment run (`md5` of the captured case lines equal). This is the invariant the increment is held to: no `speculative_config`, no MTP parameter declared, `forward` a plain `super().forward(...)`, so the gsm8k 94.9962 record stands unmoved |
| MTP smoke — `uv run .../smoke.py --config .../configs/mtp3.yaml` | **passed, 10/10**. Engine built at `max_draft_len: 3`, `max_seq_len` 163848, KV pool 32.20 GiB (968,288 tokens, 62 layers), 12 fp8 MLA decode JIT compiles per rank |
| acceptance probe — `uv run bench/perf.py --target ... --label probe-mtp3-acceptance --config .../configs/mtp3.yaml --acceptance --concurrency 1,32 --rounds 2` | **`acceptance_length` 3.9274 at con=1 and 3.4635 at con=32**, against a ceiling of 4.0 at `max_draft_len: 3` and a break-even of 1.164. Draft acceptance rate 97.58% / 82.12% (5849 of 5994 and 39152 of 47679 draft tokens accepted) |

**Two of the ten MTP continuations differ in wording from the identity run's,
and that is expected rather than a defect.** Rejection sampling makes the
emitted token the *target* model's argmax, so a correct MTP layer cannot
change what is emitted — but the trunk's own generation call now runs four
query rows per sequence instead of one, on a different decode kernel
(`HVPerCta256` rather than `HVPerCta128`) with a different accumulation order.
Greedy decoding is chaotic under a 1-ulp logit difference at a near-tie: cases
2 and 3 diverge a few tokens in and stay grammatical and correct, and all ten
keywords pass. **Byte-identity is the right bar for the identity config and
the wrong one for a variant that changes the attention tile.**

### The MTP variant's own gate records

The two gates the increment is released on were run after the assembly, on
the same host and the same **GPUs 3-6**, 2026-08-02. Like everything else in
this file they are **sm_100 records and do not gate the sm_103 target**; the
acceptance row in particular has to be repeated, because it is the *only*
detector of a miscomputed draft path — rejection sampling keeps the emitted
distribution correct, so boot and accuracy both pass while a wrong draft
layer merely runs slower.

| Gate | Result |
|---|---|
| gsm8k, **paired in one session** — `uv run bench/accuracy.py --target ...` with and without `--config .../configs/mtp3.yaml` | **passed.** identity **95.0720** (±0.5962) / strict 94.9204; mtp3 **95.2237** (±0.5874) / strict 94.8446. **Δ = +0.1517 flexible, −0.0758 strict** against a `\|Δ\| < 1.2` criterion (2σ at σ = 0.6005) — 0.25σ, two questions of 1319, and the two filters move in opposite directions |
| acceptance vs stock trtllm — `uv run bench/perf.py --target ... --label trtllm-mtp3-acceptance --trtllm --config .../configs/trtllm-ref-mtp3.yaml --acceptance --concurrency 1,32 --rounds 2`, with `TRTLLM_CAN_USE_DEEP_EP=0` | **passed.** `acceptance_length` **3.9274 / 3.4635** (ours) against **3.8447 / 3.4040** (stock) at con 1 / 32 — ratios 1.0215 and 1.0175. Draft acceptance 97.58% / 82.12% against 94.82% / 80.13% |

**The paired accuracy run is the one to quote, not a cross-session
comparison.** The same identity code path measured 94.7688 on 2026-08-01 and
95.0720 on 2026-08-02 — a 0.30 spread on a bit-identical forward, which is
the scale any accuracy claim about this variant has to be read at. Both runs
of the pair were back to back on the same devices.

**One incidental number worth keeping**, because it is the only *real-text*
measurement of what MTP buys here: gsm8k execution time fell from **37.025 s
to 31.440 s, −15.1% (1.178x)**, on 1319 genuine prompts. It is not a Pareto
point — lm-eval drives its own concurrency rather than a swept one — so it
does not belong on the perf curve, but it is a far better answer to "is MTP
worth enabling" than the flattered random-prompt throughput. Engine init
rose 193.6 s → 278.4 s (layer 61's weights plus the extra decode JIT).

**Read the ~2% acceptance lead as agreement, not as an improvement.** The two
systems do not run the same knobs: the reference is forced to `max_num_tokens`
/ `max_seq_len` 2048 to boot at all and to a different MoE transport (below),
so scheduling, batching and block reuse differ and each system verifies a
slightly different token stream. What the comparison establishes is that this
target's MTP layer computes what the checkpoint says — a miscomputed one sits
at 1.0, a subtly miscomputed one clearly below the reference.

**Stock trtllm cannot serve this checkpoint at `dep4` with MTP under its
default MoE communication strategy**, and that is why the reference carries an
environment variable as well as a config. All four ranks die in
`Failed to initialize executor` on
`deep_ep_low_latency.py:238`'s `assert hidden_states.dtype == torch.uint8`:
the communication factory falls through `NVLinkOneSided` / `NVLinkTwoSided` /
`DeepEP` (each `not available: Invalid Argument`) to `DeepEPLowLatency`, whose
dispatch accepts only NVFP4 hidden states — and **the MTP layer is bf16**,
by the export's own `exclude_modules`. The existing `trtllm` Pareto curve went
through DeepEPLowLatency happily because without MTP every MoE layer is NVFP4.
`TRTLLM_CAN_USE_DEEP_EP=0` lands the reference on `AllGatherReduceScatter`,
which is the strategy *this target implements by hand*, so it makes the two
systems more numerically comparable rather than less.
`configs/trtllm-ref-mtp3.yaml` carries the whole ledger.

Consequently `trtllm-mtp3-acceptance` is an **acceptance measurement only**.
Its throughput column (180.6 tok/s/user at con=1, 1668.4 tok/s at con=32) is
not comparable to the `trtllm` Pareto curve: different boot knobs, a different
transport, and a draft length that curve does not carry.

### Why the accuracy gate alone could not have released this

Rejection sampling means a miscomputed draft layer produces correct text more
slowly, so an accuracy score cannot separate a good MTP layer from a broken
one — the Δ above would have been just as small with the `eh_proj` halves
swapped. `acceptance_length` against a reference is the only measurement that
can, which is why both gates are listed and why neither is optional.

**What the acceptance numbers do and do not establish.** They establish that
the MTP layer computes something the target model agrees with: 97.6% of draft
tokens accepted at con=1 is a ceiling-adjacent 3.9274 of a possible 4.0, and a
layer with the `eh_proj` halves swapped, a norm on the wrong operand or the
expert stack packed wrong would sit near 1.0. On their own they did **not**
establish that this is the *best* achievable draft quality — that needs the
same checkpoint under stock in-tree modeling at the same load and the same
`speculative_config`, which the section below measures.

**And read the workload the other way round from the usual warning.** The
harness generates uniformly random prompt token ids with `ignore_eos`, and
`docs/models/multi-token-prediction.md` warns that such a workload gives
acceptance *far below* real text. That warning does not hold here, and the
reason is that only the **prompt** is random: the 1024 output tokens are the
model's own continuation of nonsense, which is highly repetitive, and
repetition is exactly what an MTP layer drafts perfectly. So this probe
**flatters** MTP rather than penalizing it. It is a correctness instrument
here and nothing more; the value of enabling MTP has to be judged on a
real-text workload.

Two numbers from the probe that are *not* results and must not be quoted as
Pareto points: 211.17 tok/s/user at con=1 and 1942.9 tok/s at con=32. They are
measured under a different config (`free_gpu_memory_fraction: 0.75`, and
`--acceptance` turns on `enable_iter_perf_stats`, which the harness's own
header says makes a label non-comparable to one without it), in a two-round
probe rather than a sweep, on the flattering workload above. The perf campaign
for the variant belongs to the tuner.

### The parallel split, and what it rests on

| part | split | per-rank shape |
|---|---|---|
| `q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj` | **replicated**, all 128 query heads | `[1536, 7168]`, `[1536]`, `[24576, 1536]`, `[576, 7168]`, `[512]`, `[32768, 512]`, `[7168, 16384]` |
| layers 0-2 dense MLP | **replicated**, intermediate 18432 | `gate_up [36864, 3584]`, `down [7168, 9216]` |
| shared expert | **replicated**, intermediate 2048 | `gate_up [4096, 3584]`, `down [7168, 1024]` |
| routed experts | EP window, 64 of 256 at offset `64 * moe_ep_rank` | `fc1 [64, 4096, 3584]`, `fc2 [64, 7168, 1024]` |
| per-expert NVFP4 scalars | **replicated over all 256** (deliberate) | `[256]` each |
| fp8 KV scales, router, both norms, embedding | **replicated** | unchanged |
| `lm_head` (shell-owned) | **replicated** — the shell builds the whole matrix under attention DP | `[129280, 7168]` |

Two collectives per **MoE** layer, none in layers 0-2 (dense) and none on the
attention or residual path — 116 per forward:

* `comm/allgather` on the post-attention normed hidden states, **before** the
  router GEMM;
* `comm/reducescatter` on the expert window's output over the whole gathered
  token set, handing each rank back exactly its own rows.

Three things the correctness rests on, none of them checkable inside one rank:

* **the routing agrees across ranks, and the gather placement is what makes
  it so.** The four 64-wide windows must tile the routing space exactly once;
  under attention DP the ranks hold different tokens, so that is restored by
  gathering **before** the router GEMM — the router and `noaux_tc_op` then run
  on byte-identical full token sets on all four ranks and each token's top-8
  ids agree. Routing locally and gathering afterwards would break it with no
  error (`docs/models/expert-weight-packing.md`);
* **one activation quantization feeds every window.** The routed FC1 global
  scale is `1 / input_scale` of the *shared* expert, which the checkpoint sets
  to the max over all 256 routed experts — `derive_after_load` asserts that
  over the full 256, which is why the per-expert scalars are loaded whole on
  every rank;
* **the reduce-scatter is crossed in bf16.** The op sums, and it sums
  `float8_e4m3fn` as raw bytes rather than as floats; the gather is a byte
  move and would survive fp8, so the asymmetry is a live trap on the return
  leg only.

`attn_metadata.all_rank_num_tokens` is where the engine publishes the per-rank
row counts; `_dp_rows` reads it, asserts `all_rank_num_tokens[rank] ==
hidden_states.shape[0]`, and returns `max(counts)` — every rank pads its token
block to the group-wide maximum, so both collectives run in their uniform form
and the ragged form is never used. Padded rows are sliced off after the
reduce-scatter and never reach the residual stream.

### What the fp8 latent pool moves, and why each choice is what it is

The KV cache is fp8-e4m3 per the checkpoint, and **nothing validates the fp8
round trip at any layer**: the write scale, the read scale and the two folded
FMHA scales are independent roles with no relation checked anywhere in the
chain. Getting one wrong is silently mis-scaled output, never an error.

* **`quant_mode` is derived from the checkpoint's quant config**, not
  hard-coded: `kv_cache_quant_algo: FP8` selects `QuantMode`'s fp8-KV bit
  (128), the value every MLA entry certifies. The engine's own
  `quant_config.quant_mode` is a `QuantModeWrapper` rather than an int at this
  pin, and bits outside the KV-cache group ride along unread anyway (`1152`
  and `384` measured bit-identical to a bare `128` on every MLA flavor), so the
  target maps the declared algo onto the bit itself.
* **`s = 1.0`, and it is checked rather than assumed.** All 122 per-layer
  `k_scale`/`v_scale` tensors are loaded (they are 0-dim fp32 scalars) and
  `derive_after_load` asserts each is exactly 1.0. Both scale arguments are
  then passed as `None`, which every op reads as exactly 1.0 and which is what
  the engine's own call sites pass. This is not a convenience: the fp8 MLA
  *context* path is internally inconsistent at `s != 1.0` in **both** context
  flavors (it quantizes q/k/v at 1.0 while applying `s^2`/`s` as if it had
  not), so 1.0 is the only correct value and the assert is the defence.
* **The decode producers are ordered, not concurrent.**
  `mla_rope_generation` does not write `fused_q` over an fp8 pool — it *reads*
  `fused_q[..., :C]` to build `quant_q_buffer`, which is the query the decode
  FMHA consumes. The absorbed-q BMM is issued before it on the ambient stream,
  which is what makes that safe; the bf16 reading (disjoint halves, free to
  overlap) is a silent race here.
* **The two phases divide the quantization labour oppositely.** Context:
  `mla_rope_append_paged_kv_assign_q` leaves `q` plain bf16 and
  `thop_attention`'s context call quantizes q/k/v to e4m3 itself, so `q` is
  quantized exactly once. Decode: `mla_rope_generation` produces the quantized
  query and the two folded FMHA scales, and the generation call reads
  `quant_q_buffer`, `mla_bmm1_scale[1]` and `mla_bmm2_scale[0]` — **ignoring
  both kv scale tensors entirely**. Nothing in either signature says so.
* **A cached prefix pays fp8 twice.** The engine dequantizes cached latent
  rows off the pool, `kv_b_proj` up-projects them, and `thop_attention`
  quantizes the result straight back to e4m3. Not incorrect, but a
  cached-prefix context call carries strictly more quantization error than a
  fresh prefill of the same tokens — do not attribute an accuracy gap to block
  reuse without accounting for it.
* **fp8 raises the attention workspace requirement.** Measured here:
  **2,126,512,128 B (1.98 GiB) per rank**, resized on the first call, against
  531,628,032 B for the 32-head bf16 MLA shape at the same
  `max_num_tokens=8192`. Sizing from bf16 MLA figures under-budgets.

### The rope table is the whole of the rope configuration

`thop_attention` reads the table's **content** and `q_scaling`; the seven
scalar rope arguments and `rotary_inv_freq` beside them are measured inert on
the MLA path. So the YaRN blend lives entirely in the table this target
builds, and the model's YaRN attention temperature lives entirely in
`q_scaling`:

* table `inv_freq(d) = ramp(d)/(factor*freq(d)) + (1-ramp(d))/freq(d)` with
  `low = 10`, `high = 23` at this config (`R` 64, `theta` 10000, `factor` 40,
  `original_max_position_embeddings` 4096, `beta_fast` 32, `beta_slow` 1) —
  cross-checked against the HF reference's own YaRN rope init to **1.05e-8
  max abs / 1.27e-7 max relative** on `inv_freq` (the HF value is fp32, so
  that is its rounding);
* table amplitude `m(mscale)/m(mscale_all_dim)` = **exactly 1.0** (both are
  1.0 here), matching the same reference's `attention_factor`;
* `q_scaling = 1/m(mscale_all_dim)^2` = **0.5336594470450011**, which the
  construction asserts lands on `thop_attention`'s certified fp8-pool value
  (1.0 and 0.53366 are the two certified).

**The table is built for the full `max_position_embeddings` = 163840
positions** (84 MB fp32 per rank). A short table is read out of bounds with no
check — `rope_max_positions`, the argument that looks like it bounds this, is
one of the inert seven — so the first forward additionally asserts
`max_seq_len <= max_pos`.

### Envelope

* Context sequences with a cached prefix are served; chunked prefill is not
  implemented (`AttentionRuntimeFeatures.chunked_prefill=False` under
  defaults, and the target holds `chunked_prefill_buffer_batch_size=1`).
* **MTP is a `configs/` variant, never the identity.** The checkpoint declares
  `num_nextn_predict_layers: 1` and ships the whole of layer 61 for it — its
  own 256 experts, `embed_tokens`, `eh_proj`, two extra norms and a
  `shared_head.head`, 790 keys in all.

  Under `llm_args.yaml` alone those 790 keys are a **predicted non-load** in
  the weight manifest (read off the checkpoint by layer index rather than
  enumerated), no MTP parameter is declared, and the shell's `forward` is a
  plain delegation to the inherited base — nothing about this identity changed
  when MTP was added, which is a measured claim and not a design intention:
  after the increment, smoke's 10 greedy continuations are **byte-identical**
  to the pre-increment run's.

  Under `configs/mtp{1,2,3}.yaml` the module is loaded and a second forward
  path runs. **Those variants cross the line a config variant normally
  respects, and that is stated rather than left to be inferred**: a variant is
  supposed to move a knob, and these change the **weight-loading path** — layer
  61 goes from non-load to loaded, 212 keys and ~6.3 GB per rank — add a
  second forward (`MTPLayer`, replayed `max_draft_len` times per step), and
  change what the runtime allocates (a 62-layer KV pool, `max_draft_len - 1`
  extra tokens per sequence, and a `max_seq_len` the engine raises by 8 at
  `max_draft_len: 3`). Only the gates run against a variant can speak for it;
  the identity gate records below are not evidence about it, and vice versa.
* fp8-e4m3 latent pool only (`quant_mode` = the fp8-KV bit, KV scaling factor
  1.0); beam width 1; no LoRA, no cross attention, no FlashMLA layout — each
  asserted at the first forward or in the forward itself.
* `tokens_per_block == 32` is asserted: every MLA entry's **fp8** column is
  certified at page 32 only (their bf16 columns also carry 64). 32 is what a
  default `KvCacheConfig` produces, so this binds only if someone tunes the
  page size — which would need the certification extended first.
* Pipeline parallelism and a second (tensor) split of the routed experts are
  asserted off: `pp_size == 1`, `moe_tp_size == 1`. Attention DP is asserted
  **on** — this target is the DP assembly.
* A rank with **zero** tokens was never observed (idle ranks get a 1-token
  dummy) and is not something this forward was exercised on.

Certification coverage this target consumes, and where it sits relative to
what the entries measured:

* **`thop_attention`** runs MLA at `H = 128` over an **fp8-e4m3 latent pool**,
  page 32, `q_lora_rank = 1536`, at the complete DeepSeek-R1-0528 rope/scale
  cell — the exact configuration the entry certifies for all three MLA call
  flavors and their mixed-batch pairing. `H = 128` is the only head count
  certified over an fp8 pool, and page 32 the only page size.
* **`mla_rope_generation`**, **`mla_rope_append_paged_kv_assign_q`** and
  **`load_paged_kv_cache_for_mla`** run their fp8 columns at the same cell
  (`H = 128`, page 32, `C/R/nope/v = 512/64/128/128`, scale omitted).
* **`predicted_tokens_per_seq` (`P`) is 1 on every context call and the
  generation call's own query-tokens-per-sequence.** Without a
  `speculative_config` that is 1 everywhere; under `configs/mtp{1,2,3}.yaml` the
  MLA **generation** calls run at `P = max_draft_len + 1` (2, 3, 4) on the
  trunk and on the MTP layer's draft step 0, and at `P = 1` on draft steps 1+.
  `thop_attention` and `mla_rope_generation` certify `P` at **1, 2, 3 and 4**
  over this exact fp8 cell and no further, which is why `mtp3` is the largest
  variant here: `max_draft_len: 4` would need the certification extended
  first, not just a bigger number in the yaml. Two preconditions ride with it,
  both held structurally rather than checked (a device read would cost a sync
  and is illegal under capture):
  * **`L_g >= P` for every generation sequence** — draft row 0 attends to
    `[0, L_g - P]`, so a shorter KV length leaves it no keys. `L_g` counts all
    `P` of the step's tokens, so it cannot be smaller than `P`.
  * the `G*P` cache rows one `mla_rope_generation` call writes must address
    **distinct physical slots**, which a `KVCacheManager`-allocated batch gives
    automatically.
* **`fused_moe`** (the MTP layer's routed experts, bf16) runs at
  `E = 64`, `(H, I) = (7168, 2048)`, `K = 8`, `ep_size = 4`,
  `ep_rank ∈ {0,1,2,3}` — the cell the entry certifies, including all four
  windows against a 256-expert reference — with `T <= 2048`. Its
  **distinct-expert-ids precondition** matters here and is worth stating
  rather than leaving to inference: above 256 tokens a repeated id in one
  token's row reads out of bounds in `finalizeMoeRoutingKernel` (an illegal
  access, or ~200-460 ulp of silent garbage), and this call drives `T` to 2048.
  It holds **structurally**: the ids come from `noaux_tc_op`, whose semantics
  are the indices of the top-k largest corrected scores, and a top-k over
  expert indices cannot select one twice.

  The chunk bound is 2048 rather than the trunk's 8192 for a memory reason,
  not a certification one — see *What MTP costs in memory* below.
* **`fp4_block_scale_moe_runner`** runs at the R1 routed geometry
  (`H = 7168`, `I = 2048`, `num_experts = 256`, `top_k = 8`) with
  `local_num_experts = 64`, `local_expert_offset ∈ {0, 64, 128, 192}` — the
  certified four-way split — and at **`T <= 8192`**, the top of the entry's
  certified token column, which the whole column covers at this geometry. The
  gathered token set reaches `4 * max_num_tokens = 32768`, so the expert call
  is **chunked** (`_MOE_MAX_T` in `modeling.py`). **That bound is a
  certification boundary, not a tuning knob.** Routing and the activation
  quantization are chunked with it, which keeps `noaux_tc_op` inside its own
  enumerated column (up to `num_tokens` 8192) as well.
* **`noaux_tc_op`** runs the grouped configuration
  `(num_experts, n_group, topk_group, topk) = (256, 8, 4, 8)`, which the entry
  enumerates explicitly.
* **`comm/allgather` and `comm/reducescatter`** are driven at world size 4,
  group `[0,1,2,3]`, bf16, hidden 7168, in their uniform form only, at 58
  sites per step — 58 + `max_draft_len` under an MTP variant, the MTP layer
  being one more MoE layer per draft step. Both contracts certify the
  call-order surface: calls pair by **position** on the communicator, so
  issuing the same sequence on every rank is this forward's obligation —
  discharged structurally, since every rank runs the same layer loop and pads
  to `max(all_rank_num_tokens)`.

  **Inside the draft loop the padding basis is a different list, and reading
  the wrong one is silent.** The worker leaves `attn_metadata.all_rank_num_tokens`
  holding the trunk's counts for the whole loop and passes the correct basis in
  as the `all_rank_num_tokens` **keyword** —
  `spec_metadata.all_rank_num_tokens` at draft step 0, then
  `spec_metadata.subseq_all_rank_num_tokens`, which is the per-rank *sequence*
  count. Both contracts certify that at equal byte counts a mispairing does not
  hang: every rank comes back wrong in 98-99% of elements, bitwise
  reproducibly. The defence is structural rather than vigilant — the MTP
  layer's padding helper (`_mtp_dp_rows`) takes the list as a parameter and has
  **no metadata argument at all**, so it cannot reach the wrong one; the
  trunk's `_dp_rows` keeps its own shape and the two are not interchangeable.

  **Both collectives stay on the ambient stream**, which since the
  `iter1-shared-side-stream` iteration is not the only stream the forward
  uses: the shared-expert branch is forked onto a side stream and joined
  before the add. That is inside what both contracts certify — "the side
  stream joined to the current one on both ends, which is what the engine and
  a target's forward both do" — and it changes neither collective's arguments,
  its call order, nor its uniform form.
* **`nvfp4_gemm` and `fp4_quantize` are certified at this target's shapes**,
  on receipts rather than on a domain rule. They were assembled as "inside the
  stated domain, outside the enumerated list" and closed afterwards, because
  R1's widths *exceed* every previously-run value rather than falling between
  them. `nvfp4_gemm` at `(K, N)` = (7168, 36864), (18432, 7168), (7168, 4096),
  (2048, 7168) with `M` to 8192; `fp4_quantize` at `K` = 7168 / 18432 / 2048
  through `T = 8192`, in **both** scale layouts (swizzled for the GEMM, linear
  for the MoE runner — `K = 7168` is taken in both).

  Both runs came back "the rule was sufficient after all", and both found the
  contract wrong about *why*. `nvfp4_gemm`: the tactic space does move with
  shape, but every R1 count lands inside the range the smaller shapes produce
  and all four backends agree bitwise. `fp4_quantize`: the kernel-selection
  axis its contract described (a TMA variant above 1024 rows) **does not exist
  in this build** — that text came from flashinfer's vendored source, which is
  a newer revision than the installed binary; the profiler sees one kernel at
  every shape.

### What MTP costs in memory, and why the variants carry a second knob

The MTP module adds **5.87 GiB (6.30 GB) of declared weights per rank** — 5.637
GB of it the bf16 routed expert stacks alone, which cost **3.56x what one NVFP4
trunk MoE layer costs** purely from the dtype. `nvidia-smi` read **122,076 MiB
(119.2 GiB) per rank** after model init with MTP on, against the 117,924 MiB
(115.2 GiB) recorded above for the identity config; the two readings were taken
at different points of the load, so treat them as two figures rather than as a
clean difference. The engine then adds one KV-pool layer (62 instead of 61,
+1.6%) and `max_draft_len - 1` extra tokens per sequence, neither of which the
target declares.

**That is not what makes it tight.** The engine sizes the KV pool from free
memory *after* the weights, at `free_gpu_memory_fraction`, and the drafting
forward's transient demand after that point is larger than the identity
config's. At the default 0.9 the pool takes 38.95 GiB (1,171,200 tokens per
rank) and boot then reaches CUDA-graph capture at **182.4 of 183.4 GiB and
livelocks** — three of four ranks spinning in `cudaFree` inside the CUDA
caching allocator's `release_cached_blocks`, reached from an ordinary
`empty_cuda` in the *trunk's* MoE runner, while the fourth waits at an
`MPI_Barrier`. There is no OOM exception and no error: the run simply stops
making progress and has to be killed. **Read the signature — one rank idle at a
barrier, the rest at 100% GPU with a frozen log — as memory exhaustion, not as
a mispaired collective.**

**Freeing memory before the pool is sized does not help**, and that is worth
recording because it is the obvious first move: chunking the MTP layer's
expert call to 2048 rows (from the trunk's 8192, both inside `fused_moe`'s
certified column) moved pool sizing by **0.31 GiB**, 38.64 -> 38.95, and the
boot failed identically — the pool grows into exactly whatever is freed. The
chunk was reverted to the trunk's `_MOE_MAX_T`; it is not the lever.

So `configs/mtp{1,2,3}.yaml` each carry
`kv_cache_config.free_gpu_memory_fraction: 0.75` as a **boot requirement**,
documented in the files themselves, in the same spirit as
`configs/trtllm-ref-boot.yaml`. It hands the pool ~32.5 GiB and leaves ~28 GiB
of headroom, and it constrains nothing: ~975k KV tokens per rank is an order
of magnitude past what any concurrency measured on this target uses.

### Engine-side facts observed on this checkpoint

* **The latent pool is fp8 and the engine sized it that way.** 44.77 GiB for
  1,368,032 tokens per rank = **35,136 B/token** =
  `num_layers * (kv_lora_rank + qk_rope_head_dim) * 1` = `61 * 576 * 1`. One
  byte per element, i.e. half the bf16 width — no target-side declaration was
  needed beyond keeping the checkpoint's `hf_quant_config.json` linked.
* **The engine allocates the KV pool twice**: a small profiling pool (5.50
  GiB, 167,936 tokens) is created and released before the real one.
* **Per-rank weights are ~115.2 GiB** (`nvidia-smi` read 117,924 MiB on three
  of the four devices after model init and before the pool allocation; the
  fourth carried an unrelated 810 MiB from another user's process), against a
  183 GiB card. `Model init total` is 51-55 s per rank with the checkpoint
  warm in the host page cache — the 163 shards load in ~10 s per rank there.
* **CUDA-graph capture is the default decode grid**: `batch_sizes = [1..32,
  64, 128]`, `max_batch_size 128`, `enable_padding False` — 34 sizes, all
  inside the enumerated CUDA-graph coverage both collective entries carry
  (`1..32, 64, 128, 256`).
* **First boot JIT-compiles 8 fp8 MLA decode kernels per rank** at ~6.0-7.0 s
  each — `fmhaSm100aKernel_QkvE4m3OBfloat16HQk576HV512...P32VarSeqQ16Kv128StaticSwapsAbForGen`
  and its `MultiCtasKvCga` / `HVPerCta256` siblings. `QkvE4m3` in the name is
  the fp8 pool: q, K and V are all e4m3 in the decode MMAs. No MLA context
  call triggers a compile.

  **Each `max_draft_len` costs its own decode compiles**, because
  `predicted_tokens_per_seq` becomes the kernel's `maxSeqLenQ` and moves the
  `HVPerCta` split in the name — `P` 1 and 2 take `HVPerCta128`, `P` 3 and 4
  take `HVPerCta256`, and `P` 1 and 2 pay separate compiles despite sharing a
  name. Measured on `configs/mtp3.yaml`: **12 compiles per rank** against the
  identity config's 8, i.e. **+4 at ~5.4-5.7 s each**, all four ranks
  compiling in parallel. A campaign sweeping `max_draft_len` should budget one
  first-boot compile set per value, not one for the model.
* `max_seq_len` is the config's 163840 under the identity config, so
  `max_blocks_per_seq` is 5120 at page 32. **A `speculative_config` raises it**:
  163848 at `max_draft_len: 3` (5121 blocks per sequence), which is more than
  the `max_draft_len - 1` extra KV tokens per sequence the runtime reference
  documents. The rope table is sized from the engine's own number rather than
  from a formula, on the first forward.

## Performance

![Serving Pareto](perf/figures/pareto.png)

Environment: `umbriel-b200-027`, 224-core, 8x NVIDIA B200 (178.34 GiB each),
**GPUs 3-6** — the campaign device set, held for every label — driver
**595.58.03**, `tensorrt_llm 1.3.0rc21`, `torch 2.11.0+cu130`.
`source scripts/env.sh` then `CUDA_VISIBLE_DEVICES=3,4,5,6`. ISL=OSL=1024,
concurrency 1..256. **Two sessions on the same host and the same device set**:
the first three curves 2026-08-01 10:45-15:18 UTC, the three MTP curves plus
the two new stock references 2026-08-02 13:09-17:35 UTC. The second session
opened by re-measuring `iter1` (`probe-anchor-iter1`), which reproduced it to
**−0.14% at con=1 and +0.15% at con=256**, and the `trtllm` reference was
spot-checked at the end of it to **−0.01% / +0.20%** — that is what licenses
one figure across the two days.

`con=1 tok/s/user` is `1000 / mean_tpot_ms` at con=1; `peak tok/s/GPU` is
`max(output_throughput) / 4`.

| label | config | commit | accuracy | con=1 tok/s/user | peak tok/s/GPU | change |
|---|---|---|---|---|---|---|
| `baseline` | `llm_args.yaml` (identity) | `70b1ecd` | gsm8k 94.9962 | 76.09 | 1906.86 (con=256) | trtllm defaults |
| `iter1-shared-side-stream` | `llm_args.yaml` (identity) | `75a471a` | gsm8k **94.7688** | 82.43 | 1948.78 (con=256) | shared expert forked onto a side stream |
| `mtp1` | identity + `configs/mtp1.yaml` | `bdbb1ab` | covered by the mtp3 pair, below | 135.11 | 2184.44 (con=256) | MTP, `max_draft_len: 1` |
| `mtp2` | identity + `configs/mtp2.yaml` | `bdbb1ab` | covered by the mtp3 pair, below | 164.94 | 2190.04 (con=256) | MTP, `max_draft_len: 2` |
| `mtp3` | identity + `configs/mtp3.yaml` | `bdbb1ab` | gsm8k **95.2237** (paired, +0.1517) | **190.45** | **2262.06** (con=256) | MTP, `max_draft_len: 3` — the frontier |
| `trtllm` | identity + `configs/trtllm-ref-boot.yaml` | `70b1ecd` | ungated | 73.94 | 1522.73 (con=256) | stock in-tree modeling at its **default** MoE transport (DeepEPLowLatency); config is boot-forced, see below |
| `trtllm-nodeepep` | identity + `configs/trtllm-ref-boot.yaml`, `TRTLLM_CAN_USE_DEEP_EP=0` | `471e6a3` | ungated | 79.20 | 1768.06 (con=256) | the same, on **AllGatherReduceScatter** — the transport control, 3 points |
| `trtllm-mtp3` | identity + `configs/trtllm-ref-mtp3.yaml`, `TRTLLM_CAN_USE_DEEP_EP=0` | `471e6a3` | ungated | 178.66 | 1780.70 (con=256) | **stock modeling with MTP at the same `max_draft_len: 3`** — the reference `mtp3` should be read against |
| `trtllm-tuned` | — | — | — | — | — | superseded by `trtllm-mtp3`, which measures exactly this; see below |
| **`trtllm-aligned`** | **`llm_args.yaml` (identity), `TRTLLM_CAN_USE_DEEP_EP=0`** | `471e6a3` | ungated | **77.82** | **1887.78** (con=256) | **stock modeling at this target's own config — no boot knobs at all. Supersedes `trtllm` / `trtllm-nodeepep`** |
| **`trtllm-aligned-mtp3`** | **identity + `configs/mtp3.yaml`, `TRTLLM_CAN_USE_DEEP_EP=0`** | `471e6a3` | ungated | **181.74** | **1728.86** (con=256) | **stock modeling with MTP at the same config file `mtp3` uses. Supersedes `trtllm-mtp3`** |

Per point, output tok/s (whole 4-GPU node):

| con | `trtllm` | `baseline` | `iter1` | iter1/trtllm | t TPOT | i TPOT | t TTFT | i TTFT |
|---|---|---|---|---|---|---|---|---|
| 1 | 73.28 | 75.63 | 81.90 | **1.118x** | 13.525 | 12.132 | 138.6 | 92.4 |
| 2 | 140.11 | 145.44 | 157.59 | 1.125x | 14.059 | 12.554 | 234.0 | 151.9 |
| 4 | 270.94 | 285.41 | 308.40 | 1.138x | 14.525 | 12.831 | 258.2 | 155.2 |
| 8 | 474.91 | 513.02 | 552.59 | 1.164x | 16.486 | 14.279 | 383.2 | 216.2 |
| 16 | 818.94 | 913.71 | 972.02 | 1.187x | 19.091 | 16.131 | 464.4 | 352.0 |
| 32 | 1360.02 | 1584.57 | 1673.50 | 1.230x | 23.033 | 18.514 | 518.1 | 637.5 |
| 64 | 2335.92 | 2724.10 | 2817.55 | 1.206x | 26.745 | 21.639 | 665.4 | 1116.6 |
| 128 | 3835.72 | 4507.69 | 4566.30 | 1.190x | 32.373 | 26.663 | 930.3 | 1406.9 |
| 256 | 6090.94 | 7627.45 | 7795.10 | **1.280x** | 40.129 | 31.092 | 1427.8 | 1762.4 |

**That 11.8-28.0% is measured against a reference that could not run this
target's own config, and a later session showed it did not have to be that
way.** Read the next subsection before quoting any number from the table
above. Mean TPOT is lower at every point, and the mean-TTFT crossover at
con=32..256 is a property of the reference's forced config rather than of the
modeling layer.

### The reference's boot config turned out to be avoidable, and it cost 24%

`configs/trtllm-ref-boot.yaml` exists because stock trtllm OOMs at engine
construction on this checkpoint: 113.9 GiB per rank allocated before any weight
is materialized, `= 1.96 GiB x 58 MoE layers`, which is DeepEP low-latency's
communication workspace sized from `max_num_tokens`. Capping `max_num_tokens`
and `max_seq_len` to 2048 treats that symptom. **`TRTLLM_CAN_USE_DEEP_EP=0`
removes the cause** — the MoE communication factory then falls through to
`AllGatherReduceScatter`, which its own source comment calls "always works",
and which is the transport *this target implements by hand*. Stock trtllm then
boots at the **identity config**, `max_num_tokens` 8192 and `max_seq_len`
163840, peak 179,188 MiB with ~4 GB to spare. That switch was not known when
the boot config was written.

Re-measured on 2026-08-03, GPUs 3-6, both sides byte-identical config
(`trtllm-aligned` and `trtllm-aligned-mtp3`):

| con | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| the boot config was costing the reference | +6.2% | +8.2% | +10.9% | +11.6% | +15.9% | **+24.0%** |
| **`iter1` / `trtllm-aligned`** (no MTP) | 1.052x | 1.075x | **1.109x** | 1.081x | 1.027x | **1.032x** |
| **`mtp3` / `trtllm-aligned-mtp3`** | 1.045x | 1.313x | 1.289x | 1.339x | **1.406x** | **1.308x** |

**So the honest no-MTP lead is 2.7-10.9%, not 11.8-28.0%**, and the honest MTP
lead is 1.308x. Both older reference lines and the whole
*MoE transport is worth 7-16%* subsection below are superseded by this: the
transport is no longer a variable to isolate, because both systems now run the
same one.

**And the MTP lead is not a better MTP layer.** Per system, at matched config,
MTP is worth:

| con | 1 | 8 | 32 | 128 | 256 |
|---|---|---|---|---|---|
| to `staircase` | 2.285x | 1.930x | 1.301x | 1.332x | **1.161x** |
| to stock trtllm | **2.300x** | 1.580x | 1.119x | **0.973x** | **0.916x** |

They are level at con=1 — stock is fractionally ahead — and stock goes
*negative* from con=128. The 1.308x is that divergence, not a drafting-quality
difference; acceptance agrees to a few percent everywhere.
`workbench/docs/2026-08-03-r1-dep4-staircase-vs-trtllm.md` carries the
kernel-level attribution: it is one family, MoE expert GEMM, and the driver is
**row count, not a re-run draft layer**. MTP gives every generation sequence
`max_draft_len + 1 = 4` query tokens, so the *trunk*'s MoE gathers 4x the rows
(256 -> 1024 per rank at con=256). The two implementations scale differently
under that: CUTLASS grouped GEMM goes from 5 to 8 kernels per MoE layer, while
trtllm-gen's `bmm_E2m1_*` barely moves. Expert-GEMM-family launches per rank
per step: **174 -> 192 (+10%) on our side, 290 -> 482 (+66%) on theirs** — and
of their +192, the MTP layer itself accounts for only 15. **The growth is in
the trunk, not in the draft layer.**

The MTP curves, against `iter1` (all four are full sweeps; `mtp*` from the
2026-08-02 session). `accept` is mean tokens emitted per engine step, ceiling
`max_draft_len + 1`:

| con | `iter1` | `mtp1` | `mtp2` | `mtp3` | mtp3/iter1 | i TPOT | m3 TPOT | i TTFT | m3 TTFT | m3 accept |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 81.90 | 133.54 | 162.51 | 187.12 | **2.285x** | 12.132 | 5.251 | 92.4 | 100.9 | 3.4904 |
| 2 | 157.59 | 252.37 | 313.21 | 341.41 | 2.166x | 12.554 | 5.611 | 151.9 | 124.7 | 3.4040 |
| 4 | 308.40 | 501.37 | 622.24 | 677.82 | 2.198x | 12.831 | 5.689 | 155.2 | 123.8 | 3.3923 |
| 8 | 552.59 | 803.10 | 928.24 | 1066.23 | 1.930x | 14.279 | 7.145 | 216.2 | 166.6 | 3.4151 |
| 16 | 972.02 | 1316.50 | 1313.32 | 1431.51 | 1.473x | 16.131 | 9.394 | 352.0 | 217.9 | 3.3905 |
| 32 | 1673.50 | 2135.72 | 2091.70 | 2176.90 | 1.301x | 18.514 | 12.199 | 637.5 | 282.6 | 3.3356 |
| 64 | 2817.55 | 3341.71 | 3627.89 | 3723.24 | 1.321x | 21.639 | 13.618 | 1116.6 | 415.4 | 3.4407 |
| 128 | 4566.30 | 5877.32 | 5733.13 | 6080.38 | 1.332x | 26.663 | 17.524 | 1406.9 | 612.1 | 3.4173 |
| 256 | 7795.10 | 8737.78 | 8760.17 | 9048.23 | **1.161x** | 31.092 | 23.758 | 1762.4 | 923.9 | 3.4033 |

**`mtp3` is ahead of `mtp1` and `mtp2` at every one of the nine
concurrencies**, and ahead of `iter1` at every one, so the `max_draft_len` axis
never turns over inside the certified range. Mean TTFT also falls at every
point except con=1, where it rises 92.4 -> 100.9 ms.

**Do not read the `mtp3` column against the `trtllm` row.** That row has no
speculative decoding at all, so the ratio between them is mostly "one system
drafts and the other does not", not a modeling-layer delta. The comparison
this target should be quoted on is `mtp3` against **`trtllm-mtp3`** — stock
in-tree modeling, same checkpoint, same `max_draft_len: 3`, same MoE
transport:

| con | `trtllm-mtp3` | `mtp3` | **mtp3 / trtllm-mtp3** | their accept | our accept | tm TPOT | m3 TPOT | tm TTFT | m3 TTFT |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 175.44 | 187.12 | **1.067x** | 3.5108 | 3.4904 | 5.597 | 5.251 | 110.6 | 100.9 |
| 2 | 342.16 | 341.41 | **0.998x** | 3.5415 | 3.4040 | 5.709 | 5.611 | 128.4 | 124.7 |
| 4 | 621.75 | 677.82 | 1.090x | 3.3315 | 3.3923 | 6.183 | 5.689 | 126.3 | 123.8 |
| 8 | 845.69 | 1066.23 | 1.261x | 3.3058 | 3.4151 | 8.788 | 7.145 | 225.3 | 166.6 |
| 16 | 1044.72 | 1431.51 | **1.370x** | 3.3305 | 3.3905 | 12.020 | 9.394 | 270.3 | 217.9 |
| 32 | 1768.44 | 2176.90 | 1.231x | 3.3458 | 3.3356 | 14.690 | 12.199 | 335.1 | 282.6 |
| 64 | 2797.56 | 3723.24 | 1.331x | 3.2835 | 3.4407 | 18.454 | 13.618 | 446.2 | 415.4 |
| 128 | 4510.25 | 6080.38 | 1.348x | 3.3125 | 3.4173 | 23.658 | 17.524 | 688.2 | 612.1 |
| 256 | 7122.80 | 9048.23 | **1.270x** | 3.3145 | 3.4033 | 29.746 | 23.758 | 1134.4 | 923.9 |

**The honest headline is 1.27x at con=256 and 1.00-1.09x at con=1..4**, not
the 2.29x the no-MTP row invites. Two things follow, and they are the point of
this reference:

* **Both systems draft equally well.** Acceptance agrees to within a few
  percent at every concurrency (theirs 3.28-3.54, ours 3.40-3.49), which is
  the same conclusion the level-3 acceptance gate reached and is what makes
  the throughput ratio a *speed* comparison rather than a draft-quality one.
* **The modeling-layer delta survives MTP essentially unchanged.** At con=256
  it is **1.270x with MTP against 1.280x without** — and the gap decomposition
  below prices the reference's forced config at 1.035x on our side at that
  point, so both reduce to a ~1.23x modeling delta. MTP moved the whole
  frontier; it did not move the distance between the two systems.

At con=1..4 the two are level. Stock trtllm's low-concurrency MTP path is as
good as ours; our lead only opens up from con=8, which is where `iter1`'s
side-stream overlap and the rest of the trunk work start to matter.

#### The MoE transport is worth 7-16% to stock trtllm, and it is not noise

**Superseded by `trtllm-aligned`.** This subsection isolated the transport as
a variable because the reference was stuck on `configs/trtllm-ref-boot.yaml`.
It no longer is: with `TRTLLM_CAN_USE_DEEP_EP=0` stock trtllm boots at the
identity config, so both systems now run `AllGatherReduceScatter` and there
is nothing left to isolate. The measurement below stands on its own.

`trtllm-mtp3` cannot run on DeepEPLowLatency — that transport's dispatch
accepts only NVFP4 hidden states and the MTP layer is bf16 — so it is forced
onto `AllGatherReduceScatter`, while the existing `trtllm` curve was measured
on DeepEPLowLatency. `trtllm-nodeepep` isolates that one variable: stock
modeling, **no** MTP, the same boot config, `TRTLLM_CAN_USE_DEEP_EP=0`.

| con | `trtllm` (DeepEPLowLatency) | `trtllm-nodeepep` (AllGatherReduceScatter) | delta |
|---|---|---|---|
| 1 | 73.28 | 78.64 | **+7.32%** |
| 32 | 1360.02 | 1491.47 | **+9.67%** |
| 256 | 6090.94 | 7072.22 | **+16.11%** |

This does **not** collapse into the existing reference — it is 7 to 16 times
this session's measured con=256 floor of 0.01%. Two corrections follow, and
both cut against this target:

* **Stock trtllm's default transport is the slower one here**, so the main
  table's `trtllm` row understates what stock trtllm can do on this host.
  Against the faster stock configuration, `iter1`'s no-MTP lead is
  **1.041x / 1.122x / 1.102x** at con 1/32/256 — not the 1.118x / 1.230x /
  1.280x measured against the default. The 11.8-28.0% claim above is against
  `trtllm` **as stock defaults configure it**, which is the honest definition
  of that reference line but is not the best stock can do.
* **MTP is worth far less to stock trtllm at scale than to us, once the
  transport is held fixed.** `trtllm-mtp3 / trtllm-nodeepep` is **2.231x at
  con=1, 1.186x at con=32 and 1.007x at con=256** — at the throughput end,
  drafting buys stock trtllm essentially nothing. Ours, `mtp3 / iter1`, is
  **2.285x / 1.301x / 1.161x**. That divergence at high concurrency is the
  clearest single statement of what this target's modeling layer is worth
  under MTP: the two systems draft equally well and both pay a step-cost
  penalty that grows with batch, but ours stays ahead of the penalty and
  stock's does not.

`trtllm-nodeepep` is three points, not a swept curve — con 1/32/256 at the
same rounds the full sweeps use, so it pairs point-to-point with `trtllm`.
con=128 was deliberately skipped: this session measured 4.2% run-to-run spread
there, so it could not have carried the claim.

**Read these curves as "did anything get slower, and where does it stop
paying" — not as "is MTP worth enabling".** The harness builds prompts from
uniformly random token ids and then generates with `--ignore-eos`, so what the
draft layer predicts is the model's own continuation of a nonsense prefix,
which degenerates into repetition — and repetition drafts near-perfectly. The
`accept` column above, 3.39-3.49 of a ceiling of 4.0, is that artefact. **The
value number for MTP on this checkpoint is the real-text one**, from the paired
gsm8k runs recorded under *The MTP variant's own gate records*: execution
**37.025 s -> 31.440 s, 1.178x** over 1319 genuine prompts.

### Reference lines

* **`trtllm`** = the original checkpoint under stock in-tree modeling, this
  target's `llm_args.yaml` (identity config: `tensor_parallel_size: 4`,
  `moe_expert_parallel_size: 4`, `enable_attention_dp: true`), **plus
  `configs/trtllm-ref-boot.yaml`, which is boot-forced rather than tuned.**
  Ungated. `tensorrt_llm 1.3.0rc21`.

  **Stock trtllm cannot boot this checkpoint at `dep4` on a 178.34 GiB B200
  under its own defaults.** Measured, per rank: at the default
  `max_num_tokens: 8192`, **113.9 GiB is allocated at engine construction
  outside the torch allocator, before any weight is materialized** — an
  `nvidia-smi` sampler at 5 s intervals caught it going 20 MiB ->
  116,662 MiB inside one sample, immediately after the MoE communication
  factory logged `Selected communication strategy: DeepEPLowLatency` once per
  MoE layer (`NVLinkOneSided` / `NVLinkTwoSided` / `DeepEP` all reported
  `not available: Invalid Argument`). Model init then dies in
  `init_meta_tensor` with 64.17 GiB held by PyTorch and 113.8 GiB outside it.
  113.9 GiB / 58 MoE layers = 1.96 GiB per layer. `NVSHMEM_SYMMETRIC_SIZE=1g`
  does not change it.

  The allocation is proportional to `max_num_tokens`: at 2048 the weights
  load (135.85 GiB torch, including 1.53 GiB of CUDA-graph pools) and the
  failure moves to `configure_kv_cache_capacity`, which asks for 7.00 GiB
  with 5.61 GiB free. `kv_cache_config.free_gpu_memory_fraction` does **not**
  move that 7.00 GiB (identical failure at 0.9 and at 0.6). The failure's own
  memory ledger names the remaining lever, and `max_seq_len: 2048` — exactly
  the harness's ISL+OSL, and `max_blocks_per_seq` 5120 -> 64 — is what makes
  it boot. Both knobs together are the reference's config.

* **`trtllm-nodeepep`** = the `trtllm` line with one variable moved:
  `TRTLLM_CAN_USE_DEEP_EP=0`, which disables DeepEP and DeepEPLowLatency
  together and lands the MoE communication factory on
  `AllGatherReduceScatter` — the strategy this target implements by hand.
  Stock in-tree modeling, no MTP, same boot config, three points. Ungated.
  It exists to keep the transport from being confounded with MTP, and it is
  worth 7-16%; see above.

* **`trtllm-mtp3`** = the original checkpoint under stock in-tree modeling
  with **the same speculative config this target's kept variant uses**
  (`configs/trtllm-ref-mtp3.yaml` = the boot knobs + `decoding_type: MTP`,
  `max_draft_len: 3`, `free_gpu_memory_fraction: 0.75`), plus
  `TRTLLM_CAN_USE_DEEP_EP=0`. Ungated. Full nine-point sweep, no
  `--acceptance`. **This is the reference `mtp3` is quoted against**, and it
  replaces what the `trtllm-tuned` slot was for: it *is* stock modeling under
  this campaign's final best config.

  It carries three forced deviations from `configs/mtp3.yaml` —
  `max_num_tokens` / `max_seq_len: 2048` to boot at all, and the transport
  variable — so it is not a *portable-config* split in the usual sense. Both
  are priced rather than waved away: the boot knobs are worth `+1.5% / −0.6% /
  −3.4%` on our side (the gap decomposition below), and the transport is worth
  `+7.3% / +9.7% / +16.1%` on stock's side and is applied to `trtllm-mtp3`
  already. Netting the boot knobs out of the con=256 ratio leaves ~1.23x, the
  same modeling delta the no-MTP decomposition finds.

* **`trtllm-tuned`** — **retired, superseded by `trtllm-mtp3`.** The slot means
  "stock modeling plus the campaign's final best config", and that is now
  measured rather than argued: the final best config is `configs/mtp3.yaml`
  and `trtllm-mtp3` runs stock modeling under it. (For the identity campaign
  the slot was genuinely degenerate — no config variant was kept, and stock
  trtllm cannot boot at the identity config at all.)

### The gap decomposition

Because the reference carries a forced config, the honest split is measured
rather than asserted: `probe-refcfg` ran **this target's `iter1` code under
the reference's own config**, so both systems can be compared at identical
knobs.

| con | staircase @ identity | staircase @ ref config | trtllm @ ref config | config-matched delta | end-to-end |
|---|---|---|---|---|---|
| 1 | 81.90 | 83.13 (+1.51%) | 73.28 | **1.134x** | 1.118x |
| 32 | 1682.70* | 1673.15 (−0.57%) | 1360.02 | **1.230x** | 1.230x |
| 256 | 7802.74* | 7535.64 (−3.42%) | 6090.94 | **1.237x** | 1.280x |

\* paired two-round probes, the shape `probe-refcfg` used; the full-sweep
numbers are in the table above.

**0% of the end-to-end gap is portable config and 100% of it is the
modeling-layer delta** — the campaign kept no config variant, so there is no
portable config gain to transfer. The reference's forced config is worth
`+1.5% / −0.6% / −3.4%` on our side, i.e. at con=256 the end-to-end 1.280x is
a 1.237x modeling delta plus a 1.035x config difference that happens to
favour the identity config on throughput. At con=1 and con=32 the
config-matched and end-to-end numbers agree to within the session spread.

That forced config is also a real **TTFT-vs-throughput trade on this target**,
which is what the TTFT crossover in the main table is: at con=256 it costs
−3.42% output throughput and buys **mean TTFT 2269.5 -> 1645.7 ms (−27.5%)**;
at con=32, −0.57% for 627.8 -> 384.0 ms (−38.8%). It is correctly not kept —
the Pareto axes are tok/s/user and tok/s/GPU, and it loses on one and is flat
on the other — but a latency-sensitive deployment of this target should reach
for `max_num_tokens: 2048` first.

### Iterations

**Two kept: one modeling change (`iter1`), one config axis (`mtp3`).**

**`iter1-shared-side-stream` — the shared expert overlaps the MoE round
trip.** *Evidence.* An nsys window of 100 executor iterations at con=256
(`TLLM_PROFILE_START_STOP=1200-1300`, `-c cudaProfilerApi`,
`--cuda-graph-trace node`) caught all 4 ranks, 778,000 kernels, and **100
pure-decode steps** — no context FMHA kernel appears at all, and
`cudaGraphLaunch` = 400 against 4 ranks x 100 steps = **1.00 graph replay per
rank per step, 100% coverage**. Per rank the GPU was **98.5% busy** (union
2822.3 ms against a 2865.1 ms window, idle **1.49%**), so the target is
GPU-bound, not host-bound. Walking the intervals per device — kernels on
different GPUs genuinely run in parallel — gave exclusive time (the interval
covered by no other kernel):

| family | sum µs/rank-step | exclusive µs/rank-step | exclusive/union | % of GPU busy |
|---|---|---|---|---|
| routed expert GEMMs (`bmm_E2m1*`, `bmm_Bfloat16*`) | 13488.2 | **13078.7** | 98.9% | **46.3%** |
| cuBLAS bf16 GEMMs (`nvjet_*`, attention path) | 6828.4 | 5665.5 | 86.6% | 20.1% |
| NCCL collectives (`AllGather`/`ReduceScatter` `RING_LL`) | 3572.7 | **3369.7** | **94.3%** | **11.9%** |
| MLA decode FMHA | 1079.7 | 1015.2 | 94.0% | 3.6% |
| `splitKreduce_kernel` | 1062.6 | 324.3 | 30.5% | 1.1% |

GPU busy is 28.22 ms per rank-step. The collectives are **94.3% exclusive** —
for almost all of the time one is running, the device is running nothing else
— which is a 3.37 ms per-step window with nothing in it.

*Change.* The shared expert and the routed round trip both read the
post-attention `o` and meet only at the final `add`, but on one stream the
shared expert's five kernels sat *in front of* the all-gather. `forward` now
forks that branch onto a side stream (`side.wait_stream(main)` ->
`with torch.cuda.stream(side)` -> `main.wait_stream(side)` before the add),
so it runs inside the collective window. Both collective contracts certify
"a side stream joined to the current one on both ends", which is exactly this
shape; the same fork/join pair is what propagates a CUDA-graph capture into
the branch and back, and the captured decode graph keeps working (smoke's 10
greedy continuations are byte-identical to the baseline's).

*Effect.* Output throughput rises at **every** concurrency — +8.28, +8.35,
+8.05, +7.71, +6.38, +5.61, +3.43, +1.30, +2.20 % at con 1..256 — and mean
TPOT falls at every point. Frontier: con=1 tok/s/user 76.09 -> **82.43**
(+8.3%), peak tok/s/GPU 1906.86 -> **1948.78** (+2.2%). The gain is largest
at low concurrency, which the profile predicts: the collectives are
latency-bound and roughly batch-independent, so the window they leave is a
larger share of a smaller step, while at con=256 the shared expert competes
for HBM bandwidth with the routed expert GEMMs and only part of it hides.
Accuracy: **gsm8k 94.7688** `exact_match,flexible-extract` (strict-match
94.6171) against reference 94.9962, gate >= 90.0 — passed. The 0.23 delta is
3 questions of 1319, well inside the run's own ±0.6133 stderr, and
strict-match is bit-for-bit the same score as the assembly run.

**This iteration makes two statements elsewhere in this file stale**, and
they are gate-record prose that a tuner does not edit: the *parallel split*
section and the *Vocabulary* section both say the two collectives are driven
"in their uniform form only" and that every rank "pads to
`max(all_rank_num_tokens)`". Both are still true — `iter1` did not change
either collective's arguments — but the surrounding claim that the forward is
single-stream no longer is. The certification consumed is unchanged: same
entries, same arguments, same uniform form, and both contracts certify side
streams explicitly.

**`mtp3` — multi-token prediction at `max_draft_len: 3`, the whole certified
range swept.** *Evidence.* The MTP layer's own arithmetic
(`docs/models/multi-token-prediction.md`) puts the break-even acceptance at
1.055 / 1.110 / 1.164 for draft lengths 1 / 2 / 3, and predicts the step cost
to be flat in concurrency because a decode step is weight-bandwidth-bound. It
also flags the high-concurrency end as the genuinely uncertain one: once a
rank's batch already activates all 64 of its local experts, `draft_len + 1`
times the rows means the same weight bytes with several times the arithmetic,
and the layer can cross from memory-bound to compute-bound.

*Change.* Config only — `configs/mtp{1,2,3}.yaml`, one full Pareto sweep each,
no modeling edit. Nothing else moved: `modeling.py` is byte-identical across
all three, and `--acceptance` was deliberately **not** used (below).

*Effect.* `mtp3` takes the frontier on both axes: con=1 tok/s/user
**82.43 -> 190.45 (+131.1%)** and peak tok/s/GPU **1948.78 -> 2262.06
(+16.1%)**, ahead of `iter1` at all nine concurrencies and ahead of `mtp1` and
`mtp2` at all nine. Against the matched-draft-length reference `trtllm-mtp3`
it leads by **1.270x at con=256** and is level (1.00-1.09x) at con=1..4 —
that, not the ratio to the non-drafting `trtllm` row, is what this iteration
is worth against stock. Decomposing throughput into the two terms it is made
of — the engine step now emits `accept` tokens instead of 1, and costs more —

| con | accept | step cost vs a non-drafting step | decode speedup | end-to-end |
|---|---|---|---|---|
| 1 | 3.4904 | 1.511 | 2.311x | 2.285x |
| 32 | 3.3356 | 2.198 | 1.518x | 1.301x |
| 256 | 3.4033 | 2.600 | 1.309x | 1.161x |

**Acceptance is essentially flat in concurrency and the whole decay is on the
cost side.** Across con 1..256 acceptance moves only 3.49 -> 3.40 (and
`mtp1` 1.967 -> 1.949 of 2.0, `mtp2` 2.748 -> 2.754 of 3.0) — never below 2.86x
the 1.164 break-even. The step cost, which the weight-bytes model says
should sit flat at 1.164, instead climbs **1.511 -> 2.600**. Reading the
marginal cost of each successive draft step at con=256 — **+0.645, +0.569,
+0.386** — against con=1's **+0.200, +0.173, +0.137** shows what it is: the
term that grows is proportional to the *rows* a step carries, not to the MTP
layer's fixed weight bytes, and the marginal cost falls with each further draft
step in the same way the marginal row count does (1 -> 2 rows is +100%, 2 -> 3
is +50%, 3 -> 4 is +33%). That is the memory-bound-to-compute-bound crossover
the mechanism doc predicted, measured. It never becomes steep enough to
overtake acceptance, which is why the curve is still climbing at 3.

*Gate.* No new accuracy run was needed and none is claimed: the paired gsm8k
record under *The MTP variant's own gate records* was measured on
**`configs/mtp3.yaml` itself**, and `modeling.py` has not changed since
(`4194dcb`, and this campaign added no code). identity **95.0720** / mtp3
**95.2237**, Δ **+0.1517** against `|Δ| < 1.2`. `mtp1` and `mtp2` are not
separately gated and are not the recommended setting; they are the two
measured points that establish the axis is monotone, and the note in each file
says so. What the gate certifies is that the variant did not break the model —
rejection sampling makes even a miscomputed draft layer emit correct text, so
`acceptance_length` against stock trtllm, not the score, is what says drafting
works.

### Rejected, with what was actually measured

* **`max_seq_len: 2048` — rejected, and it retires a blocked knob.**
  `kv_cache_config.tokens_per_block: 64` is uncertified over this target's
  fp8 latent pool, and it was worth **+22% at con=256** on the
  `deepseek-v3-lite-nvfp4/sm_100/tp1` sibling, so it is the knob a campaign
  here would want most. `max_seq_len` reaches the *same*
  `max_blocks_per_seq` reduction by the other route (5120 -> 64, an 80x cut
  where page 64 gives 2x), and measured **−0.13% at con=32 and +4.08% at
  con=256** against a paired probe — where a clean re-measure puts the same
  point at +0.10%. The profile says why: **GPU idle is 1.49%**, so there is
  no exposed host bookkeeping to recover. The tp1 sibling's win came from a
  host-bound decode (GPU idle 0.401); this target is 40x larger per step and
  the same host work is entirely hidden. **A vocabulary request to certify
  page 64 over an fp8 pool would not pay for itself here.**
* **Ragged collectives on the non-captured steps — rejected.** Under
  attention DP a mixed step (one rank prefilling beside three decoding) pads
  every rank to `max(counts)`, so the expert call runs over `4 x max` rows of
  which ~70% are zeros. `_dp_rows` was changed to return a `sizes` vector
  whenever the counts disagree and to pass it to both collectives (the
  uniform form is unreachable-by-construction under capture, where counts are
  equal). The engine does publish true per-rank counts —
  `_get_all_rank_num_tokens` is a plain `tp_allgather` of
  `attn_metadata.num_tokens`, no padding — so the branch is reachable, and
  smoke passed. It measured **+0.23% at con=32 and +4.08% at con=256**
  against the same perturbed baseline, i.e. **+0.10% against the clean
  cluster**: 7634.91 against 7635.26 (`max_seq_len` probe) and 7627.45
  (full baseline). The padded rows are real but are not a material share of
  GPU time on this workload; the assembler's uniform-only choice stands.
* **`stream_interval`, `cuda_graph_config.*`, `kv_cache_config.*`,
  `scheduler_config.capacity_scheduler_policy` — not swept, on profile
  evidence.** GPU idle 1.49% leaves nothing for the host-side response path
  to recover; graph replay coverage is already 100% at `enable_padding:
  false`, so padding could only coarsen the grid (−2.55% at con=256 on the
  `dep4` sibling); and the KV pool is 44.76 GiB = **1,368,000 tokens per
  rank** against the 64 requests x 2048 tokens = 131,072 a rank holds at
  con=256, a 10.4x margin, so capacity cannot bind.
* **`speculative_config.draft_len_schedule` — rejected twice over: the premise
  is refuted, and on this target it deadlocks.** What was measured:
  `configs/mtp3-sched.yaml`, `max_draft_len: 3` with
  `draft_len_schedule: {2: 3, 16: 2, 128: 1}` (thresholds are per-*rank* batch
  size, so under attention DP at dep4 they map to con 1..8 / 16..64 /
  128..256), full sweep attempted on GPUs 3-6.

  *The premise.* The schedule's case is "long drafts at low concurrency,
  shorter at high", which needs the best draft length to fall as batch grows.
  The three full sweeps say it does not: `mtp3` leads at **all nine**
  concurrencies, and acceptance decays by only 3.49 -> 3.40 over the whole
  range. There is no crossover inside the certified range, so a schedule that
  shortens the draft at high batch can only give throughput away.

  *The deadlock.* The run never produced a point. Engine build, CUDA-graph
  capture (one graph per `(batch_size, draft_len)` pair — the log shows
  `batch size=3..16, draft_len=2` and `batch size=1,2, draft_len=3`) and warmup
  all succeeded; roughly a minute into serving, **all four ranks hung and
  trtllm's own HangDetector fired at 300 s and hard-killed via `MPI_Abort`**.
  The stacks put the ranks at three *different* points of one forward — the q
  up-projection, the MLA generation call, and the MoE `fp4_quantize` — i.e. not
  in lockstep. The mechanism is structural: `_handle_dynamic_draft_len`
  resolves `runtime_draft_len` from **`scheduled_batch.batch_size`, which is
  each rank's own local batch**, with no cross-rank reduction, and attention DP
  does not equalize batch sizes — `_pad_attention_dp_dummy_request` only tops a
  rank up from zero to one. Two ranks either side of a threshold therefore run
  different numbers of MTP replays, hence different numbers of MoE collectives,
  and the job wedges. Nothing validates the combination at config time.

  *And it is not a single-variable comparison anyway*: setting the schedule
  makes the runtime log `Automatically enabling cuda_graph_config.enable_padding
  because draft_len_schedule is set`, flipping a knob this target measured as
  costly (padding can only coarsen an already 100%-covered replay grid). The
  variant file was deleted; this entry is its record.
* **`--acceptance` on the Pareto sweeps — rejected, and not needed.** The flag
  sets `enable_iter_perf_stats: true` and lifts `iter_stats_max_iterations`, so
  a label carrying it is not comparable to `baseline` / `iter1`. Priced in a
  paired probe on one `mtp3` config, back to back, con 1/32/256:
  **−1.39% / −3.52% / −3.96%** (212.34 -> 209.38, 2056.91 -> 1984.44,
  8229.25 -> 7903.35 tok/s). That is far outside the sub-1% floor, so it was
  carried on no sweep. It is also unnecessary here: the benchmark client
  already reports `avg_decoded_tokens_per_iter` per request, sourced from the
  **response body** rather than the `/metrics` iteration-stats stream, so it
  survives with the flag off — at con=1 the two probes recorded a bit-identical
  `3.8863`, and against the engine's own `acceptance_length` the client proxy
  agrees to **0.99-1.02x**. Every `accept` number in this section is that free
  proxy, measured at zero config cost on the same sweeps as the throughput.
* **`max_draft_len` 1 and 2 — measured and dominated.** `mtp1` and `mtp2` are
  full sweeps in the tables above; `mtp3` beats both at every concurrency. They
  are kept as configs because they are the evidence that the axis is monotone,
  not as recommended settings.

### Caveats

* **Session variance is measured, and the dominant term is co-tenancy, not
  sampling.** Three unchanged-config measurements of con=256 landed at
  7627.45 (full sweep, 1280 requests), 7635.26 and 7634.91 (two-round probes,
  512 requests) — a **0.10% spread across two different request counts**. A
  fourth, `probe-baseline`, read 7335.94 (**−3.9%**); an 8-GPU neighbour
  holding 1.9 GiB on GPUs 0-6 — including this campaign's device set — was
  caught in `nvidia-smi` minutes later and was gone within two. That probe is
  the only contaminated measurement in the campaign and no kept result rests
  on it. **Read the noise floor as well under 1% between clean runs, with a
  transient co-tenant worth ~4%.**
* **Probes and full sweeps are not interchangeable at the throughput end for
  TTFT.** The same config measured mean TTFT 1765.6 ms over 1280 requests and
  2279.7 ms over 512: with only two waves of 256 the first wave's queue
  dominates the mean. Throughput is insensitive to this (0.10% above); TTFT
  is not. Every TTFT comparison above is probe-to-probe or sweep-to-sweep.
* Host compute-apps and load average were recorded before and after every
  label (`logs/hoststate/`); GPUs 3-6 carried no other tenant for any
  measured curve.
* Both identity-config curves are single full sweeps; `iter1`'s two endpoints
  were additionally reproduced in a paired probe before the sweep was run.
* `perf/data/` is machine-local and not tracked; the figure is.
* **The two sessions are stitched by measurement, not by assumption.** Opening
  the 2026-08-02 session, `probe-anchor-iter1` re-measured the `iter1` config
  and landed within **−0.14% (con=1) and +0.15% (con=256)** of the 2026-08-01
  full sweep. At wrap-up the `trtllm` reference was spot-checked the same way:
  **−0.01% (con=1, TPOT bit-identical at 13.525 ms) and +0.20% (con=256)**.
  Neither curve was re-swept.
* **This session's noise floor, measured rather than inherited.** `mtp1` at
  con=256 read **8737.78** in its full sweep and **8736.8** in a re-measure
  2.5 hours later at the same request count (`probe-recheck-mtp1`) —
  **0.01%**. The same pair at **con=128 differs by 4.2%** (5877.32 vs 5630.1),
  so con=128 is this target's noisy point and no claim above rests on it
  alone. The `mtp3`-over-`mtp2` margin at con=256 is +3.29%, comfortably
  outside the con=256 floor.
* **Probes and full sweeps are much further apart under MTP than without it,
  and it is throughput this time, not just TTFT.** The same `mtp3` config
  measured con=256 at **8229.25** over 512 requests and **9048.23** over 1280 —
  **+9.95%** — where the identity config's probe-to-sweep gap at that point is
  0.15%. A two-round probe under-reports MTP because the ramp and drain, where
  batches are small and drafting is least profitable, are a much larger share
  of a 512-request point. **Every MTP comparison in this section is
  sweep-to-sweep**; an early probe-vs-sweep reading of these curves inverted
  the `mtp1`/`mtp3` ordering at con=256 before the full sweeps corrected it.
* **Acceptance at con=1 is prompt-dependent and needs more than a two-request
  probe.** The `mtp3` full sweep (20 requests at con=1) reads **3.4904**, while
  the two-request probes recorded under *The MTP variant's own gate records*
  read 3.8863/3.9274. Both are far above the 1.164 break-even, so the
  correctness reading there is unaffected, but the sweep number is the better
  estimate of acceptance.
* **A cold host page cache can push the stock-trtllm boot past the harness's
  900 s limit, and it looks like a failure rather than a slow start.** The
  first `trtllm-mtp3` attempt returned `[perf] FAILED: server not healthy
  after 900s`. It was not hung — it had reached `[Autotuner] Autotuning
  process starts` and was still progressing. The time went into reading the
  397 GB checkpoint: the log shows 101 s, 57 s and 38 s gaps between
  `Finished prefetching ...` lines, and boot took **837 s to reach the
  autotuner** against **220 s** for the same config earlier the same day.
  Re-reading the 163 shards took 7 s (~56 GB/s, i.e. already resident, the
  failed boot having warmed them), and the retry reached the autotuner in
  **219 s** and completed the full sweep. **Read a 900 s boot timeout on this
  checkpoint as a page-cache miss first**; warm the shards and retry before
  suspecting the model. Nothing in the harness was changed.
* **Host CPU load moved a lot and the measurement did not.** Load average over
  the campaign ranged 0.31 to 13.72, and a neighbour ran on GPUs 1-2 at 100%
  during wrap-up; the `trtllm` spot-check taken under the *heaviest* load still
  reproduced to 0.20%. The previous campaign's −3.9% contamination came from a
  co-tenant **on the device set itself**, which is the case to keep watching.

### Regenerating the figure

The committed figure plots exactly seven curves — the two **aligned**
stock-trtllm references and the five staircase sweeps — in this order:

```
uv run utils/plot_pareto.py \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/trtllm-aligned \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/trtllm-aligned-mtp3 \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/baseline \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/iter1-shared-side-stream \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/mtp1 \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/mtp2 \
    targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/data/mtp3 \
    -o targets/deepseek-r1-0528-nvfp4/sm_100/dep4/perf/figures/pareto.png
```

**Explicit paths, not the `perf/data` directory.** That directory also holds
the two acceptance-only gate-record labels and the three superseded
boot-forced reference lines; expanding it would put all of them on the figure.
The committed figure plots the two aligned references plus the five staircase
curves.

**`trtllm-nodeepep` is on the figure on purpose even though it is only three
points.** It is the fastest stock configuration measured here, and it sits
almost on top of `baseline` / `iter1`; leaving it off would let the figure
imply a larger no-MTP lead over stock trtllm than exists.

**The paths are explicit on purpose: pointing the plotter at `perf/data`
wholesale is wrong here.** That directory also holds `probe-mtp3-acceptance`
and `trtllm-mtp3-acceptance` — a two-point probe and an acceptance-only
reference from the MTP gate records, neither of them a Pareto curve — and the
plotter picks up every subdirectory carrying a `meta.json`, which would put a
second grey reference line on the figure and blow past its categorical slots.

The `trtllm` curve needs its boot config:
`uv run bench/perf.py --target targets/deepseek-r1-0528-nvfp4/sm_100/dep4
--label trtllm --trtllm --config
targets/deepseek-r1-0528-nvfp4/sm_100/dep4/configs/trtllm-ref-boot.yaml`.
The three MTP curves are
`uv run bench/perf.py --target <t> --label mtp<N> --config <t>/configs/mtp<N>.yaml`
— no `--acceptance`.

### Remaining headroom

* **The `max_draft_len` curve is still climbing where the certification
  stops.** `mtp3` is the best of the three at every concurrency and the gain
  from 2 to 3 is still positive everywhere (con=1 162.51 -> 187.12, con=256
  8760.17 -> 9048.23), while acceptance holds at 3.40 of a 4.0 ceiling — 2.9x
  the 1.164 break-even. Nothing in the *data* says 3 is the optimum; 3 is the
  top of what `catalog/attention/mla_rope_generation` and
  `catalog/attention/thop_attention` certify, which is
  `predicted_tokens_per_seq` ∈ {1,2,3,4} and hence `max_draft_len` ≤ 3. **A
  campaign that wants 4 or beyond needs those two entries certified at
  `predicted_tokens_per_seq` 5+ first — a vocabulary request, not a bigger
  number in a config.** The step-cost series says the return is decelerating
  (marginal cost per draft step at con=256: +0.645, +0.569, +0.386 against
  acceptance gains of +0.95, +0.80, +0.65 tokens), so the crossover is probably
  not far past 3, but it is unmeasured.
* **The `mtp3` cost decomposition is arithmetic, not a timeline.** The
  memory-bound-to-compute-bound reading above is derived from the step-cost
  series across four draft lengths and nine concurrencies; no nsys capture was
  taken under MTP. A timeline at con=256 with `max_draft_len: 3` would say
  *which* kernel family absorbed the growth — the routed expert GEMMs at 4x
  rows, the bf16 MTP expert stack, or the extra collectives — and that is the
  next thing to profile if the MTP path is tuned further.
* **The collectives are the one identified lever and they are blocked on
  vocabulary.** **Corrected 2026-08-03:** the 3369.7 µs / 94.3%-exclusive
  figure below is the **pre-`iter1`** profile — it is the measurement that
  motivated the side stream ("a 3.37 ms window with nothing in it"), not the
  state after it. A post-`iter1` capture at the same window and phase measures
  **1930.7 µs per rank per decode step at 55.2% exclusive**: the side stream
  moved the shared expert into that window, so the recoverable time is 57% of
  what this paragraph claimed. Sizing a transport change off 3369.7 overstates
  it. The pre-`iter1` numbers, kept because the rest of the analysis rests on
  them: **3369.7 µs per rank per decode step of exclusive GPU time, 11.9% of
  the step, at 94.3% exclusive**, in two
  `RING_LL` NCCL calls per MoE layer. `comm/allgather` and
  `comm/reducescatter` expose **no strategy argument and no workspace**, so
  the ONESHOT swap that won 2.03x per call on the `tep4` sibling does not
  exist here. Closing it needs new catalog entries: a strategy-carrying
  gather/scatter, or the `moe_a2a_dispatch` / `moe_a2a_combine` family (a
  stateful workspace pair, so `moe_a2a_initialize` and
  `moe_a2a_get_combine_payload_tensor` come with it). Note `iter1` has
  already spent part of this window — a transport change would now compete
  with the shared expert for it, so the two do not simply add.
* **Reducing collective *bytes* is not the lever.** At con=256 each rank's
  decode batch is 64 rows, so one all-gather receives `3 x 64 x 7168 x 2` =
  **2.753 MB** in its measured 25.472 µs = **108 GB/s**, an order of
  magnitude under NVLink: `RING_LL` at these sizes is latency-bound, not
  bandwidth-bound. Gathering NVFP4 activations instead of bf16 (4032 vs
  14336 B/token, **3.56x** fewer) would buy nothing and would cost extra
  calls.
* **The routed expert GEMMs are near the HBM roofline and are the floor.**
  46.3% of GPU busy, 98.9% exclusive. With top-8 of 256 experts over 256
  gathered tokens every one of a rank's 64 local experts is active, so a step
  reads the whole local stack: FC1 (gate+up) is `2 x 2048 x 7168 x 64` at
  0.5625 B/element (NVFP4 data plus the fp8 block scale) = **1.057 GB in
  155.25 µs = 6.81 TB/s**; FC2 (down) is 0.528 GB in 77.97 µs = **6.78
  TB/s** — ~85% of a ~8 TB/s B200 roofline. No config knob and no scheduling
  change moves this; only fewer weight bytes would.
* **The bf16 attention GEMMs are the second floor**: 20.1% of GPU busy. The
  largest, at 45.21 µs x 61 layers per rank-step, reads `o_proj`'s
  16384x7168 bf16 (235 MB) at **5.19 TB/s** — the one family with visible
  daylight to the roofline, but it is cuBLAS's tactic choice, not ours.
  These weights are bf16 by checkpoint design (`hf_quant_config.json`
  quantizes the MLP, not attention), so this is a checkpoint property rather
  than a target one.
