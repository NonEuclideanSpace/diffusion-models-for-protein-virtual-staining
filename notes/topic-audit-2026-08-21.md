# 课题审计报告：Diffusion Models for Protein Virtual Staining

**审计对象**：`notes/core-claims.md`（2026-08-22 版）
**审计日期**：2026-08-21
**审计方式**：核心文献逐条外部核实（bioRxiv / arXiv / Nature / Science / OUP / CCF 目录原文检索），共四路定向检索 + 全文抓取核对引文原话
**总结论**：**课题成立，空白真实，动机自然，不需要降级。** 所有引用的文献中仅一处年份错误、一处引用无法核实、一处归因错位。最大的科学风险不在"编造"，而在 (i) C1 效应量级未知且与细胞周期混杂缠斗，(ii) C2 的 R≈0 是可预期结论、需靠对照臂设计才能避免"无用的负面报告"。两条风险都有明确的、不降级课题的解法，见第 5 节。

---

## 1. 查重结果：逐条核实

### 1.1 模型与评测文献（C2 侧）

| core-claims 引用 | 核实结果 | 备注 |
|---|---|---|
| CELL-Diff, ICML 2025 | ✅ 真实 | bioRxiv 2024.10.15.618585；ICML 2025 poster 确认。指标确为 MSF/IoU/FID，**无多样性指标** |
| PUPS, Nature Methods 2025 | ✅ 真实 | 22:1265–1275，DOI 10.1038/s41592-025-02696-1，Zhang/Tseo/Bai/Chen/Uhler。报 Pearson/MSE 类指标 |
| ProtiCelli, Lundberg 组, bioRxiv 2026-04 | ✅ 真实，原话核对通过 | bioRxiv 2026.03.31.715748（v1 2026-03-31，v2 2026-05-13）。"idealized, canonical localizations" 与 "generalized consensus map" 均为原文原话；generative precision/recall 确在 SubCell 嵌入空间、高 precision 低 recall。**重要细节：其 P/R 是在 protein-level 平均嵌入（200 个生成细胞池化）上算的，比 core-claims 所述更 marginal——缺口论证只会更稳** |
| Tonks et al., arXiv 2602.23305 | ✅ 真实，但标题需更正 | 实际标题 **"A Proper Scoring Rule for Virtual Staining"**（Samuel Tonks 等，2026-02-26），IG（information gain）是其中提议的 strictly proper scoring rule。确为 per-cell conditional posterior 评估，brightfield→荧光（HTS/Cell Painting 体系，手工特征），不是蛋白定位 |
| In Silico Labeling / Ounkomol / Cytoland | ✅ 真实 | Christiansen Cell 2018;173:792；Ounkomol Nat Methods 2018;15:917；**Cytoland = Liu, Hirata-Miyasaki, Mehta, Nat Mach Intell 2025;7:901–915**（建议补全正式引用），报 PCC/SSIM/IoU/AP |
| CELL-E 2（英文摘要提及） | ✅ 真实 | NeurIPS 2023，HPA+OpenCell |
| Naeem ICML 2020；Sajjadi NeurIPS 2018 | ✅ 真实 | 另建议补 Kynkäänniemi NeurIPS 2019（improved P/R），这是该指标族的标准引用 |
| **deepGPS** | ⚠️ **core-claims 漏列** | Yuan et al., **Briefings in Bioinformatics 2025;26(2):bbaf152**，ESM2+U-Net 生成定位图像，报分类指标（Acc/F1/AUPRC），**同样不报多样性指标**。纳入后"无一报多样性/覆盖度"的结论仍成立且更完备。这是 Briefings 的论文，Bioinformatics 审稿人大概率知道它，漏掉会被当成检索不完整 |

### 1.2 生物学文献（C1 侧）

| core-claims 引用 | 核实结果 | 备注 |
|---|---|---|
| Snijder Nature 2009 | ✅ 真实 | Nature 461:520–523。**74% / 82% / CV 0.78→0.31 三个具体数字本次未能从原文核到**（付费墙），投稿前必须回原文/2011 NRMCB 综述核对 |
| Snijder & Pelkmans NRMCB 2011；Snijder MSB 2012 | ✅ 真实 | 12:119–125；8:579 |
| Gut Science 2018 (4i) | ✅ 真实 | 361:eaar7042，40-plex，确含 local crowding → 细胞器组成分析，未报方差解释比例 |
| Viana Nature 2023 (Allen) | ✅ 真实 | 613:345–354，25 结构，>20 万活细胞，mEGFP 敲入（构造性真值），元数据 ~1,200 列且含集落边缘注释——core-claims 对外部验证集的描述全部属实 |
| Elmalam & Zaritsky, CELTIC, **Nat Methods 2026** | ⚠️ **年份错误** | 实际为 **Nature Methods 2025;23(2):405–416**（bioRxiv 2024.11.10.622841）。其 WTC-11 数据集 **S-BIAD2156 自带预计算 context 特征**——对本课题是现成资产，见第 5 节 |
| Sero et al. 2015 | ⚠️ **归因错位** | Sero et al., MSB 2015;11:790 是 **NF-κB** 的核转位受细胞形状/微环境调控（cell-cell contact、面积、protrusiveness 均有贡献）。**YAP/TAZ 的密度依赖核质穿梭应引 Aragona et al., Cell 2013**（mechanical checkpoint, actin-processing factors）。当前写法把两个蛋白都挂到 Sero 2015 上 |
| Li et al. 2023（TIPARP/PTGES3/CBFB/SMAD4） | ✅ 真实 | Li, Li & Nakamura, **Sci Rep 2023;13:21723**，四个密度依赖核质穿梭蛋白确认无误 |
| Frechin 2015（铺展/拥挤） | ✅ 真实 | Nature 523:88–91 |
| Mahdessian 2021（FUCCI, 298 interphase-CCD） | ✅ 真实 | Nature 590:649–654；HPA 官网明确：298 个 FUCCI 测量 CCD + 479 个"定位于有丝分裂结构"的定义性 CCD = 748。**core-claims 第 4 条证据（下载版 `Cell cycle dependency` 列只含定义性标签）与 HPA 官方描述结构一致，该审计发现成立** |
| DBSCAN-CellX（Küchenhoff 2023） | ✅ 方法存在 | Metz-Zumaran et al. 2023 确用其 edge-degree 做 IFN 研究；角隙阈值 120–160° 建议回原文核对 |
| **Le et al., Cell Systems 2026**（"12,000 蛋白、862k 单细胞、形状解释定位变异"） | ❌ **无法核实** | 两次定向检索（Cell Systems / bioRxiv 2026，按人数与细胞数检索）均无此文。**这是全文唯一找不到的引用，必须补全准确书目（作者、标题、DOI），否则从对比表中删除**——审稿人查到一条不存在的"最接近工作"会直接摧毁全文可信度 |

### 1.3 期刊与事实性声明

| 声明 | 核实结果 |
|---|---|
| CCF 第七版 Bioinformatics 由 B 升 A（交叉/综合/新兴类） | ✅ 属实。第七版交叉类 A 类期刊第 4 位为 Bioinformatics（OUP）；第七版 2026 年发布、4 月 9 日有勘误页，与"2026-03-31 发布"吻合 |
| Bioinformatics 设 Bioimage informatics 栏目（2012 社论） | ✅ 属实。Peng et al. 2012 社论确立该类别，覆盖"显微图像的获取/分析/挖掘/可视化的信息学方法" |
| Briefings B、PLOS CB B、ISMB CCF-B 会议 | ✅ 均与第七版目录一致 |
| HPA 规模（~113 万单细胞图、12,000+ 蛋白） | ✅ 量级正确（ProtiCelli 用 1.23M 图/12,800 蛋白/39 细胞系；SubCell 同源） |
| Kaggle 2021 逐细胞标注未公开 | ✅ 与 SubCell 论文一致（其自称在 "hidden test set" 上评估） |

### 1.4 查重总结论（回答"是不是已经被做过了"）

1. **C1（population context → 亚细胞定位，proteome 尺度）：未发现撞车。** 最接近的四项确实各差一步（Gut 40 通道、Viana 25 结构/一种细胞、CELTIC 为预测而非评估、PUPS 用隐式形态而非显式邻域变量）。**否定性结论在现有检索下成立**，但有一个必须处理的邻近事实：**PUPS 已证明"单细胞定位变异不仅来自随机性，且可由 landmark 形态预测"**。C1 必须与这句话正面切割：PUPS 的协变量是细胞自身形态（隐式、未分解），C1 要测的是**显式邻域变量在固定视野内的贡献份额**。不切割，审稿人会问"这不就是 PUPS 的 Fig. 4 换了个协变量吗"。
2. **C2（conditional diversity / 协变量效应保留）：核心缺口成立，但措辞必须精确化。** "没有人做过 conditional diversity" 在字面上会被 Tonks 击穿——Tonks 做的就是 per-cell conditional posterior 评估。准确的说法是：**Tonks 评的是逐细胞后验校准（calibration，strictly proper scoring）；没有人测过"定位对 context 的条件依赖是否被生成模型保留"（effect retention）**。R = β̂/β 属于后者。另外，"效应保留"作为一个概念族在合成表格数据/医学影像文献中已有先例（合成数据保真 vs ATE 保留，arXiv 2604.23904；脑 MRI 合成数据的 β 系数对比）——**在生物成像/蛋白定位领域 R 确实是第一个，但写作时必须引用这个概念族**，否则"first ever"式表述会被 ML 审稿人纠正。
3. **HPA 变异注释外部审计（证据 3、4）**：本次检索同样未发现任何独立组做过，否定性结论可保留（措辞保持"据我们所知"）。

---

## 2. 生物学意义：真问题还是强行编造？

**判定：真问题，且机制前提扎实，不是编造。**

- "定位部分由邻域决定"有独立机制证据链：NF-κB 核转位受细胞接触/形状调控（Sero 2015）、YAP/TAZ 密度依赖穿梭（Aragona 2013）、四个密度依赖穿梭蛋白（Li 2023）、拥挤改变细胞器组成（Gut 2018）、集落边缘极化（Viana 2023）。这些不是统计联想，是有通路的机制（机械转导、接触抑制）。
- C1 的增量是**尺度与定量**（10³–10⁴ 蛋白 × 方差/互信息份额），这恰是 Pelkmans 脉络 17 年没走完的一步。空白真实（见 1.4）。
- "使能条件"（每张 HPA 裁剪图自带邻域、72% 亮核像素属邻居）是聪明的资源再利用，动机自然——"输入里有、流水线丢了、而生物学说这东西有用"，三句话都站得住。

**但要诚实指出两个"不自然"风险点：**

1. **C1 的最大敌人是细胞周期混杂，不是邻域生物学。** HPA 里 3,752 个基因有单细胞变异，其中 298 个已被 FUCCI 证明是周期驱动；密度与播种时间/周期分布天然耦合。固定视野（field）只能消去视野间差异；**同一视野内"局部密度仍与细胞周期相位相关"的可能并未被设计消去**（视野内周期相位不同步程度有限但非零）。审稿人第一个问题必然是"你的 I(context; localization | field) 里有多少是周期？"。必须在估计量层面内置 DAPI 衍生的周期相位作为部分协变量（partial MI / 混合模型中作固定效应），而不是等审稿人问。
2. **一句话动机存在一个结构性的细微矛盾**：C2 说"模型输入里有邻居却没用，这是罪状"；而 SubCell 官方配方恰恰要求先掩膜"to remove the effects of surrounding cell regions"——说明社区隐约知道邻域是干扰/泄漏源，掩掉它分类才准。这不推翻主张（分类器掩邻居是为了不受训练标签广播的干扰；生成模型不利用邻居是丢掉了真实的条件依赖），但写作时必须把这个 nuance 摊开，否则显得在左右互搏。建议把 C2 的措辞精化为：**"模型复现了边缘分布，但没有保留定位对 context 的条件依赖"**——这是统计陈述，不再有"看见/没看见"的拟人化歧义。

---

## 3. 是否有亮眼结果，还是无用的负面报告？

**当前证据包的性质：方法学取证（forensics）很强，生物学正结果为零。** 五条已到手证据里，四条（85% between-field、劈半 0.894、AUC 0.622 元数据泄漏、CCD 列缺陷、掩膜配方）都是"关于测量系统的发现"，一条（CDC20 周期群体验证）是小规模生物学。**靠这些投稿，论文会读成一份高质量的负面/审计报告——这不是 Bioinformatics 论文该有的重心，且正中你担心的"无用负面报告"。**

**亮眼结果的来源只能有三个，按优先级：**

1. **C1 的阳性地图**（论文主图）：哪些蛋白的定位是 context 依赖的、富集什么功能、效应量多大。这是"首次 proteome 尺度"的发现型结果，是真正的 headline。**前提是阳性对照先跑通**（YAP1/WWTR1、RELA、TIPARP/PTGES3/CBFB/SMAD4——注意先查 HPA 里这些蛋白的视野数是否够用）。阳性对照不过，全案不得推进到 proteome 尺度——core-claims 自己也是这么定的，坚持执行。
2. **C2 的对照臂设计**（把 R≈0 从"必然结论"变成"有判别力的测量"）：**把 CELTIC 式 context-conditioned 模型作为阳性模型对照臂**——一个显式拿到 context 的模型 R 应当显著大于 0。这一臂同时证明两件事：(a) R 这个指标测得出东西（不是地板效应）；(b) "抹平"不是宿命而是架构选择。有这一臂，论文从"所有模型都失败"变成"这里有一把能把 context-using 和 context-ignoring 模型区分开的尺子"——**同样的数据，叙事从负面翻成正面**。
3. **社区可带走的资产**：HPA CCD 列勘误 + SubCell 掩膜注意点 + R 指标开源包。这是 Bioinformatics 编辑文化里加分的"utility"部分，但只能当配菜。

**一个被低估的稳健性**：这个课题在两种结局下都可发表——若 R≈0，是评测工具 + 警示；若某模型 R>0（比如 CELTIC 臂），是"哪类条件化保留了生物学依赖"的更大新闻。这不是赌博式课题。

---

## 4. 回答 core-claims 的三个自问

**Q1（关系型估计量的抵消论证）**：方向正确，但"φ 的系统偏差两臂抵消"只在**加性、与 context 不交互、且 φ 的响应曲面在两臂间不变**时成立。已识别域漂移是对的，再补两个失效模式：
- **衰减混淆**：生成图更平滑 → φ 在生成臂的信噪比更低 → β̂ 因测量误差衰减而偏小，R<1 里混进了"衰减"和"真实 flattening"两个分量。解法：degraded-real 对照臂（把真实图退化到生成图的平滑度/亮度分布后重测 β，作衰减基线），core-claims 已计划，升级为必做。
- **φ 粒度地板效应**：31 类词表对核质比连续变化的分辨率未验证时，β 本身可能≈0，此时 R 无意义。**R 只在 C1 判定为显著 context 依赖的蛋白子集上报告**——这也把 C1/C2 的逻辑依赖变成了设计而非漏洞。

**Q2（C1 是否已被做过）**：见 1.4，未发现撞车；要处理 PUPS 的邻近声明与 Tonks 的字面冲突，措辞按上文精化。

**Q3（Bioinformatics 是否合适）**：合适，且是最优解——Bioimage informatics 栏目、CCF-A、接受 bioRxiv、读者群正好是"会用虚拟染色模型的人"。但**要按该刊的方法论文文化写**：可复现软件、明确 utility、公平的模型对比表、局限性诚实清单。落在两个共同体缝里的风险，用 WTC-11 真值臂（生物学审稿人）+ 正式指标定义与统计检验（ML 审稿人）各堵一端。

---

## 5. 提升意见（全部面向 Bioinformatics，不降级课题）

**A. 叙事结构：主打 C1 的地图，C2 作第二节， forensic 证据进补充。**
建议标题量级："Population context shapes subcellular protein localization at proteome scale, and virtual staining models erase it"。第一节是 C1 的阳性地图（哪个蛋白、哪个 context 轴、多少份额）；第二节是 R 指标与模型审计（含 CELTIC 阳性臂）；第三节是社区资产（CCD 列、掩膜配方）。这个顺序把论文从"评测批判"变成"生物发现 + 方法工具"，与 ProtiCelli/Tonks 都错开身位。

**B. 阳性对照组先行的硬门控。** YAP1/WWTR1、RELA、TIPARP/PTGES3/CBFB/SMAD4 在 HPA U2OS 上必须先复现密度依赖方向；同时报告 φ 对核质比的剂量-响应分辨率（合成混合实验）。门不过则收缩到 C2 工具论文（不降级课题，只降级叙事）。

**C. WTC-11 外验臂升级为与 HPA 并列的主结果。** CELTIC 的 S-BIAD2156 自带预计算 context 特征，Viana 元数据含集落位置与周期相位，25 个结构是 mEGFP 构造性真值——**这一臂完全不经过 φ，是对"分类器循环性"质疑的釜底抽薪**。Bioinformatics 审稿人最怕的就是 φ 循环论证，这条臂把它变成论文最强点。

**D. 模型阵容补齐 + 一个阳性模型臂。** CELL-Diff、PUPS、ProtiCelli、CELL-E 2、deepGPS、CELTIC（阳性臂）+ 自有模型。ProtiCelli 来自 Lundberg/HPA 生态圈，审稿人极可能是其作者群——全文对其保持建设性语气，强调"我们把他们 pooled 的测量推进到 conditional"。

**E. 主估计量从 MI 换成斜率族，MI 作辅助。** 你们自己的 rehearsal 已证明 plug-in KL 在 HPA 样本量下失效（B1/D6）；主结局用正则化回归斜率 + field 随机效应 + BH-FDR + 效应量 CI，MI 用 k-NN/分箱 + field 内置换作辅助验证。这同时让 R（斜率之比）与 C1 共享同一套估计框架，论文内部自洽。

**F. 把 R 包装成可复用工具。** 命名（如 context-effect retention ratio）、开源包、Nextflow/Snakemake 流水线、数据卡；适用面写明不限于蛋白定位（任何条件生成的 bioimage 翻译任务）。Bioinformatics 对"别人能用的指标"有明显偏好，这也回应 Tonks——IG 与 R 是互补的两个轴（校准 vs 效应保留），论文里给出两指标的并列对比表，化潜在撞车为综述贡献。

**G. 统计细节预置。** 周期相位（DAPI/FUCCI 代理）作为部分协变量进模型；置换检验保持 field 结构；功效分析沿用 m1_power.py 传统并写进 Methods；所有效应量带 CI 而非只报 p。

**H. 引用修正清单（投稿前必须执行）：**
1. CELTIC → Nat Methods **2025**;23(2):405–416
2. Le et al. Cell Systems 2026 → **补全书目或删除该行**（当前无法核实）
3. Sero 2015 仅用于 NF-κB；YAP/TAZ 改引 Aragona et al. Cell 2013
4. Tonks 标题更正为 "A Proper Scoring Rule for Virtual Staining"
5. ProtiCelli 日期 → bioRxiv 2026.03.31.715748
6. 调查表加入 deepGPS（Brief Bioinform 2025;26(2):bbaf152）与 Kynkäänniemi NeurIPS 2019
7. Snijder 2009 的 74%/82%/0.78→0.31 回原文核对
8. Cytoland 补正式引用（Nat Mach Intell 2025;7:901–915）
9. "没有人做过 conditional diversity" → 改为 "conditional posterior *calibration* 已有先例（Tonks）；定位对 context 的*条件依赖保留*无人测过"

**I. 投稿节奏。** 先 bioRxiv；ICLR 2027 摘要（2026-09-18）太紧，C2 未开始，不建议；主投 Bioinformatics，fallback 保留 Briefings/PLOS CB。若 C1 地图够强，可考虑 ISMB 2027 轨道（proceedings 即 Bioinformatics），一石二鸟。

---

## 附：本次核实未覆盖项（诚实清单）

- Snijder 2009 三个具体百分比未能穿透付费墙核实（标记为待核，非判定错误）
- "此前无任何独立组审计过 HPA 变异注释"为否定性声明，本次四路检索未发现反例，但不能证无
- CELL-Diff 输出亮度 21.6%、72% 亮核像素、85%/76.4% 方差分解等为课题组内部测量，不在外部核实范围
- scholar 学术库接口本次不可用（后端连接失败），查重实际由通用网络检索完成；如需系统性查重留档，建议在服务恢复后用 scholar 重跑一遍并存 CSV
