<div align="center">
  <!-- 替换 href 为你的 header.svg 的真实路径，如果是在同级目录直接写文件名 -->
  <img src="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/blob/main/assets/header.svg" width="100%" alt="Awesome-Agentic-Clinical-Dialogue/blob/main/assets/header.svg" />

  <br/>
  <br/>

  <a href="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/stargazers">
    <img src="https://img.shields.io/github/stars/xqz614/Awesome-Agentic-Clinical-Dialogue?style=for-the-badge&logo=github&color=42d392" alt="Stars">
  </a>
  <a href="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/network/members">
    <img src="https://img.shields.io/github/forks/xqz614/Awesome-Agentic-Clinical-Dialogue?style=for-the-badge&logo=github&color=647eff" alt="Forks">
  </a>
  <a href="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/issues">
    <img src="https://img.shields.io/github/issues/xqz614/Awesome-Agentic-Clinical-Dialogue?style=for-the-badge&logo=github&color=a78bfa" alt="Issues">
  </a>

  <br/>
  <br/>

  <p align="center">
    <br />
    Welcome to Awesome-Agentic-Clinical-Dialogue. This repo includes papers about methods related to agentic clinical dialogue. We believe that the agentic paradigm is still a largely unexplored area, and we hope this repository will provide you with some valuable insights!
    <br />
    Read our survey paper here: <a href="https://arxiv.org/abs/2512.01453">Reinventing Clinical Dialogue: Agentic Paradigms for LLM‑Enabled Healthcare Communication</a> 
    <br />
    <a href="#-awesome-methods-model-and-resource-list">Papers</a> •
    <a href="#%EF%B8%8Fstart-with-awesome-dataset">Datasets</a> •
    <a href="#-leading-group">Leading Group</a>
  </p>
</div>

<p align="center"><img src="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/blob/main/image/nano.png" height="500px"></p>


## 📘 Overview
This framework facilitates a systematic analysis of the intrinsic trade-offs between creativity and reliability by categorizing methods into four archetypes: Latent Space Clinicians, Emergent Planners, Grounded Synthesizers, and Verifiable Workflow Automators. For each paradigm, we deconstruct the technical realization across the entire cognitive pipeline, encompassing strategic planning, memory management, action execution, collaboration, and evolution, to reveal how distinct architectural choices balance the tension between autonomy and safety. Furthermore, we bridge abstract design philosophies with the pragmatic implementation ecosystem. By mapping real-world applications to our taxonomy and systematically reviewing benchmarks and evaluation metrics specific to clinical agents, we provide a comprehensive reference for future development.
<p align="center"><img src="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/blob/main/image/taxonomy.png" height="500px"></p>

## 📁 Table of Contents
- [Key Categories](#-key-categories)
- [Start with Awesome Dataset](#%EF%B8%8Fstart-with-awesome-dataset)
  - [QA Dialogue](#i-qa-dialogue)
  - [Task-oriented Dialogue](#ii-task-oriented-dialogue)
  - [Recommendation Dialogue](#iii-recommendation-dialogue)
  - [Supportive Dialogue](#iv-supportive-dialogue)
  - [Hybrid-function Dialogue](#v-hybrid-function)
- [Leading Group](#-leading-group)
- [Awesome Methods, Model and Resource List](#-awesome-methods-model-and-resource-list)
  - [LSC](#lsc)
    - [Planning](#planning)
    - [Memory](#memory)
    - [Cooperation](#cooperation)
    - [Self-evolution](#self-evolution)
  - [EP](#ep)
    - [Planning](#planning-1)
    - [Memory](#memory-1)
    - [Cooperation](#cooperation-1)
    - [Self-evolution](#self-evolution-1)
  - [GS](#gs)
    - [Planning](#planning-2)
    - [Memory](#memory-2)
    - [Action](#action)
    - [Cooperation](#cooperation-2)
    - [Self-evolution](#self-evolution-2)
  - [VWA](#vwa)
    - [Planning](#planning-3)
    - [Memory](#memory-3)
    - [Action](#action-1)
    - [Cooperation](#cooperation-3)
    - [Self-evolution](#self-evolution-3)
- [Contributing](#-contributing)
- [Citation](#%EF%B8%8F-citation)
## 🔑 Key Categories
- 🤖**Latent Space Clinicians (LSC)**. These agents leverage the LLM's vast internal knowledge for creative synthesis and forming a coherent understanding of a clinical situation. Their philosophy is to trust the model's emergent reasoning capabilities to function like an experienced clinical assistant providing insights. For example, the zero/few-shot reasoning capabilities of Med-PaLM or MedAgents exemplify this paradigm.
- 🤖**Emergent Planners (EP)**. This paradigm grants the LLM a high degree of autonomy, allowing it to dynamically devise its own multi-step plan to achieve a complex clinical goal. The agent's behavior is emergent, as it independently determines the necessary steps and goals. Frameworks like AgentMD, which uses ReAct-style prompting.
- 🤖**Grounded Synthesizers (GS)**. These agents operate under the principle that LLMs should function as powerful natural language interfaces to reliable external information rather than as knowledge creators. Their primary role is to retrieve, integrate, and accurately summarize information from verifiable sources like medical databases or imaging data. Exemplars include the foundational frameworks medical retrieval and indexing techniques such as Med-RAG and MA-COIR.
- 🤖**Verifiable Workflow Automators (VWA)**. In this paradigm, agent autonomy is strictly constrained within pre-defined, verifiable clinical workflows or decision trees. The LLM acts as a natural language front-end to a structured process, executing tasks rather than making open-ended decisions, which ensures maximum safety and predictability. This approach is exemplified by commercial triage bots, the structured conversational framework of systems like Google's AMIE, and principles from classic task-oriented dialogue systems sush as MeDi-TODER.

## ✳️**Start with Awesome Dataset**
[Back to Content](#-table-of-contents)
### **I. QA Dialogue**
[Back to Content](#-table-of-contents)
| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **MedQA** | 2020 | Medical Examination (QA) | Large-scale multiple-choice questions collected from professional medical board exams (USMLE, Mainland China, Taiwan). | [paper](https://arxiv.org/abs/2009.13081), [source](https://github.com/jind11/MedQA/) |
| **MedMCQA** | 2022 | Medical Examination (QA) | Large-scale, multiple-choice QA dataset derived from Indian medical entrance examinations (AIIMS/NEET). | [paper](https://arxiv.org/abs/2203.14371), [source](https://github.com/MedMCQA/MedMCQA) |
| **cMedQA2** | 2019 | QA / Retrieval | Chinese medical QA dataset with queries and answers from online health counseling platforms. | [paper](https://ieeexplore.ieee.org/abstract/document/8548603), [source](https://github.com/zhangsheng93/cMedQA2) |
| **CMExam** | 2023 | Medical Examination (QA) | 60K+ multiple-choice questions from the Chinese National Medical Licensing Examination with detailed annotations. | [paper](https://arxiv.org/abs/2306.03030), [source](https://github.com/williamliujl/CMExam) |
| **Medbullets** | 2024 | Medical Examination (QA) | High-quality USMLE Step 2 & 3 style questions with expert-written explanations for reasoning evaluation. | [paper](https://arxiv.org/abs/2402.18060), [source](https://github.com/HanjieChen/ChallengeClinicalQA) |
| **HeadQA** | 2019 | Medical Examination (QA) | Multiple-choice questions from Spanish healthcare exams (MIR, EIR, etc.) for testing complex reasoning. | [paper](https://aclanthology.org/P19-1092/), [source](https://github.com/aghie/head-qa) |
| **Huatuo-26M** | 2023 | Medical Examination / QA | Massive Chinese medical QA dataset with 26 million QA pairs, used for pre-training and retrieval. | [paper](https://arxiv.org/abs/2305.01526), [source](https://github.com/FreedomIntelligence/Huatuo-26M) |
| **CasiMedicos-Arg** | 2024 | Medical Examination (QA) | Multilingual dataset (ES, EN, FR, IT) annotated with explanatory argumentative structures for clinical cases. | [paper](https://aclanthology.org/2024.emnlp-main.1023/), [source](https://github.com/ixa-ehu/antidote-casimedicos) |
| **PubMedQA** | 2019 | Literature-based QA | Biomedical QA task to answer "yes/no/maybe" from PubMed abstracts. | [paper](https://arxiv.org/abs/1909.06146), [source](https://github.com/pubmedqa/pubmedqa) |
| **CliCR** | 2018 | Literature-based QA | Dataset of clinical case reports designed for machine reading comprehension. | [paper](https://arxiv.org/abs/1803.09102), [source](https://github.com/clips/clicr) |
| **MEDIQA-2019** | 2019 | Literature-based QA | Shared task data focusing on NLI, RQE (Recognizing Question Entailment), and QA in the medical domain. | [paper](https://www.aclweb.org/anthology/W19-5001/), [source](https://github.com/abachaa/MEDIQA2019) |
| **BioASQ** | 2013-2023 | Literature-based QA | Long-running challenge series for large-scale biomedical semantic indexing and question answering. | [paper](http://bioasq.org/), [source](https://github.com/BioASQ) |
| **Medical Meadow** | 2023 | Literature-based / Tuning | A collection of various medical tasks (Flashcards, Wikidoc) reformatted for instruction tuning. | [paper](https://arxiv.org/abs/2304.08247), [source](https://github.com/kbressem/medAlpaca) |
| **MASH-QA** | 2020 | Consumer Health QA | Multiple-span extraction QA dataset for consumer health questions (e.g., from WebMD). | [paper](https://arxiv.org/abs/2004.03008), [source](https://github.com/mingzhu0527/MASHQA) |
| **HealthQA** | 2019 | Consumer Health QA | Dataset focusing on reliability and helpfulness of health answers. | [paper](https://arxiv.org/abs/1902.08726), [source](https://github.com/mingzhu0527/HAR) |
| **AfriMed-QA** | 2025 | Domain-specific QA | Pan-African medical QA benchmark (15k Qs) covering 32 specialties and local context from 16 countries. | [paper](https://arxiv.org/abs/2411.15640), [source](https://github.com/intron-innovation/AfriMed-QA) |
| **MedCalc-Bench** | 2024 | Domain-specific QA | Benchmark for evaluating LLMs on medical calculations (formulas, scores) with patient notes. | [paper](https://arxiv.org/abs/2406.12683), [source](https://github.com/ncbi-nlp/MedCalc-Bench) |
| **MedHallu** | 2025 | QA / Hallucination | 10k Q-A pairs derived from PubMedQA annotated to detect and categorize medical hallucinations. | [paper](https://arxiv.org/abs/2502.14302), [source](https://github.com/medhallu/medhallu) |
| **MedicationQA** | 2019 | Consumer Health QA | Dataset of consumer questions about medications (drug interactions, dosage) with expert answers. | [paper](https://arxiv.org/abs/1908.10023), [source](https://github.com/abachaa/Medication_QA_MedInfo2019) |
| **RJUA-MedDQA** | 2024 | QA / Multimodal | Multimodal benchmark for medical document understanding (images/reports) and clinical reasoning. | [paper](https://arxiv.org/abs/2402.14840), [source](https://github.com/Alipay-Med/medDQA_benchmark) |

### **II. Task-oriented Dialogue**
[Back to Content](#-table-of-contents)
| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **MedDialog** | 2020 | Symptom Diagnosis | Massive dataset (English/Chinese) of doctor-patient conversations scraped from online platforms. | [paper](https://arxiv.org/abs/2004.03329), [source](https://github.com/UCSD-AI4H/Medical-Dialogue-System) |
| **DialoAMC** | 2023 | Symptom Diagnosis | Dataset for Automated Medical Consultation focusing on symptom elicitation and diagnosis. | [paper](https://doi.org/10.1145/3539618.3591901), [source](https://github.com/James-Yip/DialoAMC) |
| **MedDG** | 2022 | Symptom Diagnosis | High-quality entity-annotated medical dialogue dataset for diagnosis and treatment recommendation. | [paper](https://arxiv.org/abs/2010.07497), [source](https://github.com/lwgkzl/MedDG) |
| **MZ** (Muzhi) | 2018 | Symptom Diagnosis | Chinese medical dialogue dataset from the "Muzhi" platform for self-diagnosis agents. | [paper](https://www.aclweb.org/anthology/P18-1216/), [source](https://github.com/kahou2018/Self-Diagnosis) |
| **CMDD** | 2019 | Symptom Diagnosis | Chinese Medical Diagnostic Dialogue dataset (Pediatrics) with symptom-disease mappings. | [paper](https://arxiv.org/abs/1902.04588), [source](https://github.com/Toyohisa/CMDD) |
| **DX** (DXY) | 2019 | Symptom Diagnosis | Diagnostic dataset from DXY.cn, containing dialogue sessions with explicit symptom transitions. | [paper](https://arxiv.org/abs/1908.02402), [source](https://github.com/xuekai-jiang/DX-Dataset) |
| **CovidDialog** | 2020 | Symptom Diagnosis (COVID) | Dialogues specifically regarding COVID-19 consultations, scraped during the pandemic. | [paper](https://arxiv.org/abs/2007.01977), [source](https://github.com/UCSD-AI4H/COVID-Dialogue) |
| **Ext-CovidDialog** | 2023 | Symptom Diagnosis (COVID) | Extended version of CovidDialog with more data covering evolving variants and scenarios. | [paper](https://aclanthology.org/2023.bionlp-1.11/), [source](https://github.com/UCSD-AI4H/COVID-Dialogue) |
| **IMCS-21** | 2021 | Symptom Diagnosis | Interactive Medical Consultation System dataset; focuses on multi-turn diagnostic dialogue. | [paper](https://arxiv.org/abs/2105.02672), [source](https://github.com/lemuria-wchen/imcs21) |
| **BC5CDR** | 2015 | Entity Recognition | BioCreative V task dataset for Chemical-Disease Relation extraction (NER/RE). | [paper](https://academic.oup.com/database/article/2016/1/baw068/2630414), [source](https://github.com/JHnlp/BC5CDR) |
| **NCBI-Disease** | 2014 | Entity Recognition | Corpus of PubMed abstracts annotated with disease mentions for NER. | [paper](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/), [source](https://github.com/spyysalo/ncbi-disease) |
| **PHEE** | 2022 | Entity Extraction | Pharmacovigilance Event Extraction dataset for identifying adverse drug events from text. | [paper](https://aclanthology.org/2022.emnlp-main.373/), [source](https://github.com/ZhaoyueSun/PHEE) |
| **MedAlign** | 2023 | Instruction Following | Clinician-generated dataset for instruction following (summaries, questions) based on EHR data. | [paper](https://arxiv.org/abs/2308.14089), [source](https://github.com/som-shahlab/medalign) |
| **MedInstruct** | 2023 | Instruction Following | Dataset of 52k medical instructions constructed from existing datasets (e.g., MedQA) for tuning. | [paper](https://arxiv.org/abs/2310.14558), [source](https://github.com/XZhang97666/AlpaCare) |
| **BianqueCorpus** | 2023 | Instruction Following | Large-scale multi-turn Chinese health conversation dataset with balanced questioning/suggestions. | [paper](https://arxiv.org/abs/2306.03030), [source](https://github.com/scutcyr/BianQue) |
| **MedSynth** | 2025 | Generation / Summarization | Synthetic medical dialogue-note pairs designed to advance dialogue-to-note and note-to-dialogue tasks. | [paper](https://arxiv.org/abs/2508.01401), [source](https://github.com/ahmadrezarm/MedSynth) |
| **MeQSum** | 2019 | Summarization / Instruction | Dataset for summarizing consumer health questions into canonical medical questions. | [paper](https://www.aclweb.org/anthology/P19-1215/), [source](https://github.com/abachaa/MeQSum) |

### **III. Recommendation Dialogue**
[Back to Content](#-table-of-contents)
| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **DialMed** | 2022 | Recommendation (Drug) | Dialogue dataset designed for medication recommendation based on patient history/dialogue. | [paper](https://arxiv.org/abs/2202.08779), [source](https://github.com/f-window/DialMed) |
| **ReMeDi** | 2021 | Recommendation | "Resources for Medical Dialogue"; focuses on movie/medical recommendation scenarios. | [paper](https://aclanthology.org/2021.emnlp-main.288/), [source](https://github.com/yanguojun123/Medical-Dialogue) |
| **MIMIC-III** | 2016 | Database (Source) | Large database of de-identified health-related data (EHRs) used to construct recommendation tasks. | [paper](https://www.nature.com/articles/sdata201635), [source](https://physionet.org/content/mimiciii/) |
| **DrugBank** | - | Knowledge Base (Source) | Comprehensive database containing information on drugs and drug targets, used for grounding recommendations. | [paper](https://go.drugbank.com/), [source](https://go.drugbank.com/releases) |
| **ProKnow-data** | 2020 | Recommendation | Data used for proactive knowledge-grounded dialogue, often adapted for medical contexts. | [paper](https://arxiv.org/abs/2010.13328), [source](https://github.com/zhw12/ProKnow) |

### **IV. Supportive Dialogue**
[Back to Content](#-table-of-contents)
| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **EmpatheticDialogues** | 2019 | General Empathetic | Large dataset of 25k conversations grounded in emotional situations (general domain). | [paper](https://arxiv.org/abs/1811.00207), [source](https://github.com/facebookresearch/EmpatheticDialogues) |
| **MELD** | 2019 | General Empathetic | Multimodal EmotionLines Dataset; textual/audio/visual emotion recognition. | [paper](https://arxiv.org/abs/1810.02508), [source](https://github.com/declare-lab/MELD) |
| **PsyQA** | 2021 | Mental Health Support | Chinese dataset of psychological health support (Q&A) with strategy annotations. | [paper](https://arxiv.org/abs/2106.01702), [source](https://github.com/thu-coai/PsyQA) |
| **ESConv** | 2021 | Mental Health Support | Emotional Support Conversation dataset designed to train agents in empathy and support strategies. | [paper](https://arxiv.org/abs/2106.01144), [source](https://github.com/thu-coai/Emotional-Support-Conversation) |
| **SoulChat-Corpus** | 2023 | Mental Health Support | Large-scale Chinese dataset for single-turn and multi-turn empathetic psychological counseling. | [paper](https://arxiv.org/abs/2304.09842), [source](https://github.com/scutcyr/SoulChat) |
| **MTS-Dialogue** | 2023 | Clinical Support/Summ. | 1.7k doctor-patient conversations paired with corresponding clinical note summaries. | [paper](https://aclanthology.org/2023.eacl-main.168/), [source](https://github.com/abachaa/MTS-Dialog) |
| **SMILECHAT** | 2023 | Mental Health Support | Dataset for mental health support focusing on cognitive distortion detection and reframing. | [paper](https://arxiv.org/abs/2311.00445), [source](https://github.com/qiuhuachuan/smile) |

### **V. Hybrid Function**
[Back to Content](#-table-of-contents)
| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **MidMed** | 2023 | Hybrid (Diag/Rec/Chat) | Mixed-type dialogue corpus covering diagnosis, recommendation, QA, and chitchat in one session. | [paper](https://arxiv.org/abs/2306.02923), [source](https://github.com/xmshi-trio/MidMed) |
| **MedEval** | 2023 | Evaluation Benchmark | Multi-level, multi-task benchmark spanning 35 body regions and 8 exam modalities for LLM eval. | [paper](https://arxiv.org/abs/2310.14088), [source](https://github.com/Zhihong-Zhu/MedEval) |
| **MedTrinity-25M** | 2024 | Multimodal / Hybrid | Massive multimodal dataset (25M images) with multigranular annotations (Image-ROI-Text). | [paper](https://arxiv.org/abs/2408.02900), [source](https://github.com/UCSC-VLAA/MedTrinity-25M) |
| **MENTAT** | 2025 | Mental Health / Hybrid | Clinician-annotated benchmark for complex psychiatric decision-making (diagnosis, triage, etc.). | [paper](https://hai.stanford.edu/research/mentat), [source](https://github.com/stanford-crfm/mentat) |
| **MedAlpaca** | 2023 | Instruction Tuning | Collection of datasets (see Medical Meadow) used to train the MedAlpaca model series. | [paper](https://arxiv.org/abs/2304.08247), [source](https://github.com/kbressem/medAlpaca) |
| **NoteChat** | 2023 | Generation / Hybrid | Synthetic patient-physician conversations conditioned on clinical notes (Note-to-Dialogue). | [paper](https://arxiv.org/abs/2310.15959), [source](https://github.com/believewhat/Dr.NoteAid) |

## ⛪ Leading Group
[Back to Content](#-table-of-contents)
| Institution | Leading Researcher/Group | Source |
| :--- | :--- | :--- | 
|Google|Google Health|[Homepage](https://health.google/pubs/)|
|NIH|Zhiyong Lu|[Homepage](https://www.ncbi.nlm.nih.gov/research/bionlp/)|
|Open AI|Health AI Team|[Homepage](https://openai.com/science/)|
|Ant Group|AI for Science Team|[Homepage](https://www.antgroup.com/en)|
|Alibaba|Tongyi Lab, Damo|[Homepage](https://github.com/Alibaba-NLP), [Homepage](https://github.com/alibaba-damo-academy)|
|Shanghai AI Lab|AI for Science Team|[Homepage](https://ai4.science/)|
|Baichuan AI|AI Lab|[Homepage](https://github.com/baichuan-inc)|
|Tecent|Jarvislab, Xiaobin Hu|[Homepage](https://jarvislab.tencent.com/), [Homepage](https://huuxiaobin.github.io/)|
|Huawei|NoAH|[Homepage](http://dev3.noahlab.com.hk/research.html)|
|ByteDance|Seed,AI for Science Team|[Homepage](https://seed.bytedance.com/zh/direction/ai_for_science)|
|Microsoft Research|Hoifung Poon|[Homepage](https://scholar.google.com/citations?user=yqqmVbkAAAAJ&hl=en)|
|Harvard|Xiang Li, Faisal Mahmood Lab, Pranav Rajpurkar, Tianxi Cai|[Homepage](https://xiangli-shaun.github.io/), [Homepage](https://faisal.ai/), [Homepage](https://pranavrajpurkar.com/), [Homepage](https://dbmi.hms.harvard.edu/people/tianxi-cai)|
|Maryland|Hanan Samet|[Homepage](https://www.cs.umd.edu/~hjs/)|
|MIT|Paul Liang, Peter Szolovits|[Homepage](https://pliang279.github.io/), [Homepage](https://people.csail.mit.edu/psz/web/publications.html)|
|Oxford|Tingting Zhu, David A. Clifton|[Homepage](https://eng.ox.ac.uk/people/tingting-zhu), [Homepage](https://eng.ox.ac.uk/chi)|
|Cambridge|Vanderschaar-lab, Andreas Vlachos|[Homepage](https://www.vanderschaar-lab.com/prof-mihaela-van-der-schaar/), [Homepage](https://andreasvlachos.github.io/)|
|NTU|Chunyan Miao|[Homepage](https://dr.ntu.edu.sg/entities/person/Miao-Chun-Yan)|
|Tsinghua University|Yang Liu, Hong-Yu Zhou, Weizhi Ma|[Homepage](https://nlp.csai.tsinghua.edu.cn/~ly/), [Homepage](https://zhouhy.org/), [Homepage](https://mawz12.github.io/)|
|UNC|Tianlong Chen, Huaxiu Yao|[Homepage](https://genai4health.github.io/), [Homepage](https://www.huaxiuyao.io/)|
|Yale|Clinical NLP Lab|[Homepage](https://medicine.yale.edu/lab/clinical-nlp/team/)|
|UBC|Xiaoxiao Li|[Homepage](https://tea.ece.ubc.ca/)|
|UIUC|Jimeng Sun,Jiawei Han|[Homepage](https://www.sunlab.org/), [Homepage](http://hanj.cs.illinois.edu/)|
|ZJU|DCDmllm|[Homepage](https://github.com/DCDmllm)|
|Notre Dame|SCLab|[Homepage](https://scl-nd.github.io/)|
|Pennsylvania|Tianyu Han, Fenglong Ma, Lyle Ungar|[Homepage](https://www.tianyuhan.ai/), [Homepage](https://fenglong-ma.github.io/), [Homepage](https://scholar.google.com/citations?hl=en&user=KCiDjbkAAAAJ&view_op=list_works&sortby=pubdate)|
|Emory|Carl Yang|[Homepage](https://www.cs.emory.edu/~jyang71/)|
|Stanford|SNAP, James Zou, Yejin Choi|[Homepage](https://snap.stanford.edu/papers.html), [Homepage](https://zou-group.github.io/index.html), [Homepage](https://yejinc.github.io/)|
|PKU|Liantao Ma, Yasha Wang|[Homepage](http://scholar.pku.edu.cn/malt/home)|
|TJU|ADM Group|[Homepage](https://www.adm-cube.online/)|
|Edinburgh|Ewen M Harrison|[Homepage](https://surgery.ed.ac.uk/research/cachexiagroup)|
|Virginia|Aidong Zhang, Xuan Wang|[Homepage](https://engineering.virginia.edu/faculty/aidong-zhang), [Homepage](https://xuanwang91.github.io/)|
|CUHK|Freedom AI, YuanWu, Michael R. Lyu|[Homepage](https://github.com/FreedomIntelligence), [Homepage](https://www.bme.cuhk.edu.hk/yuan/people.html), [Homepage](http://www.cse.cuhk.edu.hk/lyu/)|
|CityU|Xiangyu Zhao|[Homepage](https://zhaoxyai.github.io/)|
|Houston Methodist|Wang Lab|[Homepage](https://guangyuwanglab.github.io/web/)|
|Mbzuai|Jianing Qiu|[Homepage](https://mbzuai.ac.ae/study/faculty/jianing-qiu/)|
|DKFZ|German Cancer Research Center|[Homepage](https://www.dkfz.de/en/research)|
|California|Yuyin Zhou|[Homepage](https://yuyinzhou.github.io/)|
|ETH|Michael Moor|[Homepage](https://michaelmoor.me/)|
|JOHNS HOPKINS|Suchi Saria|[Homepage](https://www.cs.jhu.edu/faculty/suchi-saria/)|
|Cornell|Fei Wang, Claire Cardie|[Homepage](https://wcm-wanglab.github.io/index.html), [Homepage](https://www.cs.cornell.edu/home/cardie/)|
|GE Healthcare|Xiao Cao|[Homepage](https://sites.google.com/view/danicaxiao/home)|
|Rutgers|Mu Zhou|[Homepage](https://sites.google.com/view/mu-zhou)|
|UT|Ying Ding, Wenqi Shi|[Homepage](https://yingding.ischool.utexas.edu/), [Homepage](https://wshi83.github.io/)|
|UC Berkeley|Bin Yu|[Homepage](https://binyu.stat.berkeley.edu/#recent-publications-talks-news)|
|UW|Hannaneh Hajishirzi|[Homepage](https://hannaneh.ai/)|
|LMU Munich|Volker Tresp|[Homepage](https://www.dbs.ifi.lmu.de/~tresp/)|
|FuDan|Zhongyu Wei|[Homepage](http://www.fudan-disc.com/people/zywei)|
|Minnesota|Rui Zhang|[Homepage](https://ruizhang.umn.edu/)|

## 📖 Awesome Methods, Model, and Resource List
[Back to Content](#-table-of-contents)
### 🤖**LSC**
[Back to Content](#-table-of-contents)
#### 📊Planning
[Back to Content](#-table-of-contents)
- **BioGPT: generative pre-trained transformer for biomedical text generation and mining** (_Briefings Bioinf._, 2023) [paper](https://arxiv.org/abs/2210.10341), [code](https://github.com/microsoft/BioGPT)
  > A domain-specific generative Transformer pre-trained on large-scale biomedical literature to achieve state-of-the-art performance in text generation and mining tasks.

- **BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model** (_BioNLP_, 2022) [paper](https://arxiv.org/abs/2204.03905), [code](https://github.com/GanjinZero/BioBART)
  > Adapts the BART architecture to the biomedical domain with enhanced pre-training tasks, significantly improving performance on summarization and dialogue generation.

- **ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission** (_CHIL_, 2020) [paper](https://arxiv.org/abs/1904.05342), [code](https://github.com/kexinhuang12345/clinicalBERT)
  > Develops contextual embeddings specifically for clinical notes to effectively predict hospital readmission and model long-term clinical dependencies.

- **BioMegatron: Larger Biomedical Domain Language Model** (_EMNLP_, 2020) [paper](https://arxiv.org/abs/2010.06060), [code](https://github.com/NVIDIA/NeMo)
  > Leverages the Megatron-LM infrastructure to train a large-scale biomedical language model, demonstrating improvements in named entity recognition and QA tasks.

- **Toward expert-level medical question answering with large language models** (_Nature_, 2023) [paper](https://www.nature.com/articles/s41586-023-06291-2), [code](https://github.com/google-research-datasets/MultiMedQA)
  > Introduces Med-PaLM, utilizing instruction tuning and ensemble refinement to become the first AI to exceed the passing score on the USMLE.

- **CoD: Towards an Interpretable Medical Agent using Chain of Diagnosis** (_ICML AI4Science_, 2024) [paper](https://arxiv.org/abs/2407.13301), [code](https://github.com/WeChat-AI/CoD)
  > Proposes a Chain of Diagnosis (CoD) framework that breaks down the diagnostic process into interpretable steps to enhance transparency and accuracy.

- **HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge** (_EMNLP Findings_, 2023) [paper](https://arxiv.org/abs/2304.06975), [code](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)
  > Incorporates a structured medical knowledge graph into the LLaMA model via instruction tuning to significantly enhance Chinese medical QA capabilities.

- **Learning Causal Alignment for Reliable Disease Diagnosis** (_ICCV_, 2023) [paper](https://arxiv.org/abs/2310.01766), [code](https://github.com/Ying-Jie-Tan/Causal-Alignment)
  > Introduces a causal alignment framework to mitigate confounding biases in medical data, ensuring more reliable and generalizable disease diagnosis.

- **Reasoning with large language models for medical question answering** (_npj Digit. Med._, 2024) [paper](https://pubmed.ncbi.nlm.nih.gov/38960731/)
  > systematically evaluates different reasoning strategies (like Chain-of-Thought) in LLMs to identify the most effective methods for complex medical QA.

- **Empowering biomedical discovery with AI agents** (_Nature_, 2024) [paper](https://pubmed.ncbi.nlm.nih.gov/39486399/)
  > Discusses the paradigm shift towards autonomous AI agents capable of planning and executing experiments to accelerate biomedical research and discovery.

- **A fast nonnegative autoencoder-based approach to latent feature analysis on high-dimensional and incomplete data** (_IEEE TNNLS_, 2024) [paper](https://ieeexplore.ieee.org/abstract/document/10265117)
  > Proposes a highly efficient nonnegative autoencoder designed to extract latent features from high-dimensional, sparse, and incomplete medical datasets.

- **Multiview latent space learning with progressively fine-tuned deep features for unsupervised domain adaptation** (_Inf. Sci._, 2024) [paper](https://www.sciencedirect.com/science/article/pii/S0020025524001361)
  > Develops a method to align multiview latent spaces using progressively fine-tuned features, improving unsupervised domain adaptation in medical imaging analysis.

- **Autosurv: interpretable deep learning framework for cancer survival analysis incorporating clinical and multi-omics data** (_npj Precis. Oncol._, 2023) [paper](https://www.nature.com/articles/s41698-023-00494-6), [code](https://github.com/TencentAILabHealthcare/AutoSurv)
  > A comprehensive and interpretable deep learning framework that integrates clinical and multi-omics data to improve cancer survival prediction accuracy.

- **Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model** (_arXiv_, 2023) [paper](https://arxiv.org/abs/2310.09089), [code](https://github.com/Opencs/Qilin-Med)
  > Presents a multi-stage training strategy to inject massive medical knowledge into LLMs, enhancing their reasoning and dialogue performance in Chinese medical contexts.

- **Counterfactual reasoning using causal Bayesian networks as a healthcare governance tool** (_Sci. Rep._, 2024) [paper](https://pubmed.ncbi.nlm.nih.gov/39531901/)
  > Applies causal Bayesian networks to perform counterfactual analysis, providing a quantitative tool for evaluating healthcare policies and governance decisions.

- **Large Language Models for Medical Forecasting - Foresight 2** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2412.10848)
  > Introduces a generative foundation model trained on longitudinal patient records to forecast future medical events and health trajectories.

- **Ontology accelerates few-shot learning capability of large language model: A study in extraction of drug efficacy in a rare pediatric epilepsy** (_Comput. Methods Programs Biomed._, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40311258/)
  > Demonstrates that integrating domain ontologies significantly boosts the few-shot learning performance of LLMs for information extraction in rare diseases.

- **A generalist medical language model for disease diagnosis assistance** (_Nat. Med._, 2024) [paper](https://www.nature.com/articles/s41591-024-03416-6)
  > Presents AMIE, a generalist medical AI system optimized for diagnostic dialogue that matches or exceeds primary care physicians in simulated diagnostic tasks.

- **Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks** (_JAMIA_, 2024) [paper](https://arxiv.org/abs/2311.11608), [code](https://github.com/DUTIR-BioNLP/Taiyi-LLM)
  > A bilingual (English/Chinese) LLM specifically fine-tuned to handle a diverse range of biomedical tasks, including NER, RE, and QA.


#### 🧠Memory
[Back to Content](#-table-of-contents)

- **Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.18503)
  > Proposes an Automatic Attention Alignment (AAA) mechanism to align the visual attention of VLMs with clinical masks, enhancing interpretability and performance.

- **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?** (_EMNLP_, 2022) [paper](https://arxiv.org/abs/2202.12837), [code](https://github.com/SewonMin/icl-demonstrations)
  > Demonstrates that the ground-truth accuracy of labels in demonstrations matters less than the label space and distribution, reshaping the understanding of in-context learning.

- **HuatuoGPT-II: One-stage Training for Medical Adaption of LLMs** (_ACL Findings_, 2024) [paper](https://arxiv.org/abs/2311.09774), [code](https://github.com/FreedomIntelligence/HuatuoGPT-II)
  > Introduces a one-stage training protocol that unifies medical domain adaptation and general instruction following, simplifying the training pipeline.

- **Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine** (_npj Digit. Med._, 2024) [paper](https://www.nature.com/articles/s41746-024-01010-1), [code](https://github.com/autonlab/DR-Prompting)
  > Investigates the use of structured prompting strategies to elicit and visualize the diagnostic reasoning paths of LLMs, improving transparency.

- **AttriPrompter: Auto-Prompting with Attribute Semantics for Zero-shot Nuclei Detection via Visual-Language Pre-trained Models** (_MICCAI_, 2024) [paper](https://arxiv.org/abs/2410.16820), [code](https://github.com/Hao-Z-2000/AttriPrompter)
  > A zero-shot framework utilizing attribute-based text prompts to guide visual-language models in detecting nuclei without task-specific training.

- **A context-based chatbot surpasses radiologists and generic ChatGPT in following the ACR appropriateness guidelines** (_Sci. Rep._, 2023) [paper](https://pubmed.ncbi.nlm.nih.gov/37489981/)
  > Develops a specialized chatbot that leverages clinical context to adhere to ACR appropriateness guidelines more accurately than human radiologists.

- **MedVH: Towards Systematic Evaluation of Hallucination for Large Vision Language Models in the Medical Context** (_ECCV_, 2024) [paper](https://arxiv.org/abs/2407.02730), [code](https://github.com/OpenMEDLab/MedVH)
  > Establishes a comprehensive benchmark and evaluation dataset specifically designed to detect and analyze hallucinations in medical vision-language models.

- **The FAIIR conversational AI agent assistant for youth mental health service provision** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01647-6)
  > Presents FAIIR, a conversational agent designed to assist in the triage and service provision for youth mental health, reducing clinician workload.

- **Galactica: A Large Language Model for Science** (_arXiv_, 2022) [paper](https://arxiv.org/abs/2211.09085), [code](https://github.com/paperswithcode/galgal)
  > A large language model trained on a massive corpus of scientific knowledge, designed to store, reason, and generate scientific content.

- **Clinical ModernBERT: An efficient and long context encoder for biomedical text** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2504.03964)
  > Adapts the ModernBERT architecture to the clinical domain, offering a high-efficiency encoder capable of processing long-context electronic health records.

- **DK-BEHRT: Teaching language models international classification of disease (ICD) codes using known disease descriptions** (_CHIL_, 2024) [paper](https://proceedings.mlr.press/v281/an25a.html), [code](https://github.com/pathology-dynamics/DK-BEHRT)
  > Enhances the BEHRT model by incorporating textual descriptions of diseases, significantly improving the accuracy of automated ICD coding.

- **Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2412.16178)
  > Benchmarks various long-context LLMs on their ability to extract relevant information from lengthy and complex electronic health records.

- **Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models** (_ACL Findings_, 2024) [paper](https://arxiv.org/abs/2308.15022), [code](https://github.com/fatemehsc/Re-Sum)
  > Proposes a recursive summarization technique to compress dialogue history, enabling LLMs to maintain long-term memory in medical consultations.

- **Adapted large language models can outperform medical experts in clinical text summarization** (_Nat. Med._, 2024) [paper](https://www.nature.com/articles/s41591-024-02855-5)
  > Provides empirical evidence that domain-adapted LLMs generate clinical summaries that are rated higher in quality and accuracy than those by human experts.

- **BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights** (_EMNLP Findings_, 2023) [paper](https://arxiv.org/abs/2311.16075), [code](https://github.com/Michal-Stefanik/BioLORD-2023)
  > Produces rich semantic textual representations by grounding LLM generation in definitions and relationships from clinical knowledge graphs.

- **AI-Enabled Conversational Journaling for Advancing Parkinson's Disease Symptom Tracking** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.03532)
  > Develops a conversational agent that engages patients in journaling to track and analyze Parkinson's disease symptoms over time.


#### 👥Cooperation
[Back to Content](#-table-of-contents)

- **MEDCO: Medical Education Copilots Based on A Multi-Agent Framework** (_ECCV Workshops_, 2024) [paper](https://arxiv.org/abs/2408.12496)
  > Introduces a multi-agent educational copilot system comprising student, patient, and expert agents to simulate realistic clinical training scenarios.

- **ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration** (_WWW_, 2025) [paper](https://arxiv.org/abs/2410.02551), [code](https://github.com/PKU-AICare/ColaCare)
  > Enhances EHR predictive modeling by using a multi-agent "medical team" (DoctorAgents and MetaAgent) to collaborate on patient data analysis.

- **ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs** (_ACL Findings_, 2024) [paper](https://arxiv.org/abs/2309.13007), [code](https://github.com/ReConcile-LLM/ReConcile)
  > A multi-agent framework where diverse LLMs engage in round-table discussions to reach consensus, significantly improving reasoning accuracy.

- **MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration** (_ACL Findings_, 2025) [paper](https://arxiv.org/abs/2506.19835), [code](https://github.com/yczhou001/MAM)
  > Decomposes diagnostic tasks into specialized agent roles (General Practitioner, Specialist, Radiologist) to handle multi-modal medical data effectively.

- **MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making** (_NeurIPS_, 2024) [paper](https://arxiv.org/abs/2404.15155), [code](https://github.com/mit-medialab/MDAgents)
  > Dynamically adapts the collaboration structure (solo vs. group) of LLM agents based on the medical complexity of the query.

- **Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions (MedAgentSim)** (_MICCAI_, 2025) [paper](https://arxiv.org/abs/2503.22678), [code](https://github.com/MAXNORM8650/MedAgentSim)
  > Presents MedAgentSim, a framework where doctor and patient agents interact and evolve their diagnostic strategies through experience without human labeling.

- **MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning** (_arXiv_, 2023) [paper](https://arxiv.org/abs/2311.10537), [code](https://github.com/GersteinLab/MedAgents)
  > Leveraging a multi-agent debate mechanism to enhance zero-shot clinical reasoning capabilities by simulating medical consultations.

#### ⏫Self-evolution
[Back to Content](#-table-of-contents)

- **AlphaEvolve: A coding agent for scientific and algorithmic discovery** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2506.13131)
  > An evolutionary coding agent from DeepMind capable of autonomously discovering novel algorithms and optimizing code for scientific problems.

- **Revolutionizing healthcare: the role of artificial intelligence in clinical practice** (_BMC Med. Educ._, 2023) [paper](https://pubmed.ncbi.nlm.nih.gov/37740191/)
  > A comprehensive review discussing the transformative impact and ethical implications of integrating AI agents into clinical workflows.

- **Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2405.02957), [code](https://github.com/OpenBMB/AgentHospital)
  > Simulates a full hospital environment where doctor agents continuously evolve and improve their diagnostic skills by treating patient agents.

- **STLLaVA-Med: Self-Training Large Language and Vision Assistant for Medical Question-Answering** (_EMNLP_, 2024) [paper](https://arxiv.org/abs/2406.19973), [code](https://github.com/heliossun/STLLaVA-Med)
  > Uses a self-training pipeline with Direct Preference Optimization (DPO) to improve medical VLM performance using auto-generated data.

- **Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.22954)
  > Proposes a framework for open-ended agent evolution where the system can rewrite its own code to continuously improve its learning and reasoning mechanisms.
 
### 🤖**EP**
[Back to Content](#-table-of-contents)

#### 📊Planning
[Back to Content](#-table-of-contents)

- **Towards Medical Complex Reasoning with LLMs through Medical Verifiable Problems** (_ACL Findings_, 2025) [paper](https://aclanthology.org/2025.findings-acl.751/)
  > Introduces the MedVP dataset, focusing on verifiable medical problems to benchmark and enhance the complex reasoning capabilities of LLMs.

- **Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback** (_AAAI_, 2024) [paper](https://arxiv.org/abs/2308.03549), [code](https://github.com/Sympfer/Zhongjing)
  > Enhances Chinese medical LLMs using a complete RLHF pipeline with expert doctors involved in the feedback loop to ensure professional accuracy.

- **Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies** (_BioNLP_, 2025) [paper](https://aclanthology.org/2025.bionlp-1.14/)
  > Evaluates various prompting strategies, such as chain-of-thought and self-consistency, to improve the accuracy of biomedical claim verification.

- **Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence** (_EMNLP Findings_, 2023) [paper](https://arxiv.org/abs/2310.01299), [code](https://github.com/epfl-dlab/em-evidence)
  > Proposes a latent variable model using Expectation Maximization to select relevant evidence and generate high-quality explanations for medical questions.

- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (_ICLR_, 2023) [paper](https://arxiv.org/abs/2203.11171), [code](https://github.com/google-research/self-consistency)
  > Introduces a decoding strategy that samples multiple reasoning paths and selects the most consistent answer, significantly boosting performance on reasoning tasks.

- **S2AF: An action framework to self-check the Understanding Self-Consistency of Large Language Models** (_Neural Netw._, 2025) [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608025002448)
  > Develops a framework that enables LLMs to self-evaluate their understanding and consistency through an action-based checking mechanism.

- **Ranked Voting based Self-Consistency of Large Language Models** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.10772)
  > Proposes a ranked voting mechanism to aggregate outputs from self-consistency sampling, offering better robustness than simple majority voting.

- **A comparative evaluation of chain-of-thought-based prompt engineering techniques for medical question answering** (_Sci. Rep._, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40602316/)
  > Systematically benchmarks different Chain-of-Thought prompting variations to identify the most effective strategies for medical exams.

- **Tree-Planner: Efficient Close-loop Task Planning with Large Language Models** (_ICLR_, 2024) [paper](https://arxiv.org/abs/2310.08582), [code](https://github.com/Mestway/Tree-Planner)
  > Formulates task planning as a tree search problem, allowing agents to perform efficient closed-loop planning and error correction.

- **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** (_ICLR_, 2023) [paper](https://arxiv.org/abs/2205.10625)
  > A prompting strategy that decomposes complex problems into a sequence of simpler sub-problems, solving them sequentially to guide the model.

- **Prompt engineering in consistency and reliability with the evidence-based guideline for LLMs** (_npj Digit. Med._, 2024) [paper](https://pubmed.ncbi.nlm.nih.gov/38378899/)
  > Investigates how guideline-based prompting improves the consistency and clinical reliability of LLM responses in medical decision support.

- **Cost-Effective Framework with Optimized Task Decomposition and Batch Prompting for Medical Dialogue Summary** (_CIKM_, 2023) [paper](https://dl.acm.org/doi/abs/10.1145/3627673.3679671)
  > Proposes a framework that reduces API costs while maintaining summary quality by optimizing task decomposition and using batch prompting.

- **A brain-inspired agentic architecture to improve planning with LLMs** (_Nat. Commun._, 2025) [paper](https://www.nature.com/articles/s41467-025-63804-5)
  > Draws inspiration from human cognitive processes to design an agent architecture that separates planning, execution, and monitoring for better reliability.

- **Self-critiquing models for assisting human evaluators** (_NeurIPS_, 2022) [paper](https://arxiv.org/abs/2206.05802)
  > Trains models to generate natural language critiques of their own or others' outputs, helping human annotators find errors more efficiently.

- **FRAME: Feedback-Refined Agent Methodology for Enhancing Medical Research Insights** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.04649)
  > An agentic framework that iteratively refines its analysis of medical research papers based on structured feedback loops.

- **Agentic Feedback Loop Modeling Improves Recommendation and User Simulation** (_WWW_, 2025) [paper](https://arxiv.org/abs/2410.20027)
  > Models the interaction between recommender agents and user simulator agents as a feedback loop to improve long-term recommendation utility.


#### 🧠Memory
[Back to Content](#-table-of-contents)

- **MOTOR: A Time-To-Event Foundation Model For Structured Medical Records** (_MLHC_, 2023) [paper](https://arxiv.org/abs/2301.03150), [code](https://github.com/tanlab/MOTOR)
  > A foundation model pre-trained on longitudinal structured medical records to perform time-to-event prediction tasks with high accuracy.

- **Agentic LLM Workflows for Generating Patient-Friendly Medical Reports** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2408.01112)
  > Proposes a multi-agent workflow that transforms complex clinical notes into patient-friendly reports, improving accessibility and understanding.

- **Insights from high and low clinical users of telemedicine: a mixed-methods study of clinician workflows, sentiments, and user experiences** (_npj Digit. Med._, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40674858/)
  > A mixed-methods study analyzing clinician workflows and sentiments to understand the factors driving high versus low adoption of telemedicine.

- **Evaluating large language model workflows in clinical decision support for triage and referral and diagnosis** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01684-1)
  > Systematically evaluates LLM-based workflows in clinical decision support systems, specifically focusing on their safety and accuracy in triage and referral.

- **SoftTiger: A Clinical Foundation Model for Healthcare Workflows** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2403.00868), [code](https://github.com/Tigerrr/SoftTiger)
  > Introduces a LLaMA-based clinical foundation model optimized to integrate seamlessly into various healthcare workflows, from summarization to triage.

- **STAF-LLM: A scalable and task-adaptive fine-tuning framework for large language models in medical domain** (_Expert Syst. Appl._, 2025) [paper](https://www.sciencedirect.com/science/article/pii/S0957417425012047)
  > Presents a scalable framework for task-adaptive fine-tuning that efficiently adapts general LLMs to specific medical tasks with limited resources.

- **Addressing Overprescribing Challenges: Fine-Tuning Large Language Models for Medication Recommendation Tasks** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.03687v1)
  > Investigates fine-tuning strategies for LLMs to generate safer medication recommendations, specifically targeting the reduction of overprescribing errors.

- **From pre-training to fine-tuning: An in-depth analysis of Large Language Models in the biomedical domain** (_Artif. Intell. Med._, 2024) [paper](https://www.sciencedirect.com/science/article/pii/S0933365724002458)
  > Provides a comprehensive comparative analysis of pre-training versus fine-tuning strategies for adapting LLMs to biomedical downstream tasks.

- **Open-Ended Medical Visual Question Answering Through Prefix Tuning of Language Models** (_MICCAI_, 2023) [paper](https://arxiv.org/abs/2303.05977), [code](https://github.com/Fuying-Wang/MedVQA-Prefix)
  > Utilizes prefix tuning to adapt frozen language models for medical visual question answering, achieving high performance with few trainable parameters.

- **Diagnosing Transformers: Illuminating Feature Spaces for Clinical Decision-Making** (_NeurIPS_, 2023) [paper](https://arxiv.org/abs/2305.17588)
  > Analyzes the internal feature spaces of Transformer models to interpret how they represent clinical concepts and make decisions.

- **Embedding dynamic graph attention mechanism into Clinical Knowledge Graph for enhanced diagnostic accuracy** (_Expert Syst. Appl._, 2024) [paper](https://www.sciencedirect.com/science/article/pii/S0957417424030823)
  > Integrates a dynamic graph attention mechanism into clinical knowledge graphs to capture evolving patient states for more accurate diagnosis.

- **HALO: Hallucination Analysis and Learning Optimization to Empower LLMs with Retrieval-Augmented Context for Guided Clinical Decision Making** (_AAAI_, 2025) [paper](https://arxiv.org/abs/2409.10011)
  > A framework designed to detect and mitigate hallucinations in clinical decision-making by optimizing the retrieval-augmented context.

- **Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2506.12182)
  > Explores the synergistic effect of instruction tuning and Chain-of-Thought prompting to enhance the contextual understanding of medical QA models.

- **LIFE-CRAFT: A Multi-agentic Conversational RAG Framework for Lifestyle Medicine Coaching with Context Traceability and Case-Based Evidence Synthesis** (_HCII_, 2024) [paper](https://dl.acm.org/doi/abs/10.1007/978-3-032-06004-4_9)
  > A multi-agent RAG system designed for lifestyle medicine coaching that ensures advice is traceable to case-based evidence.

#### 👥Cooperation
[Back to Content](#-table-of-contents)

- **MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2509.23725)
  > Proposes a logic-driven multi-agent framework where agents organize reasoning into explicit syllogistic trees to ensure transparent and verifiable medical decision-making.

- **ConfAgents: A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.04915)
  > Introduces a conformal prediction-based triage mechanism that dynamically assigns cases to single agents or multi-agent teams, balancing accuracy and computational cost.

- **Advancing Healthcare Automation: Multi-Agent System for Medical Necessity Justification** (_BioNLP_, 2024) [paper](https://arxiv.org/abs/2404.17977)
  > Deploys a multi-agent system to automate the labor-intensive process of prior authorization by justifying medical necessity against clinical guidelines.

- **A Two-Stage Proactive Dialogue Generator for Efficient Clinical Information Collection Using Large Language Model** (_Expert Syst. Appl._, 2025) [paper](https://arxiv.org/abs/2410.03770)
  > Develops a diagnostic dialogue system with a two-stage recommendation structure to proactively collect critical patient information and mimic real-doctor conversational styles.

- **Mediator-Guided Multi-Agent Collaboration among Open-Source Models for Medical Decision-Making** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.05996)
  > Utilizes a mediator agent to facilitate Socratic dialogue and reflection among open-source Vision-Language Models (VLMs), enhancing multimodal diagnostic performance.

- **DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2507.02616)
  > Models clinical diagnosis as a dynamic, multi-round loop where the agent team iteratively queries a patient system (MIMIC-Patient) and adapts its strategy based on new findings.

- **MAS-PatientCare: Medical Diagnosis and Patient Management System Based on a Multi-agent Architecture** (_Springer CCIS_, 2025) [paper](https://link.springer.com/chapter/10.1007/978-3-031-84093-7_17)
  > Proposes a comprehensive multi-agent architecture for remote patient monitoring that integrates diagnostic reasoning with patient management workflows.

- **Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2405.11640)
  > A proactive framework that enables agents to autonomously inquire about missing modalities and integrate multimodal evidence for zero-shot medical reasoning.

#### ⏫Self-evolution
[Back to Content](#-table-of-contents)

- **Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions (MedAgentSim)** (_MICCAI_, 2025) [paper](https://arxiv.org/abs/2503.22678), [code](https://github.com/MAXNORM8650/MedAgentSim)
  > Introduces a simulation environment where doctor and patient agents interact and self-evolve through experience replay and feedback, significantly improving diagnostic realism.

- **Integrating Dynamical Systems Learning with Foundational Models: A Meta-Evolutionary AI Framework for Clinical Trials** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2506.14782)
  > Combines dynamical systems theory with LLMs to create a meta-evolutionary framework that optimizes clinical trial designs and simulates patient trajectories.

- **MedPAO: A Protocol-Driven Agent for Structuring Medical Reports** (_HCII_, 2025) [paper](https://link.springer.com/chapter/10.1007/978-3-032-06004-4_4)
  > Presents an agent that strictly follows medical protocols to structure unstructured clinical reports, ensuring high compliance and data quality.

- **Agentic Surgical AI: Surgeon Style Fingerprinting and Privacy Risk Quantification via Discrete Diffusion in a Vision-Language-Action Framework** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2506.08185)
  > Explores the privacy risks of agentic surgical AI by demonstrating how "surgeon style" can be identified and protected using discrete diffusion models.

- **Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.16463)
  > Enhances the initial diagnostic capabilities of LLM agents by simulating clinical experience learning, bridging the gap between passive knowledge and active inquiry.

- **Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.21503)
  > Introduces a "Catfish Agent" designed to inject structured dissent into multi-agent discussions, preventing premature consensus (groupthink) in medical diagnosis.

### 🤖**GS**
[Back to Content](#-table-of-contents)

#### 📊Planning
[Back to Content](#-table-of-contents)

- **HyKGE: A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable Medical LLMs Responses** (_ACL Findings_, 2024) [paper](https://arxiv.org/abs/2312.15883), [code](https://github.com/Global-NLP-Lab/HyKGE)
  > Constructs a hypothesis-driven knowledge graph to verify intermediate reasoning steps, ensuring LLM responses are grounded in medical facts.

- **Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback** (_ICLR_, 2023) [paper](https://arxiv.org/abs/2302.12813), [code](https://github.com/microsoft/LMOps)
  > An iterative framework where the model retrieves external knowledge and refines its answer based on automated feedback to reduce hallucinations.

- **KGARevion: An AI Agent for Knowledge-Intensive Biomedical QA** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2410.04660)
  > An agentic system capable of reviewing and refining its own retrieval and reasoning processes for high-difficulty biomedical questions.

- **EvidenceMap: Learning Evidence Analysis to Unleash the Power of Small Language Models for Biomedical Question Answering** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2501.12746)
  > Maps complex evidence chains into structured representations, enabling smaller language models to perform expert-level evidence analysis.

- **Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering** (_IEEE JBHI_, 2025) [paper](https://ieeexplore.ieee.org/document/10932873)
  > Proposes a method to inject structured multi-hop reasoning capabilities from Knowledge Graphs into smaller models to improve efficiency.

- **Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions** (_EMNLP Findings_, 2024) [paper](https://arxiv.org/abs/2408.00727)
  > Enhances RAG by generating iterative follow-up questions to clarify ambiguities and retrieve more precise medical context.

- **MedicalGLM: A Pediatric Medical Question Answering Model with a Quality Evaluation Mechanism** (_BMC Med. Inform. Decis. Mak._, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40058479/)
  > A fine-tuned GLM for pediatrics equipped with a self-evaluation module that assesses the reliability of its own generated advice.

- **A cascaded retrieval-while-reasoning multi-document comprehension framework with incremental attention for medical question answering** (_Expert Syst. Appl._, 2024) [paper](https://www.sciencedirect.com/science/article/pii/S0957417424025685)
  > Introduces a cascaded framework that interleaves retrieval and reasoning steps with incremental attention to handle multi-document contexts.

- **K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2501.13567)
  > Uses a knowledge-injected compressor to condense retrieved documents, reducing noise and context length while retaining critical medical facts.

- **MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.17784)
  > A prompting network designed to balance the generation of medical entities in CT reports, ensuring comprehensive and accurate reporting.

- **Knowledge-Induced Medicine Prescribing Network for Medication Recommendation** (_Artif. Intell. Med._, 2025) [paper](https://www.sciencedirect.com/science/article/pii/S0950705125008093)
  > Integrates pharmaceutical knowledge graphs into a deep learning network to provide safe and effective medication combinations.

- **Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13108v1)
  > A multi-task learning framework that jointly optimizes for answer extraction and medical category classification to improve overall QA performance.

- **Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.24830)
  > Decomposes model responses into atomic facts and verifies them against retrieved evidence to enhance reliability and explainability.

#### 🧠Memory
[Back to Content](#-table-of-contents)

- **Bias Evaluation and Mitigation in Retrieval-Augmented Medical Question-Answering Systems** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.15454)
  > Systematically evaluates sources of bias in medical RAG systems and proposes mitigation strategies to ensure equitable healthcare advice.

- **Rationale-Guided Retrieval Augmented Generation for Medical Question Answering** (_NAACL_, 2025) [paper](https://aclanthology.org/2025.naacl-long.635/)
  > Generates rationales first to guide the retrieval process, ensuring that retrieved documents support the logical reasoning path.

- **Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering** (_IEEE JBHI_, 2025) [paper](https://ieeexplore.ieee.org/document/10932873)
  > (See Planning section) Enhances memory capacity of small models by embedding multi-hop relations from medical KGs.

- **Seek Inner: LLM-Enhanced Information Mining for Medical Visual Question Answering** (_ACM MM_, 2024) [paper](https://dl.acm.org/doi/10.1145/3701716.3717556)
  > Mines implicit medical knowledge from Large Language Models to supplement visual features in Medical VQA tasks.

- **MMedAgent: Learning to Use Medical Tools with Multi-modal Agent** (_NeurIPS_, 2024) [paper](https://arxiv.org/abs/2407.02483), [code](https://github.com/Fair-Play/MMedAgent)
  > A multimodal agent framework that learns to retrieve and utilize external medical tools (like calculators and search) to solve complex cases.

- **ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents** (_ACL Findings_, 2025) [paper](https://arxiv.org/abs/2410.17657)
  > Enables clinical agents to reflect on the sufficiency of their current information and autonomously decide when to use tools.

- **RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13361)
  > Introduces a recurrence mechanism where the model's own generation is used to refine subsequent retrieval queries for better factuality.

- **Adaptive Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13010v1)
  > Proposes a framework where the medical knowledge graph is adaptively updated based on new findings to keep the QA system current.

- **MedEx: Enhancing Medical Question-Answering with First-Order Logic based Reasoning and Knowledge Injection** (_COLING_, 2025) [paper](https://aclanthology.org/2025.coling-main.649/)
  > Combines neural generation with symbolic First-Order Logic to inject strict medical constraints and knowledge into the memory of the QA system.

- **Explainable Knowledge-Based Learning for Online Medical Question Answering** (_PRICAI_, 2024) [paper](https://link.springer.com/chapter/10.1007/978-981-97-5489-2_26)
  > An online learning approach that updates the model's knowledge base continuously while providing explainable reasoning paths.

- **Efficient Medical Question Answering with Knowledge-Augmented Question Generation** (_ClinicalNLP_, 2024) [paper](https://aclanthology.org/2024.clinicalnlp-1.2/)
  > Augments the training data (memory) of QA models by generating diverse synthetic medical questions grounded in knowledge bases.

- **Leveraging long context in retrieval augmented language models for medical question answering** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01651-w)
  > Investigates the trade-offs and synergies between using long-context windows and RAG for accessing vast medical knowledge.

#### 🧰Action
[Back to Content](#-table-of-contents)

- **KoSEL: Knowledge subgraph enhanced large language model for medical question answering** (_Artif. Intell. Med._, 2024) [paper](https://www.sciencedirect.com/science/article/pii/S0950705124014710)
  > Retrieves relevant subgraphs from a medical knowledge graph to provide structured context, enhancing the LLM's reasoning for medical QA.

- **Are my answers medically accurate? Exploiting medical knowledge graphs for medical question answering** (_Appl. Intell._, 2024) [paper](https://link.springer.com/article/10.1007/s10489-024-05282-8)
  > Proposes a framework that cross-references LLM-generated answers with facts extracted from medical knowledge graphs to ensure accuracy.

- **Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering** (_IEEE JBHI_, 2025) [paper](https://ieeexplore.ieee.org/document/10932873)
  > Enables smaller language models to perform complex biomedical QA by injecting multi-hop reasoning paths derived from knowledge graphs.

- **Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13108v1)
  > A multi-task learning approach that simultaneously optimizes answer extraction and question categorization to improve clinical QA performance.

- **Beyond EHRs: External Clinical knowledge and cohort Features for medication recommendation** (_Artif. Intell. Med._, 2025) [paper](https://www.sciencedirect.com/science/article/pii/S0950705125008093)
  > (Same as "Knowledge-Induced Medicine..." in Planning) Integrates external clinical knowledge graphs with patient cohort features for precise medication recommendation.

- **MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.17784)
  > A network that balances the generation of various medical entities in CT reports through specialized prompting actions.

- **Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.24830)
  > Enhances RAG systems by decomposing answers into atomic facts and verifying each against retrieved evidence for better reliability.

- **K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2501.13567)
  > Employs a compressor module injected with medical knowledge to condense retrieved documents, optimizing the context for the LLM.

- **MedCoT-RAG: Causal Chain-of-Thought RAG for Medical Question Answering** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.15849)
  > Combines retrieval augmentation with causal chain-of-thought reasoning to explain the causal relationships behind medical answers.

- **Towards Efficient Methods in Medical Question Answering using Knowledge Graph Embeddings** (_IEEE BigData_, 2024) [paper](https://ieeexplore.ieee.org/document/10821824)
  > Utilizes knowledge graph embeddings to efficiently retrieve relevant medical concepts, improving QA speed and accuracy.

- **MediTriR: A Triple-Driven Approach to Retrieval-Augmented Generation for Medical Question Answering Tasks** (_IEEE Access_, 2025) [paper](https://ieeexplore.ieee.org/document/11036297)
  > A RAG approach driven by knowledge triples (Subject-Predicate-Object) to ensure the retrieval of structured and precise medical information.

- **Medical Knowledge Graph QA for Drug-Drug Interaction Prediction based on Multi-hop Machine Reading Comprehension** (_arXiv_, 2022) [paper](https://arxiv.org/abs/2212.09400)
  > Predicts drug-drug interactions by treating the task as a multi-hop machine reading comprehension problem over a knowledge graph.

- **MediSearch: Advanced Medical Web Search Engine** (_IEEE ICHI_, 2023) [paper](https://ieeexplore.ieee.org/document/10099048)
  > A specialized search engine framework that aggregates and filters medical information from the web to provide authoritative health answers.

- **Evaluating search engines and large language models for answering health questions** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01546-w)
  > A comparative study evaluating the accuracy, safety, and completeness of traditional search engines versus LLMs in answering health queries.

- **Leveraging long context in retrieval augmented language models for medical question answering** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01651-w)
  > Examines the effectiveness of using long-context LLMs to process extensive retrieved medical documents compared to standard chunking methods.

- **Using Internet search engines to obtain medical information: a comparative study** (_J. Med. Internet Res._, 2012) [paper](https://pubmed.ncbi.nlm.nih.gov/22672889/)
  > A foundational study (cited for context) comparing the efficacy of general-purpose search engines in retrieving accurate medical information.

- **Large language model agents can use tools to perform clinical calculations** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01475-8)
  > Demonstrates that LLM agents equipped with external calculator tools significantly outperform base models in performing complex clinical scores (e.g., MELD).

- **MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2410.13610)
  > Enables LLM agents to execute nested tool calls, allowing them to handle complex medical calculations that require intermediate steps.

- **MMedAgent: Learning to Use Medical Tools with Multi-modal Agent** (_NeurIPS_, 2024) [paper](https://arxiv.org/abs/2407.02483), [code](https://github.com/Fair-Play/MMedAgent)
  > A framework where multimodal agents learn to autonomously select and utilize various medical tools (search, calculators) to solve clinical problems.

- **KMTLabeler: An Interactive Knowledge-Assisted Labeling Tool for Medical Text Classification** (_IEEE ICASSP_, 2024) [paper](https://ieeexplore.ieee.org/document/10540286)
  > An interactive tool that uses medical knowledge to assist human annotators in labeling clinical text, improving efficiency and consistency.

- **ADEPT: An advanced data exploration and processing tool for clinical data insights** (_Database_, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40403533/)
  > A comprehensive software tool designed for the exploration, cleaning, and preprocessing of large-scale clinical datasets for research.

#### 👥Cooperation
[Back to Content](#-table-of-contents)

- **Error Detection in Medical Note through Multi Agent Debate** (_BioNLP_, 2025) [paper](https://aclanthology.org/2025.bionlp-1.12/)
  > Utilizes a multi-agent debate framework where agents critically analyze medical notes to identify and reach consensus on documentation errors.

- **Multi-modal Medical Diagnosis via Large-small Model Collaboration** (_IEEE_, 2025) [paper](https://ieeexplore.ieee.org/document/11095208)
  > Proposes a collaborative framework where large multi-modal models guide smaller, efficient models to improve diagnostic accuracy on resource-constrained devices.

- **MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2509.20067)
  > A multi-agent system where agents self-learn from historical diagnostic cases to build a shared knowledge base, enhancing collaborative decision-making.

- **MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.20824)
  > A comprehensive study and framework for identifying, categorizing, and mitigating safety risks (e.g., toxicity, bias) arising from agent interactions.

- **MedConMA: A Confidence-Driven Multi-agent Framework for Medical Q&A** (_Springer_, 2025) [paper](https://link.springer.com/chapter/10.1007/978-981-96-8180-8_33)
  > Introduces a confidence-driven mechanism where agents weigh their contributions to the final answer based on their self-assessed certainty levels.

- **MDTeamGPT: A Self-Evolving LLM-based Multi-Agent Framework for Multi-Disciplinary Team Medical Consultation** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.13856)
  > Simulates a Multi-Disciplinary Team (MDT) consultation where agents evolve their collaborative strategies over time to solve complex cancer cases.

#### ⏫Self-evolution
[Back to Content](#-table-of-contents)

- **Adaptive Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13010)
  > (Previously listed as *Agentic Medical Knowledge Graphs...*) A framework that autonomously updates its knowledge graph using agentic search to reflect the latest medical research.

- **Large language model agents can use tools to perform clinical calculations** (_npj Digit. Med._, 2025) [paper](https://www.nature.com/articles/s41746-025-01475-8)
  > Demonstrates that enabling LLM agents to autonomously identify the need for and use clinical calculators significantly reduces computational errors.

- **MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2509.20067)
  > (Also listed in Cooperation) Highlights the self-evolution aspect where the system improves its diagnostic logic through self-learned knowledge accumulation.

- **MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.18968)
  > An advanced agentic workflow that iteratively gathers multimodal evidence and refines its reasoning path to provide evidence-based diagnoses.

- **Improving Self-training with Prototypical Learning for Source-Free Domain Adaptation on Clinical Text** (_BioNLP_, 2024) [paper](https://aclanthology.org/2024.bionlp-1.1/)
  > Combines self-training with prototypical learning to adapt clinical NLP models to new hospitals or domains without accessing source data.

- **ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents** (_ACL Findings_, 2025) [paper](https://arxiv.org/abs/2410.17657)
  > Enables agents to "reflect" on their outputs and tool usage history, allowing them to self-correct and optimize their tool selection strategies.

- **TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs for Clinical Interviews** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.20666)
  > A multi-agent framework that assists researchers in performing thematic analysis of clinical interviews, learning from human feedback to improve coding quality.

### 🤖**VWA**
[Back to Content](#-table-of-contents)

#### 📊Planning
[Back to Content](#-table-of-contents)


- **VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation** (_AAAI_, 2024) [paper](https://arxiv.org/abs/2312.12100), [code](https://github.com/Ying-Jie-Tan/VITA)
  > Proposes a medication recommendation framework that prioritizes selecting the most critical drugs over comprehensive but redundant lists, improving safety.

- **EMRs2CSP: Mining Clinical Status Pathway from Electronic Medical Records** (_ACL Findings_, 2025) [paper](https://aclanthology.org/2025.findings-acl.886/)
  > Extracts Clinical Status Pathways (CSP) from EHRs to model the temporal progression of patient states, aiding in proactive clinical planning.

- **HealthBranches: Synthesizing Clinically-Grounded Question Answering Datasets via Decision Pathways** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.07308)
  > Generates synthetic QA datasets by simulating clinical decision pathways (branches), ensuring the data reflects realistic diagnostic logic.

- **From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.10282)
  > A comprehensive study on using LLMs to translate clinical questions directly into evidence-based recommendations, evaluating their utility in decision support.

- **CMQCIC-Bench: A Chinese Benchmark for Evaluating Large Language Models in Medical Quality Control Indicator Calculation** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.11703)
  > Establishes a benchmark for calculating Medical Quality Control Indicators (MQCIs) from medical records, testing LLMs' ability to perform precise administrative planning.

- **Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering** (_arXiv_, 2023) [paper](https://arxiv.org/abs/2309.02233)
  > Enhances black-box LLMs by retrieving relevant context from trusted medical textbooks, improving the accuracy of biomedical planning and QA.

- **M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall in Large Language Models via Question Answering** (_ACL Findings_, 2024) [paper](https://aclanthology.org/2024.findings-acl.238/)
  > A benchmark designed to evaluate long-context clinical reading comprehension, essential for planning based on extensive patient history.

- **Listening to Patients: Detecting and Mitigating Patient Misreport in Medical Dialogue System** (_ACL Findings_, 2025) [paper](https://aclanthology.org/2025.findings-acl.135/)
  > Addresses the planning challenge where patients provide incorrect information, proposing a mechanism to detect and mitigate these misreports during dialogue.

- **Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning** (_ICML_, 2025) [paper](https://icml.cc/virtual/2025/poster/43761)
  > Utilizes a Graph-of-Thought approach integrated with visual and domain knowledge to achieve professional-level reasoning in medical diagnostics.

- **MedPlan: A Two-Stage RAG-Based System for Personalized Medical Plan Generation** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.17900)
  > Generates personalized treatment plans by first retrieving general guidelines and then adapting them to specific patient data in a two-stage process.

- **PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.01592)
  > A protocol for evaluating interactive agents on their ability to plan diagnostic inquiries and gather information efficiently.

- **RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13361)
  > Introduces a recurrence mechanism where the model's own generation is used to refine subsequent retrieval queries for better factuality.

- **End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.15746)
  > Trains an end-to-end agentic system that not only diagnoses but also provides a traceable reasoning path linked to retrieved evidence.

- **Labeling-free RAG-enhanced LLM for intelligent fault diagnosis via reinforcement learning** (_Eng. Appl. Artif. Intell._, 2025) [paper](https://www.sciencedirect.com/science/article/pii/S1474034625007578)
  > *[Methodology]* Integrates RAG and RL for fault diagnosis without labeled data. (Note: Domain is primarily industrial fault diagnosis, not clinical).

- **The Helicobacter pylori AI-clinician harnesses artificial intelligence to personalise H. pylori treatment recommendations** (_Nat. Commun._, 2025) [paper](https://www.nature.com/articles/s41467-025-61329-5)
  > An AI-clinician system that personalizes antibiotic treatment plans for *H. pylori* infection, significantly improving eradication rates.

- **Continual contrastive reinforcement learning: Towards stronger agent for environment-aware fault diagnosis of aero-engines through long-term optimization under highly imbalance scenarios** (_Eng. Appl. Artif. Intell._, 2025) [paper](https://www.sciencedirect.com/science/article/pii/S1474034625001909)
  > *[Methodology]* A reinforcement learning agent for diagnosing aero-engine faults. (Note: Domain is industrial engineering, included for completeness of input list).

- **Integration of Multi-Source Medical Data for Medical Diagnosis Question Answering** (_IEEE Access_, 2024) [paper](https://ieeexplore.ieee.org/document/10752912)
  > Proposes a method to integrate heterogeneous medical data sources (text, structured data) to answer diagnostic questions more accurately.

- **Stage-Aware Hierarchical Attentive Relational Network for Diagnosis Prediction** (_IEEE JBHI_, 2023) [paper](https://ieeexplore.ieee.org/document/10236511)
  > A hierarchical network that captures the stage-wise progression of diseases from EHR data for precise diagnosis prediction.

- **RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2407.05131)
  > Enhances the factuality of medical VLMs by retrieving and grounding responses in reliable multimodal evidence during generation.

#### 🧠Memory
[Back to Content](#-table-of-contents)

- **M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall in Large Language Models via Question Answering** (_ACL Findings_, 2024) [paper](https://aclanthology.org/2024.findings-acl.238/)
  > Establishes a benchmark specifically designed to evaluate the clinical reading comprehension and long-term knowledge recall capabilities of LLMs.

- **PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2505.01592)
  > (Also listed in Planning) A protocol evaluating how agents manage diagnostic history and plan information-gathering steps in interactive scenarios.

- **EMRs2CSP: Mining Clinical Status Pathway from Electronic Medical Records** (_ACL Findings_, 2025) [paper](https://aclanthology.org/2025.findings-acl.886/)
  > (Also listed in Planning) Mines Clinical Status Pathways (CSP) to represent the temporal progression and memory of patient states from EHRs.

- **Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation** (_ACL_, 2025) [paper](https://aclanthology.org/2025.acl-long.1381/)
  > Enhances LLM memory by integrating a medical knowledge graph into the RAG process, ensuring generation is grounded in structured evidence.

- **MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.04413)
  > Uses knowledge graph-elicited reasoning to optimize the retrieval component, providing a more robust memory mechanism for healthcare copilots.

- **CardioTRAP: Design of a Retrieval Augmented System (RAG) for Clinical Data in Cardiology** (_IEEE_, 2025) [paper](https://ieeexplore.ieee.org/document/11081642)
  > Designs a specialized RAG system for cardiology that effectively retrieves and utilizes patient-specific clinical data (memory) for decision support.

- **CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2507.06715)
  > A RAG framework capable of handling structured clinical data and maintaining context awareness during long-text generation.

- **HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation** (_AAAI_, 2025) [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33301)
  > Utilizes a health status-aware attention mechanism and an enhanced EHR graph to capture patient history memory for precise medication recommendation.

- **Listening to Patients: Detecting and Mitigating Patient Misreport in Medical Dialogue System** (_ACL Findings_, 2025) [paper](https://aclanthology.org/2025.findings-acl.135/)
  > (Also listed in Planning) Focuses on verifying the reliability of patient-provided information (memory of symptoms) during medical dialogues.

- **End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.15746)
  > (Also listed in Planning) Trains an agentic system where the retrieval (memory) and reasoning components are optimized end-to-end for traceability.

#### 🧰Action
[Back to Content](#-table-of-contents)

- **CardioTRAP: Design of a Retrieval Augmented System (RAG) for Clinical Data in Cardiology** (_IEEE Access_, 2025) [paper](https://ieeexplore.ieee.org/document/11081642)
  > Designs a specialized RAG system tailored for cardiology that retrieves and processes patient-specific clinical data to support cardiologist decision-making.

- **CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2507.06715)
  > Introduces a RAG framework capable of handling the complex structure and context of clinical texts, enabling more accurate medical report generation.

- **HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation** (_AAAI_, 2025) [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33301)
  > (Also listed in Memory) Uses an action-oriented recommendation engine that leverages health status-aware attention and EHR graphs to prescribe medications.

- **Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation** (_ACL_, 2025) [paper](https://aclanthology.org/2025.acl-long.1381/)
  > (Also listed in Memory) Enhances the retrieval action by utilizing a medical knowledge graph to ground LLM generations in structured, evidence-based facts.

- **MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.04413)
  > Optimizes the retrieval action through knowledge graph-elicited reasoning, improving the relevance and accuracy of information provided by healthcare copilots.

- **KPL: Training-Free Medical Knowledge Mining of Vision-Language Models** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2501.11231)
  > Proposes a training-free method to actively mine and extract medical knowledge hidden within pre-trained Vision-Language Models.

- **End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.15746)
  > Trains an agentic system to perform end-to-end diagnostic actions where every reasoning step is traceable to a specific retrieved document.

- **SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2502.13233)
  > Investigates the utility of integrating commercial search engine actions into the RAG pipeline to supplement internal knowledge bases for medical QA.

- **Enhancing medical information retrieval: Re-engineering the tala-med search engine for improved performance and flexibility** (_BMC Med. Inform. Decis. Mak._, 2025) [paper](https://pubmed.ncbi.nlm.nih.gov/40991950/)
  > Details the re-engineering of the 'tala-med' search engine, optimizing its architecture for more flexible and high-performance medical information retrieval.

- **Designing a Distributed LLM-Based Search Engine as a Foundation for Agent Discovery** (_IEEE_, 2025) [paper](https://ieeexplore.ieee.org/document/10967406)
  > Proposes a distributed architecture for LLM-based search that serves as a foundational layer for autonomous agents to discover and access medical knowledge.

- **How the Algorithmic Transparency of Search Engines Influences Health Anxiety: The Mediating Effects of Trust in Online Health Information Search** (_CHI_, 2025) [paper](https://dl.acm.org/doi/10.1145/3706598.3713199)
  > A user study analyzing how the transparency of search engine algorithms affects user trust and health anxiety during online health information seeking.

- **Transforming Medical Data Access: The Role and Challenges of Recent Language Models in SQL Query Automation** (_MIPRO_, 2024) [paper](https://www.semanticscholar.org/paper/Transforming-Medical-Data-Access%3A-The-Role-and-of-Tankovi%C4%87-%C5%A0ajina/bb37f6df4218feb32cf86bb95fc5a5cc95465a18)
  > Evaluates the capability of LLMs to automate SQL query generation (Text-to-SQL), facilitating easier access to medical databases for non-technical users.

- **Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.16463)
  > (Also listed in Self-evolution) Enhances the agent's diagnostic actions by allowing it to learn from simulated clinical experiences and feedback.

- **Designing VR Simulation System for Clinical Communication Training with LLMs-Based Embodied Conversational Agents** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2503.01767)
  > Integrates LLM-based embodied agents into a Virtual Reality simulation to train medical students in clinical communication actions.

#### 👥Cooperation
[Back to Content](#-table-of-contents)

- **Enhancing Clinical Trial Patient Matching through Knowledge Augmentation and Reasoning with Multi-Agent** (_arXiv_, 2024) [paper](https://arxiv.org/abs/2411.14637)
  > Introduces MAKA, a multi-agent framework that improves patient-trial matching by dynamically augmenting criteria with domain knowledge and performing structured reasoning.

- **TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.08115)
  > Integrates the "Big Five" human teamwork components (e.g., leadership, trust) into a multi-agent system to systematically improve medical decision-making.

- **ClinicalLab: Aligning Agents for Multi-Departmental Clinical Diagnostics in the Real World** (_ACL Findings_, 2025) [paper](https://arxiv.org/abs/2406.13890), [code](https://github.com/Haitian-Liu/ClinicalLab)
  > Presents a comprehensive suite for aligning and evaluating medical agents across 24 clinical departments, featuring a realistic benchmark (ClinicalBench).

- **The Optimization Paradox in Clinical AI Multi-Agent Systems** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2506.06574)
  > Reveals a paradox where systems built from individually optimized "best-of-breed" components underperform due to poor information flow, advocating for end-to-end system validation.

#### ⏫Self-evolution
[Back to Content](#-table-of-contents)


- **EvoAgentX: An Automated Framework for Evolving Agentic Workflows** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2507.03616), [code](https://github.com/EvoAgentX/EvoAgentX)
  > An open-source platform that automates the generation and evolutionary optimization of multi-agent workflows using algorithms like TextGrad and AFlow.

- **MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.00271), [code](https://github.com/qhjqhj00/MetaAgent)
  > Proposes an agent that evolves through "learning by doing," autonomously creating tools and building a knowledge base from its own experiences.

- **ZERA: Zero-init Instruction Evolving Refinement Agent** (_EMNLP_, 2025) [paper](https://arxiv.org/abs/2509.18158), [code](https://github.com/younatics/zera-agent)
  > An automated prompt optimization agent that evolves structured prompts from zero initial instructions using principle-based self-correction.

- **HealthBranches: Synthesizing Clinically-Grounded Question Answering Datasets via Decision Pathways** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2508.07308)
  > (Also listed in Planning) A benchmark generation framework that synthesizes QA datasets from clinical decision pathways to test complex reasoning.

- **A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence** (_arXiv_, 2025) [paper](https://arxiv.org/abs/2507.21046)
  > A comprehensive survey categorizing self-evolving agents by *what* (model/tool/context), *when*, and *how* they evolve, positioning them as a path to ASI.

- **Evolving Collective Cognition in Human-Agent Hybrid Societies: How Agents Form Stances and Boundaries** (_CogSci_, 2025) [paper](https://arxiv.org/abs/2508.17366)
  > Investigates the emergence of collective cognition and social boundaries in hybrid societies where humans and self-evolving agents interact.

## Star History of Awesome-Agentic-Clinical-Dialogue
[Back to Content](#-table-of-contents)

[![Star History Chart](https://api.star-history.com/svg?repos=xqz614/Awesome-Agentic-Clinical-Dialogue&type=date&legend=top-left)](https://www.star-history.com/#xqz614/Awesome-Agentic-Clinical-Dialogue&type=date&legend=top-left)

## 🤝 Contributing
[Back to Content](#-table-of-contents)

Your contributions are always welcome! Please contact [Xiaoquan Zhi](https://github.com/xqz614) or [Chuang Zhao](https://github.com/Data-Designer)
## ✍️ Citation
[Back to Content](#-table-of-contents)

If you find this code useful for your research, please cite our paper:
```bibtex
@article{zhi2025reinventing,
  title={Reinventing Clinical Dialogue: Agentic Paradigms for LLM Enabled Healthcare Communication},
  author={ADM Lab},
  journal={arXiv preprint arXiv:2512.01453},
  year={2025}
}
```
