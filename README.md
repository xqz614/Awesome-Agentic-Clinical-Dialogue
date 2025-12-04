# 🤖 Awesome-Agentic-Clinical-Dialogue
This repo includes papers about methods related to agentic clinical dialogue. We believe that the agentic paradi is still a largely unexplored area, and we hope this repository will provide you with some valuable insights!

Read our survey paper here: [Reinventing Clinical Dialogue: Agentic Paradigms for LLM‑Enabled Healthcare Communication](https://arxiv.org/abs/2512.01453)
## 📘 Overview
This framework facilitates a systematic analysis of the intrinsic trade-offs between creativity and reliability by categorizing methods into four archetypes: Latent Space Clinicians, Emergent Planners, Grounded Synthesizers, and Veriffable Workffow Automators. For each paradigm, we deconstruct the technical realization across the entire cognitive pipeline, encompassing strategic planning, memory management, action execution, collaboration, and evolution, to reveal how distinct architectural choices balance the tension between autonomy and safety. Furthermore, we bridge abstract design philosophies with the pragmatic implementation ecosystem. By mapping real-world applications to our taxonomy and systematically reviewing benchmarks and evaluation metrics speciffc to clinical agents, we provide a comprehensive reference for future development.
<p align="center"><img src="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/blob/main/image/taxonomy.png" height="500px"></p>

## 📁 Table of Contents
- [Key Categories](#-key-categories)
- [Start with Awesome Dataset](#%EF%B8%8Fawesome-dataset)
  - [QA Dialogue](i-qa-dialogue)
  - [Task-oriented Dialogue](ii-task-oriented-dialogue)
  - [Recommendation Dialogue](iii-recommendation-dialogue)
  - [Supportive Dialogue](iv-supportive-dialogue)
  - [Hybrid-function Dialogue](v-hybrid-function)
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

### **I. QA Dialogue**

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

| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **DialMed** | 2022 | Recommendation (Drug) | Dialogue dataset designed for medication recommendation based on patient history/dialogue. | [paper](https://arxiv.org/abs/2202.08779), [source](https://github.com/f-window/DialMed) |
| **ReMeDi** | 2021 | Recommendation | "Resources for Medical Dialogue"; focuses on movie/medical recommendation scenarios. | [paper](https://aclanthology.org/2021.emnlp-main.288/), [source](https://github.com/yanguojun123/Medical-Dialogue) |
| **MIMIC-III** | 2016 | Database (Source) | Large database of de-identified health-related data (EHRs) used to construct recommendation tasks. | [paper](https://www.nature.com/articles/sdata201635), [source](https://physionet.org/content/mimiciii/) |
| **DrugBank** | - | Knowledge Base (Source) | Comprehensive database containing information on drugs and drug targets, used for grounding recommendations. | [paper](https://go.drugbank.com/), [source](https://go.drugbank.com/releases) |
| **ProKnow-data** | 2020 | Recommendation | Data used for proactive knowledge-grounded dialogue, often adapted for medical contexts. | [paper](https://arxiv.org/abs/2010.13328), [source](https://github.com/zhw12/ProKnow) |

### **IV. Supportive Dialogue**

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

| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source |
| :--- | :--- | :--- | :--- | :--- |
| **MidMed** | 2023 | Hybrid (Diag/Rec/Chat) | Mixed-type dialogue corpus covering diagnosis, recommendation, QA, and chitchat in one session. | [paper](https://arxiv.org/abs/2306.02923), [source](https://github.com/xmshi-trio/MidMed) |
| **MedEval** | 2023 | Evaluation Benchmark | Multi-level, multi-task benchmark spanning 35 body regions and 8 exam modalities for LLM eval. | [paper](https://arxiv.org/abs/2310.14088), [source](https://github.com/Zhihong-Zhu/MedEval) |
| **MedTrinity-25M** | 2024 | Multimodal / Hybrid | Massive multimodal dataset (25M images) with multigranular annotations (Image-ROI-Text). | [paper](https://arxiv.org/abs/2408.02900), [source](https://github.com/UCSC-VLAA/MedTrinity-25M) |
| **MENTAT** | 2025 | Mental Health / Hybrid | Clinician-annotated benchmark for complex psychiatric decision-making (diagnosis, triage, etc.). | [paper](https://hai.stanford.edu/research/mentat), [source](https://github.com/stanford-crfm/mentat) |
| **MedAlpaca** | 2023 | Instruction Tuning | Collection of datasets (see Medical Meadow) used to train the MedAlpaca model series. | [paper](https://arxiv.org/abs/2304.08247), [source](https://github.com/kbressem/medAlpaca) |
| **NoteChat** | 2023 | Generation / Hybrid | Synthetic patient-physician conversations conditioned on clinical notes (Note-to-Dialogue). | [paper](https://arxiv.org/abs/2310.15959), [source](https://github.com/believewhat/Dr.NoteAid) |


## 📖 Awesome Methods, Model and Resource List
### 🤖**LSC**
#### 📊Planning
- BioGPT: generative pre-trained transformer for biomedical text generation and mining (_Briefings Bioinf._, 2023) [paper](https://arxiv.org/abs/2210.10341), [code](https://github.com/microsoft/BioGPT)
  > one sentence describing core contribution
- [BioBART: Pretraining and Evaluation of {A} Biomedical Generative Language Model](https://arxiv.org/abs/2204.03905)
- [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)
- [BioMegatron: Larger Biomedical Domain Language Model](https://arxiv.org/abs/2010.06060)
- [Toward expert-level medical question answering with large language models](https://arxiv.org/abs/2305.09617)
- [CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis](https://arxiv.org/abs/2407.13301)
- [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975)
- [Learning Causal Alignment for Reliable Disease Diagnosis](https://arxiv.org/abs/2310.01766)
- [Reasoning with large language models for medical question answering](https://pubmed.ncbi.nlm.nih.gov/38960731/)
- [Empowering biomedical discovery with AI agents](https://pubmed.ncbi.nlm.nih.gov/39486399/)
- [A fast nonnegative autoencoder-based approach to latent feature analysis on high-dimensional and incomplete data](https://ieeexplore.ieee.org/abstract/document/10265117)
- [Multiview latent space learning with progressively fine-tuned deep features for unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0020025524001361)
- [Autosurv: interpretable deep learning framework for cancer survival analysis incorporating clinical and multi-omics data](https://www.nature.com/articles/s41698-023-00494-6)
- [Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model](https://arxiv.org/abs/2310.09089)
- [Counterfactual reasoning using causal Bayesian networks as a healthcare governance tool](https://pubmed.ncbi.nlm.nih.gov/39531901/)
- [Large Language Models for Medical Forecasting - Foresight 2](https://arxiv.org/abs/2412.10848)
- [Ontology accelerates few-shot learning capability of large language model: A study in extraction of drug efficacy in a rare pediatric epilepsy](https://pubmed.ncbi.nlm.nih.gov/40311258/)
- [A generalist medical language model for disease diagnosis assistance](https://www.nature.com/articles/s41591-024-03416-6)
- [Taiyi: A Bilingual Fine-Tuned Large Language Model for Diverse Biomedical Tasks](https://arxiv.org/abs/2311.11608)


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
- [Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning](https://arxiv.org/abs/2505.18503)
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
- [HuatuoGPT-II, One-stage Training for Medical Adaption of LLMs](https://arxiv.org/abs/2311.09774)
- [Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine](https://www.nature.com/articles/s41746-024-01010-1)
- [AttriPrompter: Auto-Prompting with Attribute Semantics for Zero-shot Nuclei Detection via Visual-Language Pre-trained Models](https://arxiv.org/abs/2410.16820)
- [A context-based chatbot surpasses radiologists and generic ChatGPT in following the ACR appropriateness guidelines](https://pubmed.ncbi.nlm.nih.gov/37489981/)
- [MedVH: Towards Systematic Evaluation of Hallucination for Large Vision Language Models in the Medical Context](https://arxiv.org/abs/2407.02730)
- [The FAIIR conversational AI agent assistant for youth mental health service provision](https://www.nature.com/articles/s41746-025-01647-6)
- [Galactica: A Large Language Model for Science](https://arxiv.org/abs/2211.09085)
- [Clinical ModernBERT: An efficient and long context encoder for biomedical text](https://arxiv.org/abs/2504.03964)
- [Dk-behrt: Teaching language models international classification of disease (icd) codes using known disease descriptions](https://proceedings.mlr.press/v281/an25a.html)
- [Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks on EHRs](https://arxiv.org/abs/2412.16178)
- [Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models](https://arxiv.org/abs/2308.15022)
- [Adapted large language models can outperform medical experts in clinical text summarization](https://www.nature.com/articles/s41591-024-02855-5)
- [BioLORD-2023: Semantic Textual Representations Fusing LLM and Clinical Knowledge Graph Insights](https://arxiv.org/abs/2311.16075)
- [Towards evaluating and building versatile large language models for medicine](https://arxiv.org/abs/2311.16075)
- [AI-Enabled Conversational Journaling for Advancing Parkinson's Disease Symptom Tracking](https://arxiv.org/abs/2503.03532)
#### 👥Cooperation
- [MEDCO: Medical Education Copilots Based on A Multi-Agent Framework](https://arxiv.org/abs/2408.12496)
- [ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration](https://dl.acm.org/doi/abs/10.1145/3696410.3714877)
- [ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs](https://arxiv.org/abs/2309.13007)
- [MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](https://arxiv.org/abs/2506.19835)
- [MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making](https://arxiv.org/abs/2404.15155)
- [Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions](https://arxiv.org/abs/2503.22678)
- [MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning](https://arxiv.org/abs/2311.10537)
#### ⏫Self-evolution
- [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)
- [Revolutionizing healthcare: the role of artificial intelligence in clinical practice](https://pubmed.ncbi.nlm.nih.gov/37740191/)
- [Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)
- [STLLaVA-Med: Self-Training Large Language and Vision Assistant for Medical Question-Answering](https://arxiv.org/abs/2406.19973)
- [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954)
### 🤖**EP**
#### 📊Planning
- [Towards Medical Complex Reasoning with LLMs through Medical Verifiable Problems](https://aclanthology.org/2025.findings-acl.751/)
- [Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue](https://arxiv.org/abs/2308.03549)
- [Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies](https://aclanthology.org/2025.bionlp-1.14/)
- [Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence](https://arxiv.org/abs/2310.01299)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
- [S2AF: An action framework to self-check the Understanding Self-Consistency of Large Language Models](https://www.sciencedirect.com/science/article/abs/pii/S0893608025002448)
- [Ranked Voting based Self-Consistency of Large Language Models](https://arxiv.org/abs/2505.10772)
- [A comparative evaluation of chain-of-thought-based prompt engineering techniques for medical question answering](https://pubmed.ncbi.nlm.nih.gov/40602316/)
- [Tree-Planner: Efficient Close-loop Task Planning with Large Language Models](https://arxiv.org/abs/2310.08582)
- [Least-to-Most Prompting Enables Complex Reasoning in Large Language Models](https://arxiv.org/abs/2205.10625)
- [Prompt engineering in consistency and reliability with the evidence-based guideline for LLMs](https://pubmed.ncbi.nlm.nih.gov/38378899/)
- [Cost-Effective Framework with Optimized Task Decomposition and Batch Prompting for Medical Dialogue Summary](https://dl.acm.org/doi/abs/10.1145/3627673.3679671)
- [A brain-inspired agentic architecture to improve planning with LLMs](https://www.nature.com/articles/s41467-025-63804-5)
- [Self-critiquing models for assisting human evaluators](https://arxiv.org/abs/2206.05802)
- [FRAME: Feedback-Refined Agent Methodology for Enhancing Medical Research Insights](https://arxiv.org/abs/2505.04649)
- [Agentic Feedback Loop Modeling Improves Recommendation and User Simulation](https://arxiv.org/abs/2410.20027)
#### 🧠Memory
- [MOTOR: A Time-To-Event Foundation Model For Structured Medical Records](https://arxiv.org/abs/2301.03150)
- [Agentic LLM Workflows for Generating Patient-Friendly Medical Reports](https://arxiv.org/abs/2408.01112)
- [Insights from high and low clinical users of telemedicine: a mixed-methods study of clinician workflows, sentiments, and user experiences](https://pubmed.ncbi.nlm.nih.gov/40674858/)
- [Evaluating large language model workflows in clinical decision support for triage and referral and diagnosis](https://www.nature.com/articles/s41746-025-01684-1)
- [SoftTiger: A Clinical Foundation Model for Healthcare Workflows](https://arxiv.org/abs/2403.00868)
- [STAF-LLM: A scalable and task-adaptive fine-tuning framework for large language models in medical domain](https://www.sciencedirect.com/science/article/pii/S0957417425012047)
- [Addressing Overprescribing Challenges: Fine-Tuning Large Language Models for Medication Recommendation Tasks](https://arxiv.org/abs/2503.03687v1)
- [From pre-training to fine-tuning: An in-depth analysis of Large Language Models in the biomedical domain](https://www.sciencedirect.com/science/article/pii/S0933365724002458)
- [Open-Ended Medical Visual Question Answering Through Prefix Tuning of Language Models](https://arxiv.org/abs/2303.05977)
- [Diagnosing Transformers: Illuminating Feature Spaces for Clinical Decision-Making](https://arxiv.org/abs/2305.17588)
- [Embedding dynamic graph attention mechanism into Clinical Knowledge Graph for enhanced diagnostic accuracy](https://www.sciencedirect.com/science/article/pii/S0957417424030823)
- [HALO: Hallucination Analysis and Learning Optimization to Empower LLMs with Retrieval-Augmented Context for Guided Clinical Decision Making](https://arxiv.org/abs/2409.10011)
- [Instruction Tuning and CoT Prompting for Contextual Medical QA with LLMs](https://arxiv.org/abs/2506.12182)
- [LIFE-CRAFT: A Multi-agentic Conversational RAG Framework for Lifestyle Medicine Coaching with Context Traceability and Case-Based Evidence Synthesis](https://dl.acm.org/doi/abs/10.1007/978-3-032-06004-4_9)
#### 👥Cooperation
- [MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](https://arxiv.org/abs/2509.23725)
- [ConfAgents: A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis](https://arxiv.org/abs/2508.04915)
- [Advancing Healthcare Automation: Multi-Agent System for Medical Necessity Justification](https://arxiv.org/abs/2404.17977)
- [A Two-Stage Proactive Dialogue Generator for Efficient Clinical Information Collection Using Large Language Model](https://arxiv.org/abs/2410.03770)
- [Mediator-Guided Multi-Agent Collaboration among Open-Source Models for Medical Decision-Making](https://arxiv.org/abs/2508.05996)
- [DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making](https://arxiv.org/abs/2507.02616)
- [MAS-PatientCare: Medical Diagnosis and Patient Management System Based on a Multi-agent Architecture](https://link.springer.com/chapter/10.1007/978-3-031-84093-7_17)
- [Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning](https://arxiv.org/abs/2405.11640)
#### ⏫Self-evolution
- [Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions](https://arxiv.org/abs/2503.22678)
- [Integrating Dynamical Systems Learning with Foundational Models: A Meta-Evolutionary AI Framework for Clinical Trials](https://arxiv.org/abs/2506.14782)
- [MedPAO: A Protocol-Driven Agent for Structuring Medical Reports](https://link.springer.com/chapter/10.1007/978-3-032-06004-4_4)
- [Agentic Surgical AI: Surgeon Style Fingerprinting and Privacy Risk Quantification via Discrete Diffusion in a Vision-Language-Action Framework](https://arxiv.org/abs/2506.08185)
- [Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning](https://arxiv.org/abs/2503.16463)
- [Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making](https://arxiv.org/abs/2505.21503)
### 🤖**GS**
#### 📊Planning
- [HyKGE: A Hypothesis Knowledge Graph Enhanced Framework for Accurate and Reliable Medical LLMs Responses](https://arxiv.org/abs/2312.15883)
- [Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback](https://arxiv.org/abs/2302.12813)
- [KGARevion: An AI Agent for Knowledge-Intensive Biomedical QA](https://arxiv.org/abs/2410.04660)
- [EvidenceMap: Learning Evidence Analysis to Unleash the Power of Small Language Models for Biomedical Question Answering](https://arxiv.org/abs/2501.12746)
- [Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering](https://ieeexplore.ieee.org/document/10932873)
- [Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions](https://arxiv.org/abs/2408.00727)
- [MedicalGLM: A Pediatric Medical Question Answering Model with a quality evaluation mechanism](https://pubmed.ncbi.nlm.nih.gov/40058479/)
- [A cascaded retrieval-while-reasoning multi-document comprehension framework with incremental attention for medical question answering](https://www.sciencedirect.com/science/article/pii/S0957417424025685)
- [K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://arxiv.org/abs/2501.13567)
- [MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation](https://arxiv.org/abs/2503.17784)
- [Knowledge-Induced Medicine Prescribing Network for Medication Recommendation](https://www.sciencedirect.com/science/article/pii/S0950705125008093)
- [Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization](https://arxiv.org/abs/2502.13108v1)
- [Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs](https://arxiv.org/abs/2505.24830)
#### 🧠Memory
- [Bias Evaluation and Mitigation in Retrieval-Augmented Medical Question-Answering Systems](https://arxiv.org/abs/2503.15454)
- [Rationale-Guided Retrieval Augmented Generation for Medical Question Answering](https://aclanthology.org/2025.naacl-long.635/)
- [Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering](https://ieeexplore.ieee.org/document/10932873)
- [Seek Inner: LLM-Enhanced Information Mining for Medical Visual Question Answering](https://dl.acm.org/doi/10.1145/3701716.3717556)
- [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/abs/2407.02483)
- [ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](https://arxiv.org/abs/2410.17657)
- [RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering](https://arxiv.org/abs/2502.13361)
- [Adaptive Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge](https://arxiv.org/abs/2502.13010v1)
- [MedEx: Enhancing Medical Question-Answering with First-Order Logic based Reasoning and Knowledge Injection](https://aclanthology.org/2025.coling-main.649/)
- [Explainable Knowledge-Based Learning for Online Medical Question Answering](https://link.springer.com/chapter/10.1007/978-981-97-5489-2_26)
- [Efficient Medical Question Answering with Knowledge-Augmented Question Generation](https://aclanthology.org/2024.clinicalnlp-1.2/)
- [Leveraging long context in retrieval augmented language models for medical question answering](https://www.nature.com/articles/s41746-025-01651-w)
#### 🧰Action
- [KoSEL: Knowledge subgraph enhanced large language model for medical question answering](https://www.sciencedirect.com/science/article/pii/S0950705124014710)
- [Are my answers medically accurate? Exploiting medical knowledge graphs for medical question answering](https://link.springer.com/article/10.1007/s10489-024-05282-8)
- [Infusing Multi-Hop Medical Knowledge Into Smaller Language Models for Biomedical Question Answering](https://ieeexplore.ieee.org/document/10932873)
- [Improving Clinical Question Answering with Multi-Task Learning: A Joint Approach for Answer Extraction and Medical Categorization](https://arxiv.org/abs/2502.13108v1)
- [Beyond EHRs: External Clinical knowledge and cohort Features for medication recommendation](https://www.sciencedirect.com/science/article/pii/S0950705125008093)
- [MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation](https://arxiv.org/abs/2503.17784)
- [Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs](https://arxiv.org/abs/2505.24830)
- [K-COMP: Retrieval-Augmented Medical Domain Question Answering With Knowledge-Injected Compressor](https://arxiv.org/abs/2501.13567)
- [MedCoT-RAG: Causal Chain-of-Thought RAG for Medical Question Answering](https://arxiv.org/abs/2508.15849)
- [Towards Efficient Methods in Medical Question Answering using Knowledge Graph Embeddings](https://ieeexplore.ieee.org/document/10821824)
- [MediTriR: A Triple-Driven Approach to Retrieval-Augmented Generation for Medical Question Answering Tasks](https://ieeexplore.ieee.org/document/11036297)
- [Medical Knowledge Graph QA for Drug-Drug Interaction Prediction based on Multi-hop Machine Reading Comprehension](https://arxiv.org/abs/2212.09400v3)
- [MediSearch: Advanced Medical Web Search Engine](https://ieeexplore.ieee.org/document/10099048)
- [Evaluating search engines and large language models for answering health questions](https://www.nature.com/articles/s41746-025-01546-w)
- [Leveraging long context in retrieval augmented language models for medical question answering](https://www.nature.com/articles/s41746-025-01651-w)
- [Using Internet search engines to obtain medical information: a comparative study](https://pubmed.ncbi.nlm.nih.gov/22672889/)
- [Large language model agents can use tools to perform clinical calculations](https://pubmed.ncbi.nlm.nih.gov/40097720/)
- [MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling](https://arxiv.org/abs/2410.13610)
- [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/abs/2407.02483)
- [KMTLabeler: An Interactive Knowledge-Assisted Labeling Tool for Medical Text Classification](https://ieeexplore.ieee.org/document/10540286)
- [ADEPT: An advanced data exploration and processing tool for clinical data insights](https://pubmed.ncbi.nlm.nih.gov/40403533/)
#### 👥Cooperation
- [Error Detection in Medical Note through Multi Agent Debate](https://aclanthology.org/2025.bionlp-1.12/)
- [Multi-modal Medical Diagnosis via Large-small Model Collaboration](https://ieeexplore.ieee.org/document/11095208)
- [MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM](https://arxiv.org/abs/2509.20067)
- [MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems](https://arxiv.org/abs/2505.20824)
- [MedConMA: A Confidence-Driven Multi-agent Framework for Medical Q&A](https://link.springer.com/chapter/10.1007/978-981-96-8180-8_33)
- [MDTeamGPT: A Self-Evolving LLM-based Multi-Agent Framework for Multi-Disciplinary Team Medical Consultation](https://arxiv.org/abs/2503.13856)
#### ⏫Self-evolution
- [Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge](https://arxiv.org/abs/2502.13010)
- [Large language model agents can use tools to perform clinical calculations](https://www.nature.com/articles/s41746-025-01475-8)
- [MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM](https://arxiv.org/abs/2509.20067)
- [MedAgent-Pro: Towards Evidence-based Multi-modal Medical Diagnosis via Reasoning Agentic Workflow](https://arxiv.org/abs/2503.18968)
- [Improving Self-training with Prototypical Learning for Source-Free Domain Adaptation on Clinical Text](https://aclanthology.org/2024.bionlp-1.1/)
- [ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](https://arxiv.org/abs/2410.17657)
- [TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs for Clinical Interviews](https://arxiv.org/abs/2503.20666)
### 🤖**VWA**
#### 📊Planning
- [VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation](https://arxiv.org/abs/2312.12100)
- [EMRs2CSP : Mining Clinical Status Pathway from Electronic Medical Records](https://aclanthology.org/2025.findings-acl.886/)
- [HealthBranches: Synthesizing Clinically-Grounded Question Answering Datasets via Decision Pathways](https://arxiv.org/abs/2508.07308)
- [From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making](https://arxiv.org/abs/2505.10282)
- [CMQCIC-Bench: A Chinese Benchmark for Evaluating Large Language Models in Medical Quality Control Indicator Calculation](https://arxiv.org/abs/2502.11703)
- [Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering](https://arxiv.org/abs/2309.02233)
- [M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall in Large Language Models via Question Answering](https://aclanthology.org/2024.findings-acl.238/)
- [Listening to Patients: Detecting and Mitigating Patient Misreport in Medical Dialogue System](https://aclanthology.org/2025.findings-acl.135/)
- [Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning](https://icml.cc/virtual/2025/poster/43761)
- [MedPlan:A Two-Stage RAG-Based System for Personalized Medical Plan Generation](https://arxiv.org/abs/2503.17900)
- [PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents](https://arxiv.org/abs/2505.01592)
- [RGAR: Recurrence Generation-augmented Retrieval for Factual-aware Medical Question Answering](https://arxiv.org/abs/2502.13361)
- [End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](https://arxiv.org/abs/2508.15746)
- [Labeling-free RAG-enhanced LLM for intelligent fault diagnosis via reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1474034625007578)
- [The Helicobacter pylori AI-clinician harnesses artificial intelligence to personalise H. pylori treatment recommendations](https://www.nature.com/articles/s41467-025-61329-5)
- [Continual contrastive reinforcement learning: Towards stronger agent for environment-aware fault diagnosis of aero-engines through long-term optimization under highly imbalance scenarios](https://www.sciencedirect.com/science/article/pii/S1474034625001909)
- [Integration of Multi-Source Medical Data for Medical Diagnosis Question Answering](https://ieeexplore.ieee.org/document/10752912)
- [Stage-Aware Hierarchical Attentive Relational Network for Diagnosis Prediction](https://ieeexplore.ieee.org/document/10236511)
- [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://arxiv.org/abs/2407.05131)
#### 🧠Memory
- [M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall in Large Language Models via Question Answering](https://aclanthology.org/2024.findings-acl.238/)
- [PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents](https://arxiv.org/abs/2505.01592)
- [EMRs2CSP : Mining Clinical Status Pathway from Electronic Medical Records](https://aclanthology.org/2025.findings-acl.886/)
- [Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-long.1381/)
- [MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot](https://arxiv.org/abs/2502.04413)
- [CardioTRAP: Design of a Retrieval Augmented System (RAG) for Clinical Data in Cardiology](https://ieeexplore.ieee.org/document/11081642)
- [CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs](https://arxiv.org/abs/2507.06715)
- [HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33301)
- [Listening to Patients: Detecting and Mitigating Patient Misreport in Medical Dialogue System](https://aclanthology.org/2025.findings-acl.135/)
- [End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](https://arxiv.org/abs/2508.15746)
#### 🧰Action
- [CardioTRAP: Design of a Retrieval Augmented System (RAG) for Clinical Data in Cardiology](https://ieeexplore.ieee.org/document/11081642)
- [CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs](https://arxiv.org/abs/2507.06715)
- [HI-DR: Exploiting Health Status-Aware Attention and an EHR Graph+ for Effective Medication Recommendation](https://ojs.aaai.org/index.php/AAAI/article/view/33301)
- [Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-long.1381/)
- [MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot](https://arxiv.org/abs/2502.04413)
- [KPL: Training-Free Medical Knowledge Mining of Vision-Language Models](https://arxiv.org/abs/2501.11231)
- [End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](https://arxiv.org/abs/2508.15746)
- [SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?](https://arxiv.org/abs/2502.13233)
- [Enhancing medical information retrieval: Re-engineering the tala-med search engine for improved performance and flexibility](https://pubmed.ncbi.nlm.nih.gov/40991950/)
- [Designing a Distributed LLM-Based Search Engine as a Foundation for Agent Discovery](https://ieeexplore.ieee.org/document/10967406)
- [How the Algorithmic Transparency of Search Engines Influences Health Anxiety: The Mediating Effects of Trust in Online Health Information Search](https://dl.acm.org/doi/10.1145/3706598.3713199)
- [Transforming Medical Data Access: The Role and Challenges of Recent Language Models in SQL Query Automation](https://www.semanticscholar.org/paper/Transforming-Medical-Data-Access%3A-The-Role-and-of-Tankovi%C4%87-%C5%A0ajina/bb37f6df4218feb32cf86bb95fc5a5cc95465a18)
- [Improving Interactive Diagnostic Ability of a Large Language Model Agent Through Clinical Experience Learning](https://arxiv.org/abs/2503.16463)
- [Designing VR Simulation System for Clinical Communication Training with LLMs-Based Embodied Conversational Agents](https://arxiv.org/abs/2503.01767)
#### 👥Cooperation
- [Enhancing Clinical Trial Patient Matching through Knowledge Augmentation and Reasoning with Multi-Agent](https://arxiv.org/abs/2411.14637)
- [TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork](https://arxiv.org/abs/2508.08115)
- [ClinicalLab: Aligning Agents for Multi-Departmental Clinical Diagnostics in the Real World](https://arxiv.org/abs/2406.13890)
- [The Optimization Paradox in Clinical AI Multi-Agent Systems](https://arxiv.org/abs/2506.06574)
#### ⏫Self-evolution
- [EvoAgentX: An Automated Framework for Evolving Agentic Workflows](https://arxiv.org/abs/2507.03616)
- [MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning](https://arxiv.org/abs/2508.00271)
- [ZERA: Zero-init Instruction Evolving Refinement Agent -- From Zero Instructions to Structured Prompts via Principle-based Optimization](https://arxiv.org/abs/2509.18158)
- [HealthBranches: Synthesizing Clinically-Grounded Question Answering Datasets via Decision Pathways](https://arxiv.org/abs/2508.07308)
- [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](https://arxiv.org/abs/2507.21046)
- [Evolving Collective Cognition in Human-Agent Hybrid Societies: How Agents Form Stances and Boundaries](https://arxiv.org/abs/2508.17366)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xqz614/Awesome-Agentic-Clinical-Dialogue&type=date&legend=top-left)](https://www.star-history.com/#xqz614/Awesome-Agentic-Clinical-Dialogue&type=date&legend=top-left)


## 🤝 Contributing
Your contributions are always welcome! Please contact [Xiaoquan Zhi](https://github.com/xqz614) or [Chuang Zhao](https://github.com/Data-Designer)
## ✍️ Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@inproceedings{ADM2025reinv,
  title={Reinventing Clinical Dialogue: Agentic Paradigms for LLM‑Enabled Healthcare Communication},
  author={ADM Lab},
  year={2025}
}
```
