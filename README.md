# 🤖 Awesome-Agentic-Clinical-Dialogue
This repo includes papers about methods related to agentic clinical dialogue. We believe that the agentic paradi is still a largely unexplored area, and we hope this repository will provide you with some valuable insights!

Read our survey paper here: [Reinventing Clinical Dialogue: Agentic Paradigms for LLM‑Enabled Healthcare Communication](https://arxiv.org/abs/2512.01453)
## 📘 Overview
This framework facilitates a systematic analysis of the intrinsic trade-offs between creativity and reliability by categorizing methods into four archetypes: Latent Space Clinicians, Emergent Planners, Grounded Synthesizers, and Veriffable Workffow Automators. For each paradigm, we deconstruct the technical realization across the entire cognitive pipeline, encompassing strategic planning, memory management, action execution, collaboration, and evolution, to reveal how distinct architectural choices balance the tension between autonomy and safety. Furthermore, we bridge abstract design philosophies with the pragmatic implementation ecosystem. By mapping real-world applications to our taxonomy and systematically reviewing benchmarks and evaluation metrics speciffc to clinical agents, we provide a comprehensive reference for future development.
<p align="center"><img src="https://github.com/xqz614/Awesome-Agentic-Clinical-Dialogue/blob/main/image/taxonomy.png" height="500px"></p>

## 📁 Table of Contents
- [Key Categories](#-key-categories)
- [Resource List](#-resource-list)
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
## 📖 Resource List
### 🤖**LSC**
#### 📊Planning
- [BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://arxiv.org/abs/2210.10341)
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
### Dataset

| Dataset Name | Time (Pub) | Downstream Task | Brief Description | Source (Paper/Link) |
| :--- | :--- | :--- | :--- | :--- |
| **I. QA Dialogue** | | | | |
| **MedQA** | 2020 | Medical Examination (QA) | Large-scale multiple-choice questions collected from professional medical board exams (USMLE, Mainland China, Taiwan). | [Jin et al., 2020](https://arxiv.org/abs/2009.13081) |
| **MedMCQA** | 2022 | Medical Examination (QA) | Large-scale, multiple-choice QA dataset derived from Indian medical entrance examinations (AIIMS/NEET). | [Pal et al., 2022](https://proceedings.mlr.press/v174/pal22a.html) |
| **cMedQA2** | 2019 | QA / Retrieval | Chinese medical QA dataset with queries and answers from online health counseling platforms. | [Zhang et al., 2019](https://github.com/zhangsheng93/cMedQA2) |
| **CMExam** | 2023 | Medical Examination (QA) | 60K+ multiple-choice questions from the Chinese National Medical Licensing Examination with detailed annotations. | [Liu et al., 2023](https://arxiv.org/abs/2306.03030) |
| **Medbullets** | 2024 | Medical Examination (QA) | High-quality USMLE Step 2 & 3 style questions with expert-written explanations for reasoning evaluation. | [Chen et al., 2024](https://arxiv.org/abs/2402.18060) |
| **HeadQA** | 2019 | Medical Examination (QA) | Multiple-choice questions from Spanish healthcare exams (MIR, EIR, etc.) for testing complex reasoning. | [Vilares et al., 2019](https://aclanthology.org/P19-1092/) |
| **Huatuo-26M** | 2023 | Medical Examination / QA | Massive Chinese medical QA dataset with 26 million QA pairs, used for pre-training and retrieval. | [Li et al., 2023](https://arxiv.org/abs/2305.01526) |
| **CasiMedicos-Arg**| 2024 | Medical Examination (QA) | Multilingual dataset (ES, EN, FR, IT) annotated with explanatory argumentative structures for clinical cases. | [Sviridova et al., 2024](https://aclanthology.org/2024.emnlp-main.1023/) |
| **PubMedQA** | 2019 | Literature-based QA | Biomedical QA task to answer "yes/no/maybe" from PubMed abstracts. | [Jin et al., 2019](https://arxiv.org/abs/1909.06146) |
| **CliCR** | 2018 | Literature-based QA | Dataset of clinical case reports designed for machine reading comprehension. | [Suster et al., 2018](https://arxiv.org/abs/1803.09102) |
| **MEDIQA-2019** | 2019 | Literature-based QA | Shared task data focusing on NLI, RQE (Recognizing Question Entailment), and QA in the medical domain. | [Ben Abacha et al., 2019](https://www.aclweb.org/anthology/W19-5001/) |
| **BioASQ** | 2013-2023| Literature-based QA | Long-running challenge series for large-scale biomedical semantic indexing and question answering. | [Tsatsaronis et al.](http://bioasq.org/) |
| **Medical Meadow** | 2023 | Literature-based / Tuning | A collection of various medical tasks (Flashcards, Wikidoc) reformatted for instruction tuning. | [Kbressem et al., 2023](https://github.com/kbressem/medAlpaca) |
| **MASH-QA** | 2020 | Consumer Health QA | Multiple-span extraction QA dataset for consumer health questions (e.g., from WebMD). | [Zhu et al., 2020](https://arxiv.org/abs/2004.03008) |
| **HealthQA** | 2023 | Consumer Health QA | Dataset focusing on reliability and helpfulness of health answers. | [Zhu et al., 2023](https://arxiv.org/abs/2302.06871) |
| **AfriMed-QA** | 2025 | Domain-specific QA | Pan-African medical QA benchmark (15k Qs) covering 32 specialties and local context from 16 countries. | [Nimo et al., 2025](https://arxiv.org/abs/2502.x) |
| **MedCalc-Bench** | 2024 | Domain-specific QA | Benchmark for evaluating LLMs on medical calculations (formulas, scores) with patient notes. | [Wang et al., 2024](https://arxiv.org/abs/2406.12683) |
| **MedHallu** | 2025 | QA / Hallucination | 10k Q-A pairs derived from PubMedQA annotated to detect and categorize medical hallucinations. | [Gao et al., 2025](https://arxiv.org/abs/2502.14691) |
| **MedicationQA** | 2019 | Consumer Health QA | Dataset of consumer questions about medications (drug interactions, dosage) with expert answers. | [Ben Abacha et al., 2019](https://arxiv.org/abs/1908.10023) |
| **RJUA-MedDQA** | 2024 | QA / Multimodal | Multimodal benchmark for medical document understanding (images/reports) and clinical reasoning. | [Jin et al., 2024](https://arxiv.org/abs/2402.14840) |
| **II. Task-oriented Dialogue** | | | | |
| **MedDialog** | 2020 | Symptom Diagnosis | Massive dataset (English/Chinese) of doctor-patient conversations scraped from online platforms. | [Zeng et al., 2020](https://arxiv.org/abs/2004.03329) |
| **DialoAMC** | 2023 | Symptom Diagnosis | Dataset for Automated Medical Consultation focusing on symptom elicitation and diagnosis. | [Chen et al., 2023](https://doi.org/10.1145/3539618.3591901) |
| **MedDG** | 2022 | Symptom Diagnosis | High-quality entity-annotated medical dialogue dataset for diagnosis and treatment recommendation. | [Liu et al., 2022](https://arxiv.org/abs/2010.07497) |
| **MZ** (Muzhi) | 2018 | Symptom Diagnosis | Chinese medical dialogue dataset from the "Muzhi" platform for self-diagnosis agents. | [Wei et al., 2018](https://www.aclweb.org/anthology/P18-1216/) |
| **CMDD** | 2019 | Symptom Diagnosis | Chinese Medical Diagnostic Dialogue dataset (Pediatrics) with symptom-disease mappings. | [Lin et al., 2019](https://arxiv.org/abs/1902.04588) |
| **DX** (DXY) | 2019 | Symptom Diagnosis | Diagnostic dataset from DXY.cn, containing dialogue sessions with explicit symptom transitions. | [Xu et al., 2019](https://arxiv.org/abs/1908.02402) |
| **CovidDialog** | 2020 | Symptom Diagnosis (COVID) | Dialogues specifically regarding COVID-19 consultations, scraped during the pandemic. | [Zhou et al., 2020](https://arxiv.org/abs/2007.01977) |
| **Ext-CovidDialog**| 2023 | Symptom Diagnosis (COVID) | Extended version of CovidDialog with more data covering evolving variants and scenarios. | [Varshney et al., 2023](https://aclanthology.org/2023.bionlp-1.11/) |
| **IMCS-21** | 2021 | Symptom Diagnosis | Interactive Medical Consultation System dataset; focuses on multi-turn diagnostic dialogue. | [Chen et al., 2021](https://arxiv.org/abs/2105.02672) |
| **BC5CDR** | 2015 | Entity Recognition | BioCreative V task dataset for Chemical-Disease Relation extraction (NER/RE). | [Li et al., 2016](https://academic.oup.com/database/article/2016/1/baw068/2630414) |
| **NCBI-Disease** | 2014 | Entity Recognition | Corpus of PubMed abstracts annotated with disease mentions for NER. | [Doğan et al., 2014](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/) |
| **PHEE** | 2022 | Entity Extraction | Pharmacovigilance Event Extraction dataset for identifying adverse drug events from text. | [Sun et al., 2022](https://aclanthology.org/2022.emnlp-main.373/) |
| **MedAlign** | 2023 | Instruction Following | Clinician-generated dataset for instruction following (summaries, questions) based on EHR data. | [Fleming et al., 2023](https://arxiv.org/abs/2308.14089) |
| **MedInstruct** | 2023 | Instruction Following | Dataset of 52k medical instructions constructed from existing datasets (e.g., MedQA) for tuning. | [Zhang et al., 2023](https://arxiv.org/abs/2304.09842) |
| **BianqueCorpus** | 2023 | Instruction Following | Large-scale multi-turn Chinese health conversation dataset with balanced questioning/suggestions. | [Chen et al., 2023](https://arxiv.org/abs/2306.03030) |
| **MedSynth** | 2025 | Generation / Summarization | Synthetic medical dialogue-note pairs designed to advance dialogue-to-note and note-to-dialogue tasks. | [Rezaei et al., 2025](https://arxiv.org/abs/2508.01401) |
| **MeQSum** | 2019 | Summarization / Instruction | Dataset for summarizing consumer health questions into canonical medical questions. | [Abacha et al., 2019](https://www.aclweb.org/anthology/P19-1215/) |
| **III. Recommendation Dialogue**| | | | |
| **DialMed** | 2022 | Recommendation (Drug) | Dialogue dataset designed for medication recommendation based on patient history/dialogue. | [He et al., 2022](https://arxiv.org/abs/2202.08779) |
| **ReMeDi** | 2021 | Recommendation | "Resources for Medical Dialogue"; focuses on movie/medical recommendation scenarios. | [Yan et al., 2021](https://aclanthology.org/2021.emnlp-main.288/) |
| **MIMIC-III** | 2016 | Database (Source) | Large database of de-identified health-related data (EHRs) used to construct recommendation tasks. | [Johnson et al., 2016](https://www.nature.com/articles/sdata201635) |
| **DrugBank** | - | Knowledge Base (Source) | Comprehensive database containing information on drugs and drug targets, used for grounding recommendations. | [Wishart et al.](https://go.drugbank.com/) |
| **ProKnow-data** | 2020 | Recommendation | Data used for proactive knowledge-grounded dialogue, often adapted for medical contexts. | [Liu et al., 2020](https://arxiv.org/abs/2010.13328) |
| **IV. Supportive Dialogue** | | | | |
| **EmpatheticDialogues**| 2019 | General Empathetic | Large dataset of 25k conversations grounded in emotional situations (general domain). | [Rashkin et al., 2019](https://arxiv.org/abs/1811.00207) |
| **MELD** | 2019 | General Empathetic | Multimodal EmotionLines Dataset; textual/audio/visual emotion recognition. | [Poria et al., 2019](https://arxiv.org/abs/1810.02508) |
| **PsyQA** | 2021 | Mental Health Support | Chinese dataset of psychological health support (Q&A) with strategy annotations. | [Sun et al., 2021](https://arxiv.org/abs/2106.01702) |
| **ESConv** | 2021 | Mental Health Support | Emotional Support Conversation dataset designed to train agents in empathy and support strategies. | [Liu et al., 2021](https://arxiv.org/abs/2106.01144) |
| **SoulChat-Corpus** | 2023 | Mental Health Support | Large-scale Chinese dataset for single-turn and multi-turn empathetic psychological counseling. | [Chen et al., 2023](https://arxiv.org/abs/2304.09842) |
| **MTS-Dialogue** | 2023 | Clinical Support/Summ. | 1.7k doctor-patient conversations paired with corresponding clinical note summaries. | [Abacha et al., 2023](https://aclanthology.org/2023.eacl-main.168/) |
| **SMILECHAT** | 2023 | Mental Health Support | Dataset for mental health support focusing on cognitive distortion detection and reframing. | [Qiu et al., 2023](https://arxiv.org/abs/2304.09842) |
| **V. Hybrid Function** | | | | |
| **MidMed** | 2023 | Hybrid (Diag/Rec/Chat) | Mixed-type dialogue corpus covering diagnosis, recommendation, QA, and chitchat in one session. | [Shi et al., 2023](https://arxiv.org/abs/2306.02923) |
| **MedEval** | 2023 | Evaluation Benchmark | Multi-level, multi-task benchmark spanning 35 body regions and 8 exam modalities for LLM eval. | [He et al., 2023](https://arxiv.org/abs/2310.14088) |
| **MedTrinity-25M** | 2024 | Multimodal / Hybrid | Massive multimodal dataset (25M images) with multigranular annotations (Image-ROI-Text). | [Xie et al., 2024](https://arxiv.org/abs/2408.02900) |
| **MENTAT** | 2025 | Mental Health / Hybrid | Clinician-annotated benchmark for complex psychiatric decision-making (diagnosis, triage, etc.). | [Stanford, 2025](https://hai.stanford.edu/research/mentat) |
| **MedAlpaca** | 2023 | Instruction Tuning | Collection of datasets (see Medical Meadow) used to train the MedAlpaca model series. | [Han et al., 2023](https://arxiv.org/abs/2304.08247) |
| **NoteChat** | 2023 | Generation / Hybrid | Synthetic patient-physician conversations conditioned on clinical notes (Note-to-Dialogue). | [Wang et al., 2023](https://arxiv.org/abs/2310.15959) |
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
