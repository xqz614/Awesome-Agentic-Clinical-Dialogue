# 🤖 Awesome-Agentic-Clinical-Dialogue
This repo includes papers about methods related to agentic clinical dialogue. We believe that the agentic paradi is still a largely unexplored area, and we hope this repository will provide you with some valuable insights!

Read our survey paper here: [Reinventing Clinical Dialogue: Agentic Paradigms for LLM‑Enabled Healthcare Communication]()
## 📘 Overview
This framework facilitates a systematic analysis of the intrinsic trade-offs between creativity and reliability by categorizing methods into four archetypes: Latent Space Clinicians, Emergent Planners, Grounded Synthesizers, and Veriffable Workffow Automators. For each paradigm, we deconstruct the technical realization across the entire cognitive pipeline, encompassing strategic planning, memory management, action execution, collaboration, and evolution, to reveal how distinct architectural choices balance the tension between autonomy and safety. Furthermore, we bridge abstract design philosophies with the pragmatic implementation ecosystem. By mapping real-world applications to our taxonomy and systematically reviewing benchmarks and evaluation metrics speciffc to clinical agents, we provide a comprehensive reference for future development.
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
#### 🧠Memory
#### 🧰Action
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
#### 🧠Memory
#### 🧰Action
#### 👥Cooperation
#### ⏫Self-evolution
## 🤝 Contributing

## ✍️ Citation
