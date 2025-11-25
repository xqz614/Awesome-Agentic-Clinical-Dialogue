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
#### 👥Cooperation
#### ⏫Self-evolution
### 🤖**EP**
#### 📊Planning
#### 🧠Memory
#### 👥Cooperation
#### ⏫Self-evolution
### 🤖**GS**
#### 📊Planning
#### 🧠Memory
#### 🧰Action
#### 👥Cooperation
#### ⏫Self-evolution
### 🤖**VWA**
#### 📊Planning
#### 🧠Memory
#### 🧰Action
#### 👥Cooperation
#### ⏫Self-evolution
## 🤝 Contributing

## ✍️ Citation
