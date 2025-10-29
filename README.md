# ![Routesit Banner](routesitbanner/Routesit.png)


# Routesit — Advanced Road Safety Decision System

**Built for:** National Road Safety Hackathon 2025, IIT Madras  

**Team:** [MechaSys]  

---

## 🛣 Project Overview

**Routesit** is a locally operable AI tool designed to assist engineers and road authorities in **planning, prioritizing, and simulating road safety interventions**. Unlike a simple GPT bot, Routesit:

- Integrates **text input, optional images, and road metadata**
- Retrieves and reasons over an **expanded, evidence-backed intervention dataset**
- Optimizes interventions based on **cost, impact, dependencies, and conflicts**
- Produces **actionable, field-ready recommendations** with citations
- Visualizes **scenario comparisons** for informed decision-making

Routesit is designed to be **offline, lightweight, and demonstration-ready**, highlighting technical depth and practical feasibility for real-world road safety planning.

---

## ⚙ How It Works

**Input Types:**
- Free form text describing road safety issues
- Optional images for visual analysis (e.g., faded signs or markings)
- Metadata: road type, speed limit, traffic volume, accident history

**Processing Pipeline:**
1. **Data Interpretation & Normalization** — cleans input, parses metadata, optional image analysis (YOLOv8)
2. **Intelligent Retrieval** — searches vectorized intervention database (FAISS/Chroma embeddings)
3. **Local Reasoning** — LLM ranks interventions, predicts impact, flags dependencies/conflicts
4. **Scenario Optimization** — compares multiple intervention combinations (cost/impact/urgency)
5. **Output Generation** — produces interactive reports, charts, step-by-step instructions, and citations

---

## 🛠 Tech Stack

- **Language:** Python  
- **Embeddings & Retrieval:** sentence-transformers, FAISS/Chroma  
- **LLM Reasoning:** LLaMA 3 8B Quantized 4bit - 10 bit  (local, quantized)  
- **Computer Vision (optionally under development):** YOLOv8 for sign and road feature detection  
- **Dependency/Conflict Graph:** NetworkX  
- **Visualization:** matplotlib / plotly  
- **Frontend Demo:** Streamlit / Flask
- Not Fully Classified (Under Development)

---

## 📂 Dataset

- Base: Hackathon-provided 50 interventions  
- Expanded: Additional interventions curated from **IRC, MoRTH, WHO, and global best practices etc.**  
- Annotated with: **dependencies, cost brackets, predicted impact, implementation complexity, and references**  

> Quality > quantity,  each intervention entry is actionable, referenced, and ready for scenario simulation.

---

## 💡 Key Features

- **Multi-modal Input Fusion:** Text + photo + metadata interpretation  
- **Dependency & Conflict Reasoning:** Ensures interventions are compatible and feasible  
- **Scenario Simulation:** Compare “A+B” vs “C+D” to predict crash reduction and cost-effectiveness  
- **Actionable Output:** Generates field-ready recommendations with justification and citations  
- **Offline Operation:** Runs locally on modest hardware for hackathon demo
---



## 📜 References

- IRC:67-2022, Clause 14.4  
- MoRTH Guidelines  
- WHO Road Safety Reports  
- Public domain international best practices  

---

## 📞 Contact

**Members:** Divine R | Anand S 
**Email:** [mechainthemail@gmail.com]  
**Hackathon Submission:** National Road Safety Hackathon 2025, IIT Madras

