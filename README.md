# ![Routesit Banner]([https://link.url.image.png](https://media.discordapp.net/attachments/1420092566715633784/1432759384923963415/568631390_1825759678079625_4856681336625592065_n.png?ex=690238a8&is=6900e728&hm=2efd4e89354fbe90c4e45c347bf0aa07b68650cd09560333016119f088e27c6a&=&format=webp&quality=lossless&width=1280&height=800))

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
- **LLM Reasoning:** LLaMA 2 7B (local, quantized)  
- **Computer Vision (optional):** YOLOv8 for sign and road feature detection  
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


**Email:** [mechainthemail@gmail.com]  
**Hackathon Submission:** National Road Safety Hackathon 2025, IIT Madras

