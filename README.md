# Modulation_AIAgent_for_LEO_-LRFHSS-LoRa-
Generative AI Agent (LangChain/RAG) optimizing LEO satellite links. It analyzes geometry (elevation, Doppler) to switch between LR-FHSS (DR8/9) and LoRa (SF7-12). Goal: minimize PER (>25% gain) with &lt;500ms latency. Includes 1000+ scenario CSV dataset, real-time Dashboard, and REST API.

## Repository Structure

- `/src`: LangChain agent code and RAG logic.
- `/data`: Directory for the CSV dataset of 1000 scenarios.
- `/notebooks`: Exploratory analysis notebooks and PER visualizations.
- `/dashboard`: Interactive dashboard application code.
- `/.github/workflows`: CI/CD workflow scripts (GitHub Actions).
