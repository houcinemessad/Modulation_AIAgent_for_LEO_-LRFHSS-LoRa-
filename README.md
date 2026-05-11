# Modulation_AIAgent_for_LEO_-LRFHSS-LoRa-
Generative AI Agent (LangChain/RAG) optimizing LEO satellite links. It analyzes geometry (elevation, Doppler) to switch between LR-FHSS (DR8/9) and LoRa (SF7-12). Goal: minimize PER (>25% gain) with &lt;500ms latency. Includes 1000+ scenario CSV dataset, real-time Dashboard, and REST API.

## Repository structure

- `/src` : Code de l'agent LangChain et logique RAG.
- `/data` : Le dataset CSV de 1000 scénarios.
- `/notebooks` : Pour tes analyses exploratoires et visualisations du PER.
- `/dashboard` : Code de l'interface interactive.
- `/.github/workflows` : Pour tes scripts CI/CD (GitHub Actions).
