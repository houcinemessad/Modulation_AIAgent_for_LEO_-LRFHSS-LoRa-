"""Contient les trois prompts spécialisés de l'agent."""
import json
#########################
#Document en anglais svp#
#########################

def geometry_analysis_prompt_zeroshot():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power)  
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt
#Sortie en JSON 
    return """[INST] You are an embedded NTN LoRa-Satellite channel analyzer.
Evaluate the 3 input metrics and output a JSON object classifying the radio channel severity from 1 (Excellent) to 5 (Critical).

INPUT METRICS:
- ELEVATION: {ELEVATION} degrees
- DOPPLER: {DOPPLER} kHz
- RX_POWER: {RX_POWER} dBm

CRITICAL RULES:
1. Max Severity (5) applies immediately if ANY of these conditions are met:
   - ELEVATION < 15
   - ABS(DOPPLER) > 40
   - RX_POWER <= -120
2. Min Severity (1) applies only if ALL of these conditions are met:
   - ELEVATION >= 45
   - ABS(DOPPLER) < 10
   - RX_POWER >= -90
3. Scale severity from 2 to 4 for intermediate values.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations. 
Follow this exact structure:
{
  "elevation_status": "critical" or "nominal",
  "doppler_status": "critical" or "nominal",
  "rx_status": "critical" or "nominal",
  "severity": 1, 2, 3, 4, or 5
}
[/INST]
{

"""

def geometry_analysis_prompt_fewshot():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power) + 2-3 exemples concrets injectés
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt + 2-3 exemples concrets injectés
#ex : "élévation 12°, Doppler 18kHz → sévérité 5"
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite channel analyzer.
Evaluate the 3 input metrics and output a JSON object classifying the radio channel severity from 1 (Excellent) to 5 (Critical).

INPUT METRICS:
- ELEVATION: {ELEVATION} degrees
- DOPPLER: {DOPPLER} kHz
- RX_POWER: {RX_POWER} dBm

CRITICAL RULES:
1. Max Severity (5) applies immediately if ANY of these conditions are met:
   - ELEVATION < 15
   - ABS(DOPPLER) > 40
   - RX_POWER <= -120
2. Min Severity (1) applies only if ALL of these conditions are met:
   - ELEVATION >= 45
   - ABS(DOPPLER) < 10
   - RX_POWER >= -90
3. Scale severity from 2 to 4 for intermediate values.

EXAMPLES OF EXPECTED OUTPUTS (FEW-SHOT):

Example 1 (Critical Case - Severity 5):
Input: ELEVATION = 11, DOPPLER = 12, RX_POWER = -95
Output: {
  "elevation_status": "critical",
  "doppler_status": "nominal",
  "rx_status": "nominal",
  "severity": 5
}

Example 2 (Nominal Case - Severity 1):
Input: ELEVATION = 52, DOPPLER = 4, RX_POWER = -85
Output: {
  "elevation_status": "nominal",
  "doppler_status": "nominal",
  "rx_status": "nominal",
  "severity": 1
}

Example 3 (Intermediate Case - Severity 3):
Input: ELEVATION = 30, DOPPLER = 22, RX_POWER = -105
Output: {
  "elevation_status": "nominal",
  "doppler_status": "nominal",
  "rx_status": "nominal",
  "severity": 3
}

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations. 
Follow this exact structure:
{
  "elevation_status": "critical" or "nominal",
  "doppler_status": "critical" or "nominal",
  "rx_status": "critical" or "nominal",
  "severity": 1, 2, 3, 4, or 5
}
[/INST]
{
"""

def geometry_analysis_prompt_COT():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power)  
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt + instruction "raisonne étape par étape avant de conclure"
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite channel analyzer.
Evaluate the 3 input metrics and output a JSON object classifying the radio channel severity from 1 (Excellent) to 5 (Critical).

INPUT METRICS:
- ELEVATION: {ELEVATION} degrees
- DOPPLER: {DOPPLER} kHz
- RX_POWER: {RX_POWER} dBm

CRITICAL RULES:
1. Max Severity (5) applies immediately if ANY of these conditions are met:
   - ELEVATION < 15
   - ABS(DOPPLER) > 40
   - RX_POWER <= -120
2. Min Severity (1) applies only if ALL of these conditions are met:
   - ELEVATION >= 45
   - ABS(DOPPLER) < 10
   - RX_POWER >= -90
3. Scale severity from 2 to 4 for intermediate values.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations. 
Follow this exact structure:
{
  "elevation_status": "critical" or "nominal",
  "doppler_status": "critical" or "nominal",
  "rx_status": "critical" or "nominal",
  "thought": "Short 1-sentence step-by-step logic checking each rule",
  "severity": 1, 2, 3, 4, or 5
}
[/INST]
{
"""

def per_prediction_via_embedding_prompt_zeroshot():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite performance predictor.
Estimate the exact Packet Error Rate (PER) as an INTEGER (0 to 100) for 8 specific modulations based on current real-time metrics and 3 historical reference simulations.

CURRENT REAL-TIME METRICS:
- SEVERITY: {SEVERITY}
- ELEVATION: {ELEVATION}
- DOPPLER: {DOPPLER}
- RX_POWER: {RX_POWER}

HISTORICAL REFERENCE SIMULATIONS (RAG CONTEXT):
{RAG_CONTEXT}

PREDICTION RULES:
1. Compare CURRENT METRICS with SIM 1, SIM 2, and SIM 3 metrics (focus heavily on SEVERITY and RX_POWER).
2. Interpolate the PER values logic: if current metrics fall between two simulations, the predicted PER must fall proportionally between them.
3. Keep the payload constraints in mind: LoRa SF12 and LR-FHSS DR8/DR9 are more robust (lower PER) than LoRa SF7 under high stress.
4. Output every PER as a pure integer representing the percentage (e.g., 15 instead of "15%").

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "per_sf7": 0,
  "per_sf8": 0,
  "per_sf9": 0,
  "per_sf10": 0,
  "per_sf11": 0,
  "per_sf12": 0,
  "per_dr8": 0,
  "per_dr9": 0
}
[/INST]
{
"""
def per_prediction_via_embedding_prompt_fewshot():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation avec qlq exemples de résultats
#ex : SNR et Linkbudget de exemple 1
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite performance predictor.
Estimate the exact Packet Error Rate (PER) as an INTEGER (0 to 100) for 8 specific modulations based on current real-time metrics and 3 historical reference simulations.

CURRENT REAL-TIME METRICS:
- SEVERITY: {SEVERITY}
- ELEVATION: {ELEVATION}
- DOPPLER: {DOPPLER}
- RX_POWER: {RX_POWER}

HISTORICAL REFERENCE SIMULATIONS (RAG CONTEXT):
{RAG_CONTEXT}

PREDICTION RULES:
1. Compare CURRENT METRICS with SIM 1, SIM 2, and SIM 3 metrics (focus heavily on SEVERITY and RX_POWER).
2. Interpolate the PER values logic: if current metrics fall between two simulations, the predicted PER must fall proportionally between them.
3. Keep the payload constraints in mind: LoRa SF12 and LR-FHSS DR8/DR9 are more robust (lower PER) than LoRa SF7 under high stress.
4. Output every PER as a pure integer representing the percentage (e.g., 15 instead of "15%").

EXAMPLES OF EXPECTED OUTPUTS (FEW-SHOT):

Example 1:
Input Metrics: SEVERITY = 3, ELEVATION = 25, DOPPLER = 15, RX_POWER = -100
RAG Context:
SIM 1: SEV:3, ELEV:22, DOPPLER:12, RX:-102, SF7_PER:15%, SF8_PER:10%, SF9_PER:5%, SF10_PER:2%, SF11_PER:1%, SF12_PER:0%, DR8_PER:2%, DR9_PER:4%
SIM 2: SEV:4, ELEV:16, DOPPLER:25, RX:-112, SF7_PER:85%, SF8_PER:60%, SF9_PER:40%, SF10_PER:20%, SF11_PER:10%, SF12_PER:5%, DR8_PER:8%, DR9_PER:12%
SIM 3: SEV:5, ELEV:11, DOPPLER:42, RX:-122, SF7_PER:100%, SF8_PER:100%, SF9_PER:95%, SF10_PER:80%, SF11_PER:60%, SF12_PER:35%, DR8_PER:15%, DR9_PER:25%
Output: {
  "per_sf7": 12,
  "per_sf8": 8,
  "per_sf9": 4,
  "per_sf10": 1,
  "per_sf11": 0,
  "per_sf12": 0,
  "per_dr8": 1,
  "per_dr9": 3
}

Example 2:
Input Metrics: SEVERITY = 4, ELEVATION = 14, DOPPLER = 30, RX_POWER = -115
RAG Context:
SIM 1: SEV:3, ELEV:22, DOPPLER:12, RX:-102, SF7_PER:15%, SF8_PER:10%, SF9_PER:5%, SF10_PER:2%, SF11_PER:1%, SF12_PER:0%, DR8_PER:2%, DR9_PER:4%
SIM 2: SEV:4, ELEV:16, DOPPLER:25, RX:-112, SF7_PER:85%, SF8_PER:60%, SF9_PER:40%, SF10_PER:20%, SF11_PER:10%, SF12_PER:5%, DR8_PER:8%, DR9_PER:12%
SIM 3: SEV:5, ELEV:11, DOPPLER:42, RX:-122, SF7_PER:100%, SF8_PER:100%, SF9_PER:95%, SF10_PER:80%, SF11_PER:60%, SF12_PER:35%, DR8_PER:15%, DR9_PER:25%
Output: {
  "per_sf7": 92,
  "per_sf8": 78,
  "per_sf9": 65,
  "per_sf10": 47,
  "per_sf11": 32,
  "per_sf12": 18,
  "per_dr8": 11,
  "per_dr9": 18
}

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "per_sf7": 0,
  "per_sf8": 0,
  "per_sf9": 0,
  "per_sf10": 0,
  "per_sf11": 0,
  "per_sf12": 0,
  "per_dr8": 0,
  "per_dr9": 0
}
[/INST]
{
"""

def per_prediction_via_embedding_prompt_COT():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation avec raisonnement contextualisé
#ex : prompt + instructions de raisonnement sur les données chromaDB
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite performance predictor.
Estimate the exact Packet Error Rate (PER) as an INTEGER (0 to 100) for 8 specific modulations based on current real-time metrics and 3 historical reference simulations.

CURRENT REAL-TIME METRICS:
- SEVERITY: {SEVERITY}
- ELEVATION: {ELEVATION}
- DOPPLER: {DOPPLER}
- RX_POWER: {RX_POWER}

HISTORICAL REFERENCE SIMULATIONS (RAG CONTEXT):
{RAG_CONTEXT}

PREDICTION RULES:
1. Compare CURRENT METRICS with SIM 1, SIM 2, and SIM 3 metrics (focus heavily on SEVERITY and RX_POWER).
2. Interpolate the PER values logic: if current metrics fall between two simulations, the predicted PER must fall proportionally between them.
3. Keep the payload constraints in mind: LoRa SF12 and LR-FHSS DR8/DR9 are more robust (lower PER) than LoRa SF7 under high stress.
4. Output every PER as a pure integer representing the percentage (e.g., 15 instead of "15%").

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "interpolation_logic": "1-sentence analysis comparing current metrics to SIM 1, 2, 3",
  "per_sf7": 0,
  "per_sf8": 0,
  "per_sf9": 0,
  "per_sf10": 0,
  "per_sf11": 0,
  "per_sf12": 0,
  "per_dr8": 0,
  "per_dr9": 0
}
[/INST]
{
"""

def final_decision_prompt_zeroshot():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite modem controller.
Select the optimal modulation command based on real-time ELEVATION and PREDICTED_PER_JSON values.

INPUT DATA:
- ELEVATION: {ELEVATION} degrees
- PREDICTED_PER_JSON: {PREDICTED_PER_JSON}

STRICT COMMAND DICTIONARY:
- per_sf7  -> AT+MOD=LORA-SF7
- per_sf8  -> AT+MOD=LORA-SF8
- per_sf9  -> AT+MOD=LORA-SF9
- per_sf10 -> AT+MOD=LORA-SF10
- per_sf11 -> AT+MOD=LORA-SF11
- per_sf12 -> AT+MOD=LORA-SF12
- per_dr8  -> AT+MOD=LR-FHSS-DR8
- per_dr9  -> AT+MOD=LR-FHSS-DR9

DECISION LOGIC RULES:
1. HARD RULE (ELEVATION CHECK): If ELEVATION < 15, immediately select "AT+MOD=LR-FHSS-DR8" regardless of the JSON data.
2. NOMINAL RULE: If ELEVATION >= 15, read PREDICTED_PER_JSON, find the lowest integer value (minimum PER), and select its corresponding AT command from the dictionary. If there is a tie, select the faster modulation.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "selected_command": "AT+MOD=..."
}
[/INST]
{
"""

def final_decision_prompt_fewshot():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple + Exemple de choix de modulation selon le cas ex1 et ex2
#Sortie en JSON
    return """ [INST] You are an embedded NTN LoRa-Satellite modem controller.
Select the optimal modulation command based on real-time ELEVATION and PREDICTED_PER_JSON values.

INPUT DATA:
- ELEVATION: {ELEVATION} degrees
- PREDICTED_PER_JSON: {PREDICTED_PER_JSON}

STRICT COMMAND DICTIONARY:
- per_sf7  -> AT+MOD=LORA-SF7
- per_sf8  -> AT+MOD=LORA-SF8
- per_sf9  -> AT+MOD=LORA-SF9
- per_sf10 -> AT+MOD=LORA-SF10
- per_sf11 -> AT+MOD=LORA-SF11
- per_sf12 -> AT+MOD=LORA-SF12
- per_dr8  -> AT+MOD=LR-FHSS-DR8
- per_dr9  -> AT+MOD=LR-FHSS-DR9

DECISION LOGIC RULES:
1. HARD RULE (ELEVATION CHECK): If ELEVATION < 15, immediately select "AT+MOD=LR-FHSS-DR8" regardless of the JSON data.
2. NOMINAL RULE: If ELEVATION >= 15, read PREDICTED_PER_JSON, find the lowest integer value (minimum PER), and select its corresponding AT command from the dictionary. If there is a tie, select the faster modulation.

EXAMPLES OF EXPECTED OUTPUTS (FEW-SHOT):

Example 1 (Hard Safety Rule Triggered):
Input Elevation: ELEVATION = 12
Input JSON: {
  "per_sf7": 45, "per_sf8": 30, "per_sf9": 15, "per_sf10": 8, "per_sf11": 4, "per_sf12": 2, "per_dr8": 5, "per_dr9": 10
}
Output: {
  "selected_command": "AT+MOD=LR-FHSS-DR8"
}

Example 2 (Nominal Case - Lowest PER Selection):
Input Elevation: ELEVATION = 22
Input JSON: {
  "per_sf7": 25, "per_sf8": 18, "per_sf9": 12, "per_sf10": 6, "per_sf11": 3, "per_sf12": 1, "per_dr8": 2, "per_dr9": 4
}
Output: {
  "selected_command": "AT+MOD=LORA-SF12"
}

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "selected_command": "AT+MOD=..."
}
[/INST]
{
"""

def final_decision_prompt_COT():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple en raisonnant sur les conditions précédentes
#Sortie en JSON
    return """[INST] You are an embedded NTN LoRa-Satellite modem controller.
Select the optimal modulation command based on real-time ELEVATION and PREDICTED_PER_JSON values.

INPUT DATA:
- ELEVATION: {ELEVATION} degrees
- PREDICTED_PER_JSON: {PREDICTED_PER_JSON}

STRICT COMMAND DICTIONARY:
- per_sf7  -> AT+MOD=LORA-SF7
- per_sf8  -> AT+MOD=LORA-SF8
- per_sf9  -> AT+MOD=LORA-SF9
- per_sf10 -> AT+MOD=LORA-SF10
- per_sf11 -> AT+MOD=LORA-SF11
- per_sf12 -> AT+MOD=LORA-SF12
- per_dr8  -> AT+MOD=LR-FHSS-DR8
- per_dr9  -> AT+MOD=LR-FHSS-DR9

DECISION LOGIC RULES:
1. HARD RULE (ELEVATION CHECK): If ELEVATION < 15, immediately select "AT+MOD=LR-FHSS-DR8" regardless of the JSON data.
2. NOMINAL RULE: If ELEVATION >= 15, read PREDICTED_PER_JSON, find the lowest integer value (minimum PER), and select its corresponding AT command from the dictionary. If there is a tie, select the faster modulation.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown code blocks, no preamble, no explanations.
Follow this exact structure:
{
  "thought": "1-sentence evaluation: check elevation first, then select lowest PER command",
  "selected_command": "AT+MOD=..."
}
[/INST]
{
"""