"""Contient les trois prompts spécialisés de l'agent."""
import json
#########################
#Document en anglais svp#
#########################

def geometry_analysis_prompt_zeroshot():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power)  
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt
#Sortie en JSON

def geometry_analysis_prompt_fewshot():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power) + 2-3 exemples concrets injectés
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt + 2-3 exemples concrets injectés
#ex : "élévation 12°, Doppler 18kHz → sévérité 5"
#Sortie en JSON

def geometry_analysis_prompt_COT():
#Etape 1 : prompt qui analysera la géométrie LEO (EL, Doppler, Rx_power)  
#Etape 2 : classer la sévérité du canal selon une échelle 1-5 via un second paragraphe du prompt + instruction "raisonne étape par étape avant de conclure"
#Sortie en JSON

def per_prediction_via_embedding_prompt_zeroshot():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation
#Sortie en JSON

def per_prediction_via_embedding_prompt_fewshot():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation avec qlq exemples de résultats
#ex : SNR et Linkbudget de exemple 1
#Sortie en JSON

def per_prediction_via_embedding_prompt_COT():
#prompt qui demande l'algorithme knn via les embeddings pour évaluer le PER de chaque modulation avec raisonnement contextualisé
#ex : prompt + instructions de raisonnement sur les données chromaDB
#Sortie en JSON

def final_decision_prompt_zeroshot():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple
#Sortie en JSON

def final_decision_prompt_fewshot():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple + Exemple de choix de modulation selon le cas ex1 et ex2
#Sortie en JSON

def final_decision_prompt_COT():
#prompt qui génère la décision de choix de modulation, et indique les cas de basculement si l'élevation ou la vitesse orbitale change par exemple en raisonnant sur les conditions précédentes
#Sortie en JSON