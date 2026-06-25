import json
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class CandidateRankingEngine:
    def __init__(self, embedding_model='all-mpnet-base-v2'):
        # Initialize embedding model (Semantic Understanding - Module 2)
        self.encoder = SentenceTransformer(embedding_model)
        self.vector_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index (Semantic Retrieval - Module 5)
        # Using IndexFlatIP for Cosine Similarity (assuming normalized vectors)
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.candidates_db = []
        
        # Initialize LTR Model (Learning-to-Rank - Module 6)
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            importance_type="gain"
        )
        self.is_ranker_trained = False

    def normalize_vector(self, vec):
        # FIXED: Explicitly cast to float32 to satisfy FAISS strict typing requirements
        return (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)

    def ingest_candidates(self, candidates_structured_data):
        """Embeds and indexes candidates into FAISS."""
        self.candidates_db.extend(candidates_structured_data)
        
        texts_to_embed = [
            f"{c['current_role']} {c['industry']} {', '.join(c['skills'])} {c['experience_summary']}" 
            for c in candidates_structured_data
        ]
        
        embeddings = self.encoder.encode(texts_to_embed)
        normalized_embeddings = self.normalize_vector(embeddings)
        self.index.add(normalized_embeddings)
        print(f"Indexed {len(candidates_structured_data)} candidates into FAISS.")

    def retrieve_top_k(self, job_description_text, k=500):
        """Stage 1: Fast Vector Retrieval."""
        job_emb = self.encoder.encode([job_description_text])
        job_emb = self.normalize_vector(job_emb)
        
        # Search FAISS
        distances, indices = self.index.search(job_emb, min(k, len(self.candidates_db)))
        
        top_candidates = []
        for rank, idx in enumerate(indices[0]):
            candidate = self.candidates_db[idx].copy()
            candidate['semantic_score'] = float(distances[0][rank])
            top_candidates.append(candidate)
            
        return top_candidates, job_emb

    def extract_features(self, job_reqs, candidate, job_emb):
        """Generates numerical features for the LTR model (Module 4)."""
        # Mock feature extraction logic
        skill_overlap = len(set(candidate['skills']).intersection(set(job_reqs['skills']))) / max(1, len(job_reqs['skills']))
        exp_delta = abs(candidate['years_experience'] - job_reqs['target_experience'])
        
        features = {
            'semantic_score': candidate['semantic_score'],
            'skill_overlap_ratio': skill_overlap,
            'experience_penalty': exp_delta,
            'career_stability_score': candidate['avg_tenure_years'],
            'behavior_engagement': candidate['recent_activity_score']
        }
        return features

    def rank_candidates(self, job_reqs, job_description_text, top_k=50):
        """Stage 2: Deep Reranking with LightGBM."""
        # 1. Retrieve
        candidates, job_emb = self.retrieve_top_k(job_description_text, k=top_k)
        
        # 2. Extract Features
        feature_list = []
        for c in candidates:
            features = self.extract_features(job_reqs, c, job_emb)
            feature_list.append(features)
            
        X = pd.DataFrame(feature_list)
        
        # 3. Score (Module 10: Scoring Strategy)
        if self.is_ranker_trained:
            scores = self.ranker.predict(X)
        else:
            # Fallback heuristic if untrained (Semantic 45%, Skills 20%, etc.)
            scores = (X['semantic_score'] * 0.45 + 
                      X['skill_overlap_ratio'] * 0.20 + 
                      (1 / (1 + X['experience_penalty'])) * 0.15 + 
                      X['behavior_engagement'] * 0.10 + 
                      (X['career_stability_score'] / 10) * 0.10)
            
        # 4. Attach scores and sort
        for i, c in enumerate(candidates):
            # FIXED: Handle Pandas Series warnings safely
            c['final_score'] = float(scores.iloc[i]) if isinstance(scores, pd.Series) else float(scores[i])
            
        ranked_candidates = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        return ranked_candidates

    def generate_explanation(self, candidate, job_reqs):
        """Generates Explainable AI insights (Module 7 & 13)."""
        missing_skills = list(set(job_reqs['skills']) - set(candidate['skills']))
        
        explanation = {
            "Candidate_ID": candidate['id'],
            "Overall_Score": round(candidate['final_score'] * 100, 1),
            "Breakdown": {
                "Semantic_Match": round(candidate['semantic_score'] * 100, 1),
                "Experience_Match": f"{candidate['years_experience']} yrs vs {job_reqs['target_experience']} target",
            },
            "Strengths": [f"High semantic alignment", f"Strong stability ({candidate['avg_tenure_years']} yr avg tenure)"],
            "Missing_Skills": missing_skills,
            "Recommendation": "Highly Recommended" if candidate['final_score'] > 0.8 else "Evaluate with caution"
        }
        return json.dumps(explanation, indent=2)

# --- Execution Example ---
if __name__ == "__main__":
    engine = CandidateRankingEngine()
    
    # Mock Database
    db = [
        {"id": "C1", "current_role": "Backend Engineer", "industry": "SaaS", "skills": ["Python", "AWS", "Docker"], "experience_summary": "Built scalable APIs", "years_experience": 5, "avg_tenure_years": 2.5, "recent_activity_score": 0.9},
        {"id": "C2", "current_role": "Data Scientist", "industry": "Finance", "skills": ["Python", "PyTorch", "SQL"], "experience_summary": "Trained ML models", "years_experience": 4, "avg_tenure_years": 1.2, "recent_activity_score": 0.4},
        {"id": "C3", "current_role": "Platform Engineer", "industry": "Tech", "skills": ["Go", "Kubernetes", "AWS", "Terraform"], "experience_summary": "Designed distributed cloud infrastructure", "years_experience": 7, "avg_tenure_years": 3.5, "recent_activity_score": 0.8}
    ]
    
    engine.ingest_candidates(db)
    
    # Mock Job Post
    job_desc = "Senior Backend Engineer. Must have experience designing scalable distributed systems in AWS. Knowledge of containerization (Docker/Kubernetes) is highly preferred."
    job_requirements = {
        "skills": ["AWS", "Docker", "Kubernetes", "Python"],
        "target_experience": 6
    }
    
    # Run Engine
    results = engine.rank_candidates(job_requirements, job_desc)
    
    # Output Explainability for Top Candidate
    print("\n--- Final Output: Explainable Candidate Ranking ---")
    print(engine.generate_explanation(results[0], job_requirements))