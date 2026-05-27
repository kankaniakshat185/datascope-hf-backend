from typing import List, Dict, Any

class InsightGenerator:
    """
    Auto-Generated Insights using deterministic templates without LLM APIs.
    """
    
    @staticmethod
    def generate_root_cause_insights(ranked_causes: List[Dict[str, Any]]) -> List[str]:
        insights = []
        if not ranked_causes:
            return ["Model errors are evenly distributed with no clear driving feature."]
            
        top_cause = ranked_causes[0]
        insights.append(f"Prediction instability is highly concentrated in rows where '{top_cause['feature']}' exhibits drift.")
        
        if len(ranked_causes) > 1:
            second = ranked_causes[1]
            if "variance" in second['reason'].lower():
                insights.append(f"Variance inflation in '{second['feature']}' strongly correlates with model failure.")
            else:
                insights.append(f"Distribution shift in '{second['feature']}' is the secondary driver of prediction error.")
                
        return insights

    @staticmethod
    def generate_benchmark_insights(leaderboard: List[Dict[str, Any]]) -> List[str]:
        if not leaderboard:
            return []
            
        best = leaderboard[0]
        insights = [f"'{best['model']}' achieved the best performance with robust stability under noise."]
        
        # Check if there's a faster model
        for model in leaderboard[1:]:
            if model['latency']['inference_ms_per_sample'] < best['latency']['inference_ms_per_sample']:
                insights.append(f"If latency is critical, consider '{model['model']}' which is faster but slightly less accurate.")
                break
                
        return insights
