import re
from verl.utils.reward_score import registry

@registry.register("medical_compliance")
def medical_score_func(solution_str, ground_truth, extra_info=None):
    """
    Custom reward function for medical compliance.
    Args:
        solution_str: The model's generated response.
        ground_truth: The reference label.
    Returns:
        score: float
    """
    score = 0.0
    
    # 1. Format Reward: Check for required XML structure
    if "<diagnosis>" in solution_str and "</diagnosis>" in solution_str:
        score += 0.5
        
    # 2. Content Matching: Check if keywords from ground_truth exist in solution
    if ground_truth and len(ground_truth) > 0:
        # Simple keyword matching example
        keywords = ["check", "suggest", "hospital"] 
        for kw in keywords:
            if kw in solution_str:
                score += 0.1
                
    # 3. Penalty: Penalize extremely short responses
    if len(solution_str) < 10:
        score -= 1.0
        
    return score
