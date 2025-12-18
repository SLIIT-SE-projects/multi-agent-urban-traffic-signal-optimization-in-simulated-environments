from src.config import TrainConfig

def calculate_reward(snapshot):

    total_penalty = 0
    
    for lane_id, data in snapshot['lanes'].items():
        queue = data['queue_length']
        wait = data['waiting_time'] 
        
        # 1. Quadratic Queue Penalty
        q_penalty = (queue ** 2) / 10.0
        
        # 2. Waiting Time Penalty
        w_penalty = 0
        if wait > 120:
            w_penalty = wait * 5.0
        else:
            w_penalty = wait * 0.2

        total_penalty += (q_penalty + w_penalty)
    
    return -total_penalty / 500.0