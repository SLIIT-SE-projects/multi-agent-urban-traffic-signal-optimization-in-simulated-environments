from src.config import TrainConfig

def calculate_reward(snapshot):

    total_queue = 0
    total_wait = 0
    
    # Aggregate data from all lanes
    for lane_id, info in snapshot['lanes'].items():
        total_queue += info['queue_length']
        total_wait += info['waiting_time'] 

    # Weights
    w_queue = TrainConfig.W_QUEUE
    w_wait = TrainConfig.W_WAIT
    
    penalty = (w_queue * total_queue) + (w_wait * total_wait)
    
    # return negative penalty as reward
    return -penalty