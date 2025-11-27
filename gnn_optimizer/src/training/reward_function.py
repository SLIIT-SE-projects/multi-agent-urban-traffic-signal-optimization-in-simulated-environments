def calculate_reward(snapshot):

    total_queue = 0
    total_wait = 0
    
    # Aggregate data from all lanes
    for lane_id, info in snapshot['lanes'].items():
        total_queue += info['queue_length']
        total_wait += info['waiting_time'] 

    # Weights
    w_queue = 1.0
    w_wait = 0.01 
    
    penalty = (w_queue * total_queue) + (w_wait * total_wait)
    
    # return negative penalty as reward
    return -penalty