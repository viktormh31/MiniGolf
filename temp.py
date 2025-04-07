
import torch


x = torch.tensor([1,2,3])


y = x.repeat(5,1)
ball_positions = torch.tensor([
    [1.0, 2.0, 3.0],
    [1.02, 2.01, 3.03],
    [0.95, 1.98, 2.97],
    [1.1, 2.1, 3.1],
    [0.9, 1.9, 2.9],
    [1.05, 2.04, 3.02],
    [1.06, 2.06, 3.07],
    [1.00, 2.00, 3.00],
    [1.03, 2.02, 3.01],
    [0.96, 1.99, 2.98]
])

n = torch.tensor([2,2,2])
y[2] = n
print(y,y.shape)
print(x,x.shape)
#print(y[-1])


distance = torch.norm(x - ball_positions, p=2,dim=1)
#reward = torch.tensor((distance < 1)) - 1.
#done = reward + 1

print(distance)

def is_ball_near_hole(ball_positions: torch.Tensor, hole_pos: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Checks if each ball position in a batch of 10 positions is within the threshold distance of the hole.
    
    Args:
    - ball_positions: Tensor of shape (10, 3), representing 10 (x, y, z) positions.
    - hole_pos: Tensor of shape (3,), representing a single (x, y, z) position.
    - threshold: Distance threshold (default: 0.05).
    
    Returns:
    - A boolean tensor of shape (10,), where True means the ball is within the threshold distance.
    """
    distances = torch.norm(ball_positions - hole_pos, p=2, dim=1)  # Compute distances for all 10 positions
    return torch.where(distances <= threshold, torch.tensor(0), torch.tensor(-1))

# Example usage:

hole_position = torch.tensor([1.0, 2.0, 3.0])

result = is_ball_near_hole(ball_positions, hole_position)
print(result) 
