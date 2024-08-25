import sys
sys.path.append('../')
from utils import get_bbox_center, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        # Maximum distance between a player and a ball to assign the ball to the player
        self.max_player_ball_distance = 70
        
    def assign_ball_to_player(self, players, bbox):
        # Get the center of the ball bounding box
        ball_pos = get_bbox_center(bbox)
        
        # Set the initial minimum distance to infinity
        min_distance = float('inf')
        # Set the initial assigned player to -1
        assigned_player = -1
        
        # Iterate over the players and calculate the distance between the player and the ball
        for player_id, player in players.items():
            player_bbox = player['bbox']
            distace_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_pos)
            distace_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_pos)
            distance = min(distace_left, distace_right)
            
            # Find the player closest to the ball
            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        
        # Return the player that has ball closest to them           
        return assigned_player
            