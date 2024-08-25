from utils import *
from player_ball_assigner import PlayerBallAssigner
from trackers import Tracker
from team_assigner import TeamAssigner
import numpy as np
import os

def main():
    # Read the video
    video_frames = read_video("data/sample_data.mp4")
    
    # Initialize the tracker
    tracker = Tracker('models/best.pt')
    
    # Get the object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/tracks_stubs.pkl')
    
    # Interpolate the missing ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    # Assign team to each player and color based on the team
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, player_track in player_tracks.items():
            player_team = team_assigner.get_player_team(video_frames[frame_num], player_track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = player_team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[player_team]
            
    # Assign ball aquisition to players
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']   
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
            
    # Make sure the output directory exists
    output_dir = 'out'
    os.makedirs(output_dir, exist_ok=True)
         
    # Draw the object tracks on the video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Save the video
    save_video(output_video_frames, os.path.join(output_dir, "sample_data_output.mp4"))
    
if __name__ == "__main__":
    main()