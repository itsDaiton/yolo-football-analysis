from utils import *
from trackers import Tracker

def main():
    # Read the video
    video_frames = read_video("data/sample_data.mp4")
    
    # Initialize the tracker
    tracker = Tracker('models/best.pt')
    
    # Get the object tracks
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/tracks_stubs.pkl')
    
    # Draw the object tracks on the video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save the video
    save_video(output_video_frames, "out/sample_data_output.mp4")
    
if __name__ == "__main__":
    main()