import cv2  # OpenCV (not used here directly, but often used for video/frame work)
import os  # for file and folder handling
from moviepy import VideoFileClip  # used to read and cut video + extract audio

SEGMENT_SEC = 2  # each video piece will be 2 seconds long


def segment_video(video_path, out_dir="segments"):
    # Create folder "segments" if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Load full video file
    video = VideoFileClip(video_path)

    # Get total video duration in seconds (integer)
    duration = int(video.duration)

    segments = []  # this will store info about every segment created

    # Loop through video in steps of 2 seconds
    for start in range(0, duration, SEGMENT_SEC):

        # End time = start + 2 sec (but don’t exceed video length)
        end = min(start + SEGMENT_SEC, duration)

        # Create a name like seg_0_2, seg_2_4, etc.
        seg_name = f"seg_{start}_{end}"

        # Path where small video piece will be saved
        seg_video_path = os.path.join(out_dir, seg_name + ".mp4")

        # Path where audio of that piece will be saved
        seg_audio_path = os.path.join(out_dir, seg_name + ".wav")

        # ---- Cut small video from start → end ----
        # This extracts only that 2-second portion
        clip = video.subclipped(start, end)

        # Save the small video segment (without audio)
        clip.write_videofile(seg_video_path, codec="libx264", audio=False, logger=None)


        # ---- Extract audio from that same segment ----
        # If this segment contains audio track
        if clip.audio is not None:

            # Save audio as WAV (uncompressed for better sound analysis)
            clip.audio.write_audiofile(seg_audio_path, codec="pcm_s16le", logger=None)

        else:
            # If no audio in that segment → mark audio as None
            seg_audio_path = None

        # Store information about this segment
        segments.append({
            "start": start,          # segment start time (sec)
            "end": end,              # segment end time (sec)
            "video": seg_video_path, # where video piece is saved
            "audio": seg_audio_path  # where audio piece is saved (or None)
        })

    # Return full list of all created segments
    return segments


'''This function:

Takes a full video

Cuts it into small 2-second pieces

Saves each piece as:

a small video file

its corresponding audio file

Keeps track of when each piece happened (timestamp)

Returns a list describing all pieces'''