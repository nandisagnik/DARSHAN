from temporal_segments import segment_video

segs = segment_video("video.mp4")

print("Segments created:")
for s in segs:
    print(s)
