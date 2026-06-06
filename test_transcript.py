from youtube_transcript_api import YouTubeTranscriptApi

video_id = "cOAaonpTLlc"

try:
    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    print("SUCCESS")
    print("Length:", len(transcript))

    for item in transcript[:5]:
        print(item)

except Exception as e:
    import traceback
    traceback.print_exc()