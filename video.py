import yt_dlp 

url = "https://www.youtube.com/watch?v=K6xsEng2PhU "
ydl_opts = {
    'format': 'mp4',
    'outputmpl': 'input.mp4',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])