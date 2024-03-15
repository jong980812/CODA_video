import os
import subprocess

def extract_center_frame(video_path, output_path):
    # 비디오 총 프레임 수를 구함
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path]
    total_frames = int(subprocess.check_output(cmd).decode('utf-8'))
    
    # 중앙 프레임 계산
    center_frame = total_frames // 2
    
    # 중앙 프레임을 이미지로 추출
    cmd = ['ffmpeg', '-i', video_path, '-vf', f'select=gte(n\,{center_frame})', '-vframes', '1', output_path, '-loglevel','error']
    subprocess.run(cmd)

def process_videos(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mkv')):
                video_path = os.path.join(root, file)
                output_folder = root.replace('kinetics400_320p', 'kinetics400_320p_centerframes')
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.jpg')
                extract_center_frame(video_path, output_path)

# 비디오 폴더 경로 설정
videos_root = '/local_datasets/kinetics400_320p'
# 이미지 폴더 생성 및 처리 시작
process_videos(videos_root)
