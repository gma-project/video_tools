import cv2
import argparse
from pathlib import Path

def get_video_info(video_file):
	cap = cv2.VideoCapture(video_file)
	fps = cap.get(cv2.CAP_PROP_FPS)
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	return n_frames, W, H, fps

def main():
	parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
	parser.add_argument('--video-dir', type=str, help='Path to the videos.')
	parser.add_argument('--out-file', type=str)
	args = parser.parse_args()
	
	video_file_list = [str(e) for e in Path(args.video_dir).rglob("*.mp4")]
	video_file_list = sorted(video_file_list)
	
	f = open(args.out_file, 'w')
	f.write('video,n_frames,pixel_x,pixel_y,fps')
	
	for video_file in video_file_list:
		file_name = Path(video_file).stem
		n_frames, W, H, fps = get_info(video_file)
		f.write(f'\n{file_name},{n_frames},{W},{H},{fps}')
		print(f"{file_name} | frames: {n_frames} | size (HxW): {H}x{W} | fps: {fps}")
		
	f.close()
		
	return 0

if __name__ == '__main__':
	main()