from __future__ import print_function, division
import argparse
import subprocess
from pathlib import Path
from timeit import default_timer as timer
import cv2
from get_video_info import get_video_info
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects
import traceback

def video_file_to_frames(video_file, sub_out_dir, log_dir, start_frame, end_frame):
	video_len, _, _, _ = get_video_info(video_file)
	video_name = Path(video_file).stem
	
	cap = cv2.VideoCapture(video_file)
	
	t = 1
	while t < start_frame:
		_, _ = cap.read()
		t += 1
	# t = start_frame
	while t <= end_frame and cap.isOpened():
		_, frame = cap.read()
		frame_file = str(Path(sub_out_dir, f"img_{t:05}.jpg"))
		try:
			cv2.imwrite(frame_file, frame)
		except Exception as e:
			log_file = str(Path(log_dir, f"{video_name}.txt"))
			with open(log_file, "a") as f:
				f.write(f"ERROR::video: {video_name} | frame: {t}")
				f.write(f"ERROR::{e}")
			print(traceback.print_exc())
			print(f"video: {video_file} | frame: {t}")
		t += 1

@wrap_non_picklable_objects
def process(i, video_file, out_dir, log_dir, start_frame=0, end_frame=2000):
	video_name = str(Path(video_file).stem)
	sub_out_dir = str(Path(out_dir, video_name))
	if Path(sub_out_dir).exists():
		subprocess.call('rm -r \"{}\"'.format(sub_out_dir), shell=True)
		print('remove {}'.format(sub_out_dir))
	Path(sub_out_dir).mkdir(parents=True, exist_ok=True)
	
	video_file_to_frames(video_file, sub_out_dir, log_dir, start_frame, end_frame)
	print(f'Completed!! {i:03} {video_file}\n')


def main():
	parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
	parser.add_argument('--video-dir', type=str, default="/home/nbinh/datasets/Supreme_Work/raw_data/video_data",
	                    help='Path to the videos.')
	parser.add_argument('--out-dir', type=str, default="/home/nbinh/datasets/Supreme_Work/raw_data/image_data",
	                    help='Path to the image data.')
	parser.add_argument('--log-dir', type=str,
	                    default="/home/nbinh/datasets/Supreme_Work/interim_data/logs/video_to_frames",
	                    help='Path to the log dir.')
	parser.add_argument('--file-pattern', type=str, default="*vid*.mp4", help='Number of jobs for parallel processing.')
	parser.add_argument('--start-frame', type=int, default=1)
	parser.add_argument('--end-frame', type=int, default=4000)
	parser.add_argument('--n-jobs', type=int, default=16)
	args = parser.parse_args()
	
	video_dir = args.video_dir
	out_dir = args.out_dir
	start_frame = args.start_frame
	end_frame = args.end_frame
	log_dir = args.log_dir
	file_pattern = args.file_pattern
	
	print("===================================")
	print("python generate_frames_from_videos.py")
	print("video_dir:", args.video_dir)
	print("out_dir:  ", args.out_dir)
	
	Path(log_dir).mkdir(parents=True, exist_ok=True)
	
	video_file_list = [str(e) for e in Path(video_dir).glob(file_pattern)]
	video_file_list = sorted(video_file_list)
	
	start = timer()

	set_loky_pickler('dill')
	Parallel(n_jobs=args.n_jobs, backend='loky')(delayed(process)(i, video_file, out_dir, log_dir,
	                                                              start_frame, end_frame)
	                                             for i, video_file in enumerate(video_file_list))
	
	elapsed_time = timer() - start
	print(f"Total elapsed time: {elapsed_time:.1f}.")

if __name__ == "__main__":
	main()
