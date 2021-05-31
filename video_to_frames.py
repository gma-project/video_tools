from __future__ import print_function, division
import argparse
import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects
from timeit import default_timer as timer

def video_file_to_frames(video_file, out_dir, fps):
	if fps != -1:
		cmd = f'ffmpeg -i \"{video_file}\" -vf \"fps={fps}\" \"{out_dir}/image_%05d.jpg\"'
	else:
		cmd = f'ffmpeg -i \"{video_file}\" \"{out_dir}/image_%05d.jpg\"'
	print(cmd)
	subprocess.call(cmd, shell=True)
	

@wrap_non_picklable_objects
def process_video(i, video_file, out_dir, max_frame):
	file_name = str(Path(video_file).stem)
	image_dir = str(Path(out_dir, file_name))
	if Path(image_dir).exists():
		subprocess.call('rm -r \"{}\"'.format(image_dir), shell=True)
		print('remove {}'.format(image_dir))
	Path(image_dir).mkdir(parents=True, exist_ok=True)
	
	video_file_to_frames(video_file, image_dir, max_frame)
	print(f'Completed!! {i:03} {video_file}\n')


def generate_frames(video_dir, out_dir, fps, file_pattern):
	video_file_list = [str(e) for e in Path(video_dir).glob(file_pattern)]
	video_file_list = sorted(video_file_list)
	
	start = timer()
	
	for i, video_file in enumerate(video_file_list):
		process_video(i, video_file, out_dir, fps)
		
	"""
	set_loky_pickler('dill')
	Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_video)(i, video_file, out_dir, fps)
	                                        for i, video_file in enumerate(video_file_list))
	"""
	
	elapsed = timer() - start
	print(f"Total elapsed time: {elapsed:.1f}.")


def main():
	parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
	parser.add_argument('--video-dir', type=str, default="/home/nbinh/datasets/Supreme_Work/video_data",
	                    help='Path to the videos.')
	parser.add_argument('--out-dir', type=str, default="/home/nbinh/datasets/Supreme_Work/image_data",
	                    help='Path to the image data.')
	parser.add_argument('--file-pattern', type=str, default="*vid*", help='Number of jobs for parallel processing.')
	parser.add_argument('--fps', type=int, default=-1, help="Target frame rate.")
	args = parser.parse_args()
	
	# if Path(image_path).exists():
	# 	shutil.rmtree(image_path)
	# Path(image_path).mkdir(parents=True, exist_ok=True)
	
	print("===================================")
	print("python generate_frames_from_videos.py")
	print("video_dir:", args.video_dir)
	print("out_dir:  ", args.out_dir)
	
	generate_frames(args.video_dir, args.out_dir, fps=args.fps, file_pattern=args.file_pattern)

if __name__ == "__main__":
	main()
