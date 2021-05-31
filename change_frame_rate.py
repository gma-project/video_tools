from __future__ import print_function, division
import argparse
import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects
from timeit import default_timer as timer

@wrap_non_picklable_objects
def process_video(i, video_file, out_dir, fps):
	file_name = str(Path(video_file).stem)
	out_file = str(Path(out_dir, f'{file_name}.mp4'))
	
	if Path(out_file).exists():
		subprocess.call('rm -r \"{}\"'.format(out_file), shell=True)
		print('remove {}'.format(out_file))
	
	cmd = f'ffmpeg -i \"{video_file}\" -filter:v fps={fps} \"{out_file}\"'
	print(cmd)
	subprocess.call(cmd, shell=True)
	
	print(f'----------------------------------')
	print(f'Completed!! {i:03} src: {video_file}')
	print(f'                   dst: {out_file}')


def process(video_dir, out_dir, fps):
	video_file_list = [str(e) for e in Path(video_dir).rglob("*.mp4")]
	video_file_list = sorted(video_file_list)
	
	start = timer()
	
	for i, video_file in enumerate(video_file_list):
		process_video(video_file, out_dir, fps)
		
	"""
	set_loky_pickler('dill')
	Parallel(n_jobs=1, backend='loky')(delayed(process_video)(i, video_file, out_dir, fps)
	                                   for i, video_file in enumerate(video_file_list))
	"""
	
	elapsed = timer() - start
	print(f"Total elapsed time: {elapsed:.1f}.")


def main():
	parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
	parser.add_argument('--in-dir', type=str, help='Path to the input videos.')
	parser.add_argument('--out-dir', type=str, help='Path to the output videos.')
	parser.add_argument('--fps', type=int, default=-1, help="Target frame rate.")
	args = parser.parse_args()
	
	# if Path(image_path).exists():
	# 	shutil.rmtree(image_path)
	# Path(image_path).mkdir(parents=True, exist_ok=True)
	
	print("===================================")
	print("python generate_frames_from_videos.py")
	print("in_dir:", args.in_dir)
	print("out_dir:  ", args.out_dir)
	Path(args.out_dir).mkdir(parents=True, exist_ok=True)
	
	process(args.in_dir, args.out_dir, fps=args.fps)


if __name__ == "__main__":
	main()
