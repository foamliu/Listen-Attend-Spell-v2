from config import *

if __name__ == "__main__":
    for mode in ['train', 'dev', 'test']:
        folder = os.path.join(wav_folder, mode)
        dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        num_files = 0
        for dir in dirs:
            files = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.wav')]
            num_files += len(files)

        print('mode: {}, num_files: {}'.format(mode, num_files))
