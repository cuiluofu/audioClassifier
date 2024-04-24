import json
import math
import os

import librosa

SAMPLE_RATE = 22050
DATASET_PATH = "E:\\Datasets\\Parkinson_Speech\\26-29_09_2017_KCL\\SpontaneousDialogue"
JSON_PATH = "E:\\Datasets\\Parkinson_Speech\\26-29_09_2017_KCL\\SpontaneousDialogue\\data.json"
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=20):
    """
    Extracts MFCC features from audio files and saves them to a JSON file.
    :param dataset_path(str) :
    :param json_path(str) :
    :param n_mfcc:
    :param n_fft:
    :param hop_length:
    :return:
    """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we are at sub-folder level
        if dirpath is not dataset_path:

            # save genre label(i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing:{}".format((semantic_label)))

            # process all audio files in class sub-dir
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # etract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_fft=n_fft,
                                                hop_length=hop_length,
                                                n_mfcc=n_mfcc)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment：{}".format(file_path, d + 1))

        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
