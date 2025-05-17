import datasets
import os
import csv
import copy

mnt_path = os.getenv('SYNDATA_PATH')
DATA_NAME = '<LANG>/<TEXT_NAME>'
ROOT_DIR = f'{mnt_path}/syndata/'

class SynDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="A dataset of synthetic audio and transcript pairs",
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16000),
                    "transcript": datasets.Value("string"),
                    "language": datasets.Value("string"),
                }
            ),
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "archive_iter": [
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_0.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_1.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_2.tar")), 
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_3.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_4.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_5.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_6.tar")),
                        dl_manager.iter_archive(os.path.join(ROOT_DIR, f"{DATA_NAME}/proc_7.tar")),
                    ],
                    "text_path":[
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_0.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_1.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_2.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_3.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_4.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_5.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_6.csv"),
                        os.path.join(ROOT_DIR, f"{DATA_NAME}/metadata_7.csv"),
                    ],
                },
            ),
        ]
    
    def _get_data(self, lines):
        data = {}
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            
            (
                file_name,
                transcript,
                language,
            ) = line[0], line[1], line[2]

            file_name = file_name.split("/")[1]
            data[file_name] = {
                'transcript': transcript,
                'lang': language,
                }
        return data
    
    def _generate_examples(self, archive_iter, text_path):
        key = 0
        if isinstance(text_path, list):
            text_path = text_path[0]
        if isinstance(archive_iter, list):
            archive_iter = archive_iter[0]
        
        with open(text_path, "r", encoding="utf-8") as f:
            lines = []
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                lines.append(row)
            data = self._get_data(lines)

        for audio_path, audio_file in archive_iter:
            audio_filename = audio_path.split("/")[-1]
            if audio_filename not in data.keys():
                continue
            result = copy.deepcopy(data[audio_filename])
            result['audio'] = {
                'path': audio_path,
                'bytes': audio_file.read(),
            }
            yield key, result
            key += 1
            