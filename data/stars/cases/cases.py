import itertools
import json
import time
from pathlib import Path


class FileGenerator:

    names = (
        'case1',
        'case2',
        'case3',
        'case4',
    )
    img_types = (
        'gray',
        'rgb',
    )
    detectors = (
        'orb',
        'brisk',
        'fast',
        'star',
        'harris',
        'shi_tomasi',
    )
    descriptors = (
        'brief',
        'brisk',
        'orb',
    )
    matchers = (
        'brute_force',
        'flann',
    )

    # Parameters
    video_files = (
        'THE-HAGUE-Den-Haag-Drone-Aerial-4K-|-Scheveningen-Zuid-Holland-Nederland-Netherlands.mp4',  # all cases
    )
    ref_imgs = (
        'den-haag-crossroads-3d-s.png',  # case 1 and 2
        'kurhaus-1-cut.jpg',  # case 3 and 4
    )
    start_times = (
        '0:0:25',  # case 1
        '0:8:12',  # case 2
        '0:5:30',  # case 3
        '0:9:13',  # case 4
    )
    end_times = (
        '0:0:34',  # case 1
        '0:8:45',  # case 2
        '0:6:0',  # case 3
        '0:9:44',  # case 4
    )
    codecs = (
        'mp4v',  # codec is the same for all cases
    )

    def generate_content(self, video_out_files):
        all_cases = {
            'video_file': self.video_files[0],
            'codec': self.codecs[0],
        }
        case1case2 = all_cases | {
            'ref_img': self.ref_imgs[0],
        }
        case3case4 = all_cases | {
            'ref_img': self.ref_imgs[1],
        }
        case1 = case1case2 | {
            'start_time': self.start_times[0],
            'end_time': self.end_times[0],
        }
        case2 = case1case2 | {
            'start_time': self.start_times[1],
            'end_time': self.end_times[1],
        }
        case3 = case3case4 | {
            'start_time': self.start_times[2],
            'end_time': self.end_times[2],
        }
        case4 = case3case4 | {
            'start_time': self.start_times[3],
            'end_time': self.end_times[3],
        }
        keys = [
            'img_type',
            'detector',
            'descriptor',
            'matcher',
        ]
        vals = list(itertools.product(
            self.img_types,
            self.detectors,
            self.descriptors,
            self.matchers,
        ))
        content = [dict(zip(keys, val)) for val in vals]
        cases = []
        for case in (case1, case2, case3, case4):
            cs = []
            for c in content:
                case.update(c)
                cs.append(case)

            cases.append(cs)

        cases = cases[0] + cases[1] + cases[1] + cases[2]
        for i in range(len(cases)):
            print(cases[i])
            cases[i].update({'video_out_file': f'{video_out_files[i]}.mp4'})

        return cases

    def generate_filenames(self):
        names = list(itertools.product(
            self.names,
            self.img_types,
            self.detectors,
            self.descriptors,
            self.matchers,
        ))
        filenames = []
        for n in names:
            filenames.append('_'.join(n))

        return filenames


if __name__ == '__main__':
    fgen = FileGenerator()
    filenames = fgen.generate_filenames()
    content = fgen.generate_content(filenames)
    print(len(filenames))
    print(len(content))
    for c in content:
        filename = Path(c.get('video_out_file')).with_suffix('.json')
        #with open(, 'w') as f:
        #    f.write(json.dump(c))
