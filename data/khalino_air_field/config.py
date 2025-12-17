import json
from datetime import datetime
from pathlib import Path


class Case:

    root_path = Path(__file__).parent
    cases_path = root_path / 'cases'
    input_path = root_path / 'input'
    output_path = root_path / 'output'
    enhanced_path = root_path / 'enhanced'
    single_input_path = input_path / 'single'
    single_output_path = output_path / 'single'

    ref_img = None
    colour_scheme = None
    test_video = None
    test_image = None
    detection = None

    def __init__(self, case_name: str):
        self.case_name = case_name
        self._load()

    def _load(self):
        """Load a case by cpecifying a name of the desired json file
        in the config directory.
        """
        casename = self.case_name.split('.')[0]
        with open(self.cases_path / f'{casename}.json') as f:
            case = json.load(f)

        # Load image type.
        self.colour_scheme = case.get('colour_scheme')
        # Load ref image
        self.ref_img = str(self.input_path / case['ref_img'])
        # Load test video params.
        self.test_video = case.get('test_video')
        #self.video_file = str(self.input_path / case['video_file'])
        ## load video start and end times in milliseconds
        #st = datetime.strptime(case['start_time'], '%H:%M:%S').time()
        #et = datetime.strptime(case['end_time'], '%H:%M:%S').time()
        #self.start_time = (st.hour * 3600 + st.minute * 60 + st.second) * 1000
        #self.end_time = (et.hour * 3600 + et.minute * 60 + et.second) * 1000
        ## load video utput file name
        #self.video_out_file = str(self.output_path / case['video_out_file'])
        #self.codec = case.get('codec').lower()
        # Load test image params.
        self.test_image = case.get('test_image')
        # Load detection params.
        self.detection = case.get('detection')
        # Load enhancements.
        self.enhancements = case.get('enhancements')
