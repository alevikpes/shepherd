# Shepherd

Finds unique objects in images and videos by matching a given image of
the object. It uses OpenCV Feature Matching algorithms.

Can be used as an extra validation method for autonomous drone flights.


### DISCLAIMER

This project uses some licensed mathematical algorithms as an example of
implementation. This is a research project and I am not responsible for
anything.


### Setup

Tested with Python v11, v13, v14. Use `venv` and `pip-tools` for correct
versions matching.

> Check your PYTHONPATH and modify the imports if necessary.


### Cases

`data` directory contains cases and configurations. Easiest is to copy one
of the existing directories and update `config.py` and `.json` files.

Update path and arguments in `main.py` and `video_main.py`.

Run with:
```bash
python src/main.py  # or
python src/video_main.py
```


### Possible advances

* Image enhancements
* SLAM
* Thermal images
