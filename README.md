# Realtime-Object-Detection-and-Tracking
A Modern Computer Vision Model whose main task is to gather infromation from Traffic videos. It counts the number of objects and also the type of objects that crosses a user defined line in the video and gives the output as a CSV.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- [Anaconda](https://www.anaconda.com/products/distribution) (recommended) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)


### Installation

1. **Clone the Repository**:
```sh
git clone https://github.com/rishabh700/Realtime-Object-Detection-and-Tracking.git
cd Realtime-Object-Detection-and-Tracking
```

2. **Create and Activate Conda Environment**: Create the environment using the "environment.yml" file:
```sh
conda env create -f environment.yml
conda activate yolov11
```

### Configuration

1. **Edit ["Traffic_Counting.py"](Traffic_Counting.py)**: Open "Traffic_Counting.py" and set the following variables:
    * `model_path`: Path to the YOLO model.
    * `video_path`: Path to the input video.
    * `out_put_video_path`: Path to the output video.
    * `output_csv`: Path to the output CSV.

Example:
```python
model_path = "yolo11n.pt"
video_path = r"C:\path\to\input\video.mp4"
out_put_video_path = r"C:\path\to\output\video.avi"
output_csv = r"C:\path\to\output\output.csv"
```

2. **Running the Code**: 
Run the "Traffic_Counting.py" script:

```python
python Traffic_Counting.py
```

### Output
The script will process the video and generate:

* An output video with detected objects and tracking information.
* A CSV file with the count and type of objects that crossed the user-defined line.