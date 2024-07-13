# Convolutional Kernel Density Estimation (CKDE)

## 📄 Description
The Convolutional Kernel Density Estimation (CKDE) is a method that transforms EEG signals into images and applies a convolution filter to improve the visualization and automatic detection of pathological events such as interictal epileptic spikes (IESs). More details can be found in published scientific paper [[L. Gardy et al., 2019](https://doi.org/10.5220/0008877601010109)] and in [[doctoral thesis of L. Gardy](http://thesesups.ups-tlse.fr/5164/1/2021TOU30190.pdf)].

With a time series as input, the algorithm performs the following steps:
- Transforms the timeseries into an image
- Convolves this image
- [Optional] Filters the convolved image to isolate epileptic spikes

User can then apply filters, like a low-pass filter, to isolate low density events, such as IEDs.

## ⚒️ Installation

### Prerequisites
- Python 3.11
- Python libraries
    ```sh
    pip install -r requirements.txt
    ```

## 📝 Usage
```python
### Get a timeseries filepath (look in the folder you have downloaded)
timeseries_folderpath = r"input_data\events_signal_data"
timeserie_filename = f"event_{event_num}.txt"

### Load a timeseries from the sample data provided with this program (1D)
signal = load_timeseries(timeseries_folderpath, timeserie_filename) # or,
#signal = random_signal_simulation()

### Get the timeseries info
meta_data = json.load(open(r"input_data\events_info.json"))
sfreq = meta_data["events_info"][event_num]["sampling_frequency"]

### Convert it to a 2D signal
image_2D = from_1D_to_2D(signal, bandwidth = 1)

### Convolve the 2D signal
image_2D_convolved = convolve_2D_image(image_2D, convolution = "gaussian custom")

### Plot result
fig_name = "Epileptic spike (signal duration: 400 ms) \n\n[1] raw [2] imaged [3] convoluted"
plot_result(signal, image_2D, image_2D_convolved, fig_name)
```

```sh
python main.py  # Runs the script
```

### Input data
We propose some simulated data to validate our procedure with a known frequency, duration and position. This database is structured as shown in figure 1. User can either use these data, use his own, or simulate some. A signal simulation function is also provided in the program.

![](images/image1.jpg)

### Methods
Figure 2 shows how the convolved image (2D) is drawn from the raw signal (1D). A: Convolution process. B: Full process.

![](images/image2.png)

### Results
Figure 3 shows the result of the full process. The timeseries used as input is an IED called "event_5" in the data sample we provide with this program.

![](images/image3.png)

## 📚 References
Gardy, L., Barbeau, E., and Hurter, C. (2020). Automatic detection of epileptic spikes in intracerebral eeg with convolutional kernel density estimation. In 4th International Conference on Human Computer Interaction Theory and Applications, pages 101–109. SCITEPRESS-Science and Technology Publications. https://doi.org/10.5220/0008877601010109

## 👤 Author
- LinkedIn: [Ludovic Gardy](https://www.linkedin.com/in/ludovic-gardy/)
- Doctoral thesis: [PDF](http://thesesups.ups-tlse.fr/5164/1/2021TOU30190.pdf)