# Installation and Requirements

## Running as Python Script
To run the script directly, ensure you have the following Python packages installed:

`numpy`
`matplotlib`
`sounddevice`
`scipy`
`tkinter (usually included with Python)`
`ttkbootstrap`
`asyncio`

Install the required packages using pip:
  ```
pip3 install numpy matplotlib sounddevice scipy ttkbootstrap
  ```

# User Interface Example

![Audio Measurement Tool Screenshot](Audio%20Measurement%20Tool%20Screenshot.png)

# Basic Usage
The tool provides a graphical interface with the following measurement functionalities:

1. Select Input & Output Device to select Main Input/Output Channels

2. Select the 'Crosstalk Input Channels' and click 'Measure Crosstalk' to analyze the crosstalk received at the Main Input Channel from the selected sources via the Main Output Channel.

3. Click the 'Measure Input THD+N' button to measure THD+N at approximately –10 dBFS, or use the 'Measure THD+N (–20 dBFS)' button to measure it at –20 dBFS. The results can be saved using the 'Save THD CSV' button.

4. Click the 'Measure Crosstalk' button to analyze the amount of crosstalk received at the Main Input Channel from the selected Crosstalk Input Channels. The results can be saved using the 'Save Crosstalk CSV' button.

5. Click the 'Measure Freq Response' button to perform a frequency response measurement from 20 Hz up to the Nyquist frequency. The results can be saved in .txt format using the 'Save Freq Response' button.

- If the measurement was performed using the 'Measure Input THD+N' button, the saved frequency response corresponds to a –10 dBFS test level.
- If the measurement was performed using the 'Measure THD+N (–20 dBFS)' button, the saved file contains the frequency response measured at 1,250 Hz.

6. Click the 'Save Graph' button to save the Frequency Response graph. This is only available after generating the graph by clicking the 'Graph' tab.

7. Click the 'Save THD Graph' button to save the THD graph. This graph appears in the 'THD Graph' tab and displays both THD and THD+N values as percentages.

8. Use the 'Save Preset' button to store the current device and routing configuration as a preset. Click 'Load Preset' to reload a previously saved configuration.

9. Click the 'Copy Log' button to copy the entire measurement log, including any messages displayed during the measurement process.

## Measuring
1. Measure Input THD+N (THD+N Ratio vs Input Level)
- Description: Measures Total Harmonic Distortion plus Noise (THD+N) for the input signal, following the AES17-2020 standard.
- Procedure: Select the input device, set the desired frequency, and initiate the measurement. Results are displayed in a graph.
- Notes: Ensure proper input levels to avoid clipping. THD+N calculations apply a 12th-order Butterworth low-pass filter with a 22 kHz cutoff and a default 10 Hz high-pass effect via bandpass design, as per AES17-2020 section 5.2.5.

2. Measure THD+N (-20 dBFS) (THD+N Ratio vs Frequency)
- Description: Measures THD+N at a -20 dBFS input level across a range of frequencies, adhering to the AES17-2020 standard.
- Procedure: Configure the frequency range and start the measurement. The frequency response graph will be generated.
- Notes: Ideal for evaluating distortion characteristics at reduced signal levels. A 12th-order Butterworth low-pass filter with a 22 kHz cutoff is applied, with a notch filter (Q = default 2, adjestable) removing the fundamental frequency per section 5.2.8.

3. Measure Crosstalk
- Description: Assesses channel crosstalk between audio channels, based on the AES17-2020 standard.
- Procedure: Select multiple input channels and run the test. Results show the level of signal leakage between channels.
- Notes: Requires a multi-channel audio interface for accurate measurement. A second-order Butterworth band-pass filter centered at the test frequency (997 Hz) is used, conforming to IEC 61260-1 class 1 or 2.

4. Measure Freq Response
- Description: Plots the frequency response of the audio input across a specified range.
- Procedure: Set the input level and frequency range, then generate the graph via the Graph tab.
- Notes: Graph saving is only available after generating the graph by clicking the Graph tab. No explicit low-pass or high-pass filters are applied; the response is derived from a logarithmic sweep signal processed with a Hann window and FFT, with the upper limit naturally constrained by the Nyquist frequency.

## Important Note
- Graphs can only be saved after generating them by clicking the Graph tab.
- Ensure you generate the graph first before attempting to save.
- This code has been verified to run on macOS Sequoia with Python 3.13.1. Compatibility with earlier versions or Windows systems is not guaranteed.

# Help Keep This Person Alive

Your donation helps cover hosting and keeps the content (blog posts & projects) coming.  
Thank you for your support!

<a href="https://paypal.me/JooyoungMusic/1.99" target="_blank">
  <img src="https://img.shields.io/badge/Support-Now-blue?style=for-the-badge&logo=paypal" alt="Support Now via PayPal" />
</a>

# License
Custom MIT License - Non-Commercial Use Only
Copyright (c) 2025 Jooyoung Kim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, provided that the following conditions are met:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
- The Software may not be used for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Additional Usage Notes
This software was developed as part of a research project for an academic paper.
If this tool is used for academic purposes, particularly after the paper's publication, please cite the following work:

- Author: Jooyoung Kim
- Title: [Not Published Yet]
- Journal/Conference: [Not Published Yet]
- Year: [Not Published Yet]
- DOI: [Not Published Yet]
- Citation is required to acknowledge the original research contribution.
- For the citation details, please refer to the published paper once available.

# Non-Commercial Restriction
This license explicitly prohibits any commercial use of the Software.
Use is restricted to non-commercial, educational, or personal purposes only.
Any commercial exploitation, including but not limited to selling the Software or incorporating it into a commercial product, is strictly forbidden without explicit written permission from the copyright holder.
