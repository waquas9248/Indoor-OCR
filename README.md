# Indoor-OCR

## Intro
Custom OCR for a specific use-case : entity recognition in building environments, referenced by an indoor navigation mobile app.
For local download and inference suited for hardware specs found on most pervasive mobile devices worldwide.

## Need for such indoor navigation application
![image](https://github.com/user-attachments/assets/2aa77910-645c-4df7-924b-53b8d457e7d0) 

(Reference: nfpa.org)

![image](https://github.com/user-attachments/assets/64e8bd12-3ef6-4e49-9a1c-f078f701f848)

(Reference: ncbi.nlm.nih.gov)

## Dataset
TextOCR - https://textvqa.org/textocr/dataset/

## Application architecture
![image](https://github.com/user-attachments/assets/3c473e62-3507-48b7-8e68-ca156fc66509)

## Post-processing steps
NMS for TextDetection, on IoUs of anchor boxes output
CTC for TextRecognition, on list of all time-step ascii char probabilities output
