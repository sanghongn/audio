# audio
PROJECT DESCRIPTION (INSTRUMENTS CLASSIFICATION): the data set was given. 


Full project is demonstrate on Colab, you can see detail in 'main.ipynb'


After running 'main.ipynb', the result will be 'prediction.csv' (I converted it as 'prediction.xlsx' provided in this folder).

The predicted audio files was labeled before as in column 'label'

<img width="83" alt="Screenshot 2023-01-14 at 10 59 20" src="https://user-images.githubusercontent.com/107643269/212449726-4b121a90-9f13-4e9c-8efc-a67f85ea5173.png">

an audio's probability can vary  in all trained classes. the highest probability decides which class the audio should be. 

<img width="828" alt="Screenshot 2023-01-14 at 11 03 01" src="https://user-images.githubusercontent.com/107643269/212449839-bc8a0a69-7931-484d-bd9f-9f436fff3b98.png">


filter.py is to clean the audio files (if there are silent intervals of time). 

Cleaned audio files are saved to 'train'. 

main.py is to build model base on 'train' and predict data on 'test'.





