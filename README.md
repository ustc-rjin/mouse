Code for paper CAPTCHA Farm Detection and User Authentication via Mouse-trajectory Similarity Measurement  


Identity inconsistency detection --> main_fusion.py  
Baselines --> hydra_multirocket.py, hivecote.py, inceptiontime.py  
You can find the following settings at the start of main_fusion.py:  
PRE: Use the preprocessing  
RF: Use the RF classifier only  
DL:  Use the DL classifier only  
FUSION:  Use both classifiers  


Authentication --> authentication_fusion.py  
Baselines --> Antal.py, Fu.py, Siddiqui.py  
You can find the following settings at the start of authentication_fusion.py:  
TRAINING_SET: Can be 'mix', 'balabit', or 'sapimouse' to simulate different registered user set  
SAMPLE_SELECTION: Use the enrollment sample selection  
DYNAMIC_AUTH: The number of samples used in dynamic authentication  


The used datasets are publicly available in:  
https://www.ms.sapientia.ro/~manyi/sapimouse/sapimouse.html  
https://github.com/balabit/Mouse-Dynamics-Challenge  

