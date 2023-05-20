# DC4SDP
Supplementary code and data of our paper being submitted to JSS entitled *Adjusted Trust Score: A Novel Approach for Estimating the Trustworthiness of Defect Prediction Models*

Defect prediction techniques are essential for identifying defective code regions and improving testing efficiency. Over the past decades, a plethora of defect prediction approaches has emerged, with machine learning (ML) models being the most widely employed. While these models exhibit improved predictive performance, their inherent black-box nature and associated uncertainties make it challenging for developers to trust their predictions. To mitigate this problem, we propose a novel trustworthiness score, namely adjusted trust score (ATS), to discern the circumstances under which classifier predictions should and should not be trusted. Furthermore, we also employ ATS to design a reject option for defect prediction models. Extensive experiments conducted on 20 benchmark datasets and 5 commonly used ML classifiers demonstrate that high (low) trust scores can produce high precision at identifying correct (incorrect) predictions. Furthermore, the ATS outperforms its counterparts based on the Wilcoxon signed-rank test. In addition, a comparison analysis on prediction performance with and without a reject option confirms the feasibility of designing a reject option for defect prediction models based on our proposed trustworthiness score. Our work highlights that the trustworthiness score can aid developers in better comprehending the strengths and limitations of ML-based defect prediction models. Therefore, it is an essential component for guaranteeing trust from developers and deserves further investigation.

# Acknowledgements
Part of the code references the code from the following code repositories. We are very grateful for the excellent work of the authors of these repositories:  
https://github.com/google/TrustScore  
https://github.com/AFAgarap/dnn-trust  
https://github.com/ai-se/early-defect-prediction-tse  
https://github.com/COLA-Laboratory/icse2020  
