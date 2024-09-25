
# Side Channel attack on LLMs:  LLMs Topic Inference over Encrypted Traffic by analyzing packets response times and size.


![side_channel0](https://github.com/user-attachments/assets/2b495c56-5881-42ec-80ae-3ed66d356b19)



## Overview

This repository contains the implementation of our research project on side-channel attacks targeting Large Language Models (LLMs). The project demonstrates how an adversary can infer the topic of conversation between a user and an LLM by analyzing the response times and the number of tokens in the responses, even over encrypted channels. This attack poses significant privacy risks and underscores the need for enhanced security measures in LLM deployments.


![side_cannek](https://github.com/user-attachments/assets/d550739a-7c96-4d1d-a593-9ca09bfba946)



## Abstract

With the increasing deployment of LLMs such as ChatGPT, Bard, Falcon, and LLaMa in various applications, ensuring the privacy and security of user interactions with these models is paramount. Our research reveals a novel side-channel attack where an adversary can infer the conversation topic based solely on response times and token counts, without direct access to the content of the communication. The implications of such an attack are profound, highlighting vulnerabilities in existing LLM architectures.

## Methodology

### Data Collection

1. **Dataset Creation**: We gathered a collection of question-answer pairs across various topics (Medical, Code, Sports, etc.) from publicly available datasets.
2. **LLM Interaction**: The datasets were used to prompt open-source LLMs such as LLaMa and Falcon, and the response times along with token counts were recorded.
3. **Data Preparation**: The collected data was processed into tuples of (Topic, Time, #Tokens) to be used for training predictive models.

### Attack Model

The adversary's capabilities are defined as follows:

- **A1**: The adversary can access the response time of the LLM.
- **A2**: The adversary has access to both the response time and the number of tokens generated in the response.


![side_cannel2](https://github.com/user-attachments/assets/21ac46cb-d918-4ea9-829a-efd6e00d7994)




### Predictive Models

- **XGBoost**: A gradient boosting model trained on the processed data to predict conversation topics.
- **Neural Networks**: A neural network model designed to uncover complex patterns between response times, token counts, and conversation topics.

### Experiment Design

- **Hypotheses**: 
  1. A1 can infer the topic using only response times.
  2. A2 can improve the accuracy of topic inference by incorporating token counts.
  3. Sub-topics within the main topic can be inferred by expanding the model's input features.
 

![side_channel3](https://github.com/user-attachments/assets/603ad8c1-b398-4c6c-b5d9-d3b539d8dbb5)



- **Evaluation**: The models were evaluated using Precision, Recall, F1-Score, Mean Squared Error (MSE), confusion matrices, and ROC curves.

## Results

- **A1 Results**: Low accuracy in topic inference with only response times, showing a trend of success in some topics.
- **A2 Results**: High accuracy in topic inference when both response times and token counts are used, with significantly better performance on Falcon compared to LLaMa.
- **A3 Results**: Attempted sub-topic inference showed weak results, indicating that further research is needed.



![side_channel4](https://github.com/user-attachments/assets/fa3a4bba-b16c-4d70-a95b-e0e2efbfc24a)




## Conclusion

Our study demonstrates a novel method of inferring conversation topics from encrypted traffic by analyzing LLM response times and token counts. This side-channel attack highlights a critical vulnerability in current LLM architectures and calls for the development of countermeasures such as randomized response times and token padding.

## Future Work

- Expanding the study to include more diverse LLM architectures and datasets.
- Investigating additional side-channel features to enhance attack accuracy.
- Developing and testing defense mechanisms against such attacks.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LLM-Side-Channel-Attack.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the experiments**:
   Execute the Jupyter notebook to train the models and evaluate the attack effectiveness.

## Authors

- Sagiv Antebi - [GitHub](https://github.com/yourusername)
- Ben Ganon
- Omri Ben Hemo

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
