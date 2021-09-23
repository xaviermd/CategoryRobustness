# CategoryRobustness
Human-defined categories can sometimes be vague or reveal their author's personal biases. I propose two methods for evaluating the amount of overlap between human-defined categories. The first method uses simple word count statistics, whereas the other employs machine learning algorithms to evaluate differences between sentences.

In the field of Intention Mining, algorithms are created to classify developer communications, app reviews, user feedback, and other such communications into a set of experimenter-defined categories. Many sets of categories have been defined, suggesting disagreement among researchers on the correct set of categories. This in turn suggests that these categories may not be grounded in truth, but rather in experimenters' personal biases.

I evaluate the overlap between categories by looking at which words are shared between categories and which are not. I also evaluate the overlap by augmenting the data, swapping out parts of sentences, and using machine learning algorithms to reclassify the new sentences.
