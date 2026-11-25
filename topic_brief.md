# Movie Review Sentiment Classification Using Machine Learning

## Team Members

- Prisha Shah
- Laura Botnaru

## One-Sentence Task

Given a written movie review, we will use machine learning to classify its sentiment as positive or negative.

## Illustrative Input and Desired Output

- **Example input:** "The movie had an engaging storyline, great acting, and an excellent ending. I would definitely recommend it."
- **Desired output:** Positive sentiment.

This is an illustrative example, not a prediction from a trained model.

## Background and Starting Point

We chose this project to explore how machine learning can identify opinions in written text. Reading and categorizing large numbers of reviews manually can be time-consuming, so sentiment analysis offers a way to automate this process. Through this project, we want to learn how to prepare text data, train a classification model, and evaluate its ability to recognize positive and negative opinions.

## Intended Output and Success

Our project will produce a Python workflow that classifies movie reviews as positive or negative. It will include a trained model, an evaluation summary, and examples of correct and incorrect predictions.

Success will mean that our model performs better than a simple baseline that always predicts the most common sentiment class on reviews that were not used for training.

## Preferred Data

### Corpus

We plan to use the IMDb Large Movie Review Dataset, which contains labeled movie reviews for binary sentiment classification. Each labeled review is classified as positive or negative. The unit of analysis is one movie review, and the primary fields are the review text and its sentiment label.

### Source and Access

- **Dataset name:** IMDb Large Movie Review Dataset
- **Original source:** Maas et al. (2011), associated with the Stanford AI Lab.
- **Direct link:** https://huggingface.co/datasets/stanfordnlp/imdb
- **Original dataset information:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Access method:** We plan to load the dataset in Python using the Hugging Face `datasets` library.
- **Approximate size:** 25,000 labeled training reviews and 25,000 labeled testing reviews, plus 50,000 additional unlabeled reviews.
- **Access status:** We have confirmed that the dataset page is accessible, but still need to test loading the data in Python.
- **Usage considerations:** We will review the dataset's licensing and usage conditions before using or redistributing data. Our repository will link to the original source rather than redistribute the full dataset.
- **Backup corpus:** Cornell Movie Review Data: https://www.cs.cornell.edu/people/pabo/movie-review-data/

### Examples

The following examples illustrate the type of data we intend to analyze. They are not copied from the actual dataset.

- **Positive:** "The acting was excellent, and the story kept me interested until the end."
- **Negative:** "The plot was boring, the acting was poor, and the movie was disappointing."

### Fit with the Task

The dataset contains written reviews paired with positive or negative labels, which directly supports our classification task. The labeled reviews can be used to train the model, while separate testing reviews can help us evaluate its performance on unseen text.

## Feasibility and Immediate Next Steps

- **Minimum viable project:** Load the labeled reviews, prepare the text, convert the text into numerical features using TF-IDF, train a Logistic Regression classifier, and evaluate its predictions.
- **Data-access check:** The dataset page is accessible, but loading the data in Python still needs to be tested.
- **Largest current risk:** The model may misclassify sarcasm, mixed opinions, or ambiguous language. We also need to confirm the dataset's usage conditions.
- **Backup plan:** If the primary dataset cannot be accessed or used as intended, we will investigate the Cornell Movie Review Data.

### Next Two Actions

1. **Prisha Shah:** Load the IMDb dataset in Python, inspect the review text and sentiment labels, and begin preparing the data by November 1, 2026.
2. **Laura Botnaru:** Review the dataset documentation and usage conditions, and help evaluate the initial model results by November 8, 2026.

After these steps, both team members will work on model evaluation and documentation by November 15, prepare the poster and organize the repository by November 22, and finalize the project before the November 25, 2026 deadline.

## Possible Text-Analysis Method and Evaluation (Optional)

We plan to use TF-IDF to convert review text into numerical features and Logistic Regression to classify sentiment. We will compare the model with a simple baseline and evaluate its performance using accuracy, precision, recall, and F1-score. We will also examine incorrectly classified reviews to understand the model's limitations.

## Questions for the Instructor

Is our proposed project scope and approach appropriate for the assignment?

## AI-Use Statement

We used ChatGPT to help refine our project idea, organize the README, and plan our initial methods. We reviewed and adapted its suggestions. We will verify dataset information, test our code, and evaluate the model results ourselves.
