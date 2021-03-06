[![Header](https://github.com/Riiid-Team/Riiid-Project/blob/main/images/Ai_Learning_Team_Banner.png "Header")](https://dardencapstone.com/)

<h2 align = "center"> <b> Welcome! We are the AI Learning Team &#128126; </b>
</h2>


![Header](https://github.com/Riiid-Team/Riiid-Project/blob/main/images/exe_summmary_slide.png)


## Background

### About Riiid Labs
“Riiid Labs, an AI solutions provider delivering creative disruption to the education market, empowers global education players to rethink traditional ways of learning leveraging AI. With a strong belief in equal opportunity in education, Riiid launched an AI tutor based on deep-learning algorithms in 2017 that attracted more than one million South Korean students. This year, the company released EdNet, the world’s largest open database for AI education containing more than 100 million student interactions.” [Source](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)

### TOEIC (Test of English for International Communication)
The TOEIC Listening & Reading test is an objective test that features 200 questions with a two hour time limit. Listening (approximately 45 minutes, 100 questions) and Reading (75 minutes, 100 questions). [Source](https://www.iibc-global.org/english/toeic/test/lr/about/format.html)

### Riiid's TOEIC Platform
“Santa TOEIC is the AI-based web/mobile learning platform for the TOEIC. AI tutor provides a one-on-one curriculum, effectively increasing scores based on the essential questions and lectures for each user.” [Source](https://www.riiid.co/en/product)

<p align = "center">
<img src="https://github.com/Riiid-Team/Riiid-Project/blob/main/gifs/santa_toeic_questions.gif">
</p>

## Deliverables
- [MVP Notebook](https://github.com/Riiid-Team/Riiid-Project/blob/main/MVP.ipynb)
- [Final Notebook](https://github.com/Riiid-Team/Riiid-Project/blob/main/Final-Report.ipynb)
- [Video presentation](https://youtu.be/d2yLCUWKsPY)
- [Slide Presentation](https://www.canva.com/design/DAESrRJRaro/vLbd1MMG-GG8HUCrbWu7gw/view?utm_content=DAESrRJRaro&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)

## Project Management 
- [Trello Board](https://trello.com/b/HK21qlYW) 
- [Capstone Standup](https://docs.google.com/document/d/1tSexQKQZE7XicJyN401ZG8SlkKxIRokE44qmFS5kDZI/edit?usp=sharing) 
- [Knowledge Share](https://docs.google.com/document/d/1W8FVh89gN6bMn85uHgqLIz50elTtU2H9-R7jwpOUBRw/edit?usp=sharing) 

## Data Dictionary

### Original Dataset
 
**`train.csv`**
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| row_id                      | (int64) ID code for the row                                                                 |
| timestamp                   | (int64) The time in milliseconds between this user interaction and the first event completion from the user |
| user_id                     | (int32) ID code for the user                                                                |
| content_id                  | (int16) ID Code for the user interaction                                                    |
| content_type_id             |  (int8) 0 if the event was a question prompted to the user, 1 if the event was watching a lecture|
| task_container_id           | (int16) ID code for the batch of questions or lectures. -1 for lectures                     |
| user_answer                 | (int8) The user's answer to the question, if any. -1 for lectures                           |
| answered_correctly          | (int8) If the user responded correctly, if any. -1 for lectures                             |
| prior_question_elapsed_time | (float32) The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in-between. Null for a user's first question bundle or lecture. Note: The time is the average time a user took to solve each question in the previous bundle.|
| prior_question_had_explanation | (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle ignoring any lectures in between. The value is shared across a single question bundle and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback. |

**`lectures.csv`**  metadata for the lectures watched by users as they progress in their education.
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| lecture_id                  | Foreign key for the train/test content_id column, when the content type is a lecture (1).     |
| part                        | Top-level category code for the lecture.                                                    |
| tag                         | One tag code for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.  |
| type_of                     | Brief description of the core purpose of the lecture.                                       |

**`questions.csv`**  metadata for the __questions__ posed to users.
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| question_id                 | Foreign key for the train/test content_id column, when the content-type is question (0).    |
| bundle_id                   | Code for which questions are served together.                                               |
| correct_answer              | The answer to the question. It can be compared with the train user_answer column to check if the user was right.  |
| part                        | The relevant section of the TOEIC test.                                                    |
| tags                        | One or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions together. |

### Feature Engineering
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| question_had_explanation    | Indicates if a question had an explanation                                                  |
| user_acc_mean               | The number of questions a user answered correctly divided by all questions they've answered |
| user_lectures_running_total | The running total of lectures a user has watched at a given timestamp                       |
| avg_user_q_time             | The average amount of time a user spends on a question                                      |
| mean_bundle_accuracy        | The average accuracy of a bundle of questions                                               |
| mean_content_accuracy       | The number of questions a user answered correctly divided by all questions they've answered in different content/topics|
| mean_tagcount_accuracy      | The number of tags linked to a question                                                     |
| mean_tags_accuracy          | The average accuracy for questions that share the same number of tags                       |
| mean_task_accuracy          | The average accuracy of a specific content_id                                               |
| median_content_accuracy     | The median accuracy for a specific content type                                             |
| median_tesk_accuracy        | The median accuracy for a task                                                              |
| q_time                      | The amount of time a user spent on the previous question                                    |
| question_content_asked      | The type of question asked                                                                  |
| question_priortime_asked    | The timestamp of the previous question prompted to the user                                 |
| question_task_asked         | The type of question asked in a bundle                                                      |
| question_timestamp_asked    | The timestamp a question was prompted to the user                                           |
| std_content_accuracy        | The standard deviation of content accuracy                                                  |
| std_task_accuracy           | The standard deviation of task accuracy                                                     |
| std_timestamp_accuracy      | The standard deviation of accuracy for a given timestamp                                    |
| skew_content_accuracy       | The skewness of accuracy for a specific content type                                        |
| skew_task_accuracy          | The skewness of accuracy for a specific task                                                |
| skew_timestamp_accuracy     | The skewness of accuracy for a given timestamp                                              |
| avg_user_q_time [scaled]    | Scaled version of avg_user_q_time using MinMaxScaler. Returned from prep_riiid function     | 
| user_lectures_running_total [scaled] | Scaled version of user_lectures_running_total using MinMaxScaler. Returned from prep_riiid function |


## Initial Thoughts
- Are questions with explanations answered correctly more often?
- Do users with higher accuracy take longer to answer questions?
- Are there questions that are more difficult to answer than others?
- How long does the average user engage with the platform?
- Does the number of lectures a user watch impact their accuracy?

## Project Steps
### 1. Acquire
Data acquired from [Kaggle](https://www.kaggle.com/c/riiid-test-answer-prediction/data). The data is stored in three separate files: lectures.csv, questions.csv, and train.csv. The primary dataset is train.csv, which has 100+ million user interactions from 390,000+ users. We used a random sample of 100K users for our analysis. The original 10 features describe the type of question, the time it took to answer, and whether the user’s response was correct.

### 2. Prepare
**Missing Values**
- Filled missing boolean values in `question_had_explanation` with False. Missing values indicated that the question did not have an explanation or the user viewed a lecture.
- Filled missing values in `prior_question_elapsed_time` with 0. Missing values indicated that a user viewed a lecture before answering the first question in a bundle of questions.
- Dropped columns: `lecture_id`, `tag`, `lecture_part`, `type_of`, `question_id`, `bundle_id`, `correct_answer`, `question_part`, and `tags`
- Dropped rows considered lectures: Where `answered_correctly` = -1

**Feature Engineering**
- Created new features using descriptive statistics for content, task, timestamp, and whether a question had an explanation. 
- Scaled timestamp from milliseconds to weeks to look at trends overtime.
- Refer to the feature engineering data dictionary for more information.

**Preprocessing**
- Scaled `mean_timestamp_accuracy`, `mean_priortime_accuracy`, `user_lectured_running_total`, and `avg_user_q_time` using MinMaxScaler

### 3. Explore
- Used scatterplots, barplots, and histograms to visualize interactions between features and the target variable.
- Performed hypothesis tests to find statistically significant relationships between features.

#### Hypotheses
**Hypothesis – Answered correctly vs. Question had an explanation**
> Null hypothesis: Whether a student gets a question right is independent of whether a question had an explanation.<br>
> Alternative hypothesis: Whether a student gets a question right is dependent on whether a question had an explanation.<br>
> Test: Chi-Squared Test<br>
> Results: With a p-value less than alpha, we reject the null hypothesis.

**Hypothesis – Answered correctly vs. Part**
> Null hypothesis: Whether a user answers a question correctly is independent of the type of question being asked.<br>
> Alternative hypothesis: Whether a user answers a question correctly is dependent upon the type of question being asked.<br>
> Test: Chi-Squared Test<br>
> Results: With a p-value less than alpha, we reject the null hypothesis.

**Hypothesis – Number of lectures a user has watched vs. Average task accuracy**
> Null hypothesis: There is no linear relationship between the number of lectures a user has watched and their average task accuracy.<br>
> Alternative hypothesis: There is a linear relationship between the number of lectures a user has watched and their average task accuracy.<br>
> Test: Pearson Correlation Test<br>
> Results: With a p-value less than alpha, we reject the null hypothesis.
> -	On average, as the number of lectures a user has seen increases, so does their task accuracy.
> -	Viewing lectures may have a weak positive impact on user accuracy.

**Hypothesis – Average user question time vs. Average user accuracy**
> Null hypothesis: There is no linear relationship between the average time a user takes to answer a question and their average accuracy.<br>
> Alternative hypothesis: There is a linear relationship between the average time a user takes to answer a question and their average accuracy.<br>
> Test: Pearson Correlation Test<br>
> Results: With a p-value less than alpha, we reject the null hypothesis
> -	Users who take longer to answer questions tend to have lower overall accuracy and vice versa

**Hypothesis – Average user accuracy vs. Average content accuracy**
> Null hypothesis: Users with above-average accuracy spend the same amount of time or more time on questions with lower-than-average content accuracy than users with average or lower-than-average accuracy.<br>
> Alternative hypothesis: Users with above-average accuracy spend less time on questions with lower-than-average content accuracy than users with average or lower-than-average accuracy.<br>
> Test: Two-Sample One-Tailed T-Test<br>
> Results: With a p-value less than alpha, and t less than 0, we reject the null hypothesis.
> -	If users with above-average accuracy answer questions (difficult and otherwise) more quickly than other users, then they may be more prepared for the content.

### 4. Model
First, we created a baseline model was to compare our model performances. The baseline is the most common outcome from the training dataset, answered correctly = 1. Baseline accuracy is 50%. This means that a user will get an answer correct 50% of the time.
Models evaluated on train, validate, and the test set were:
- Logistic Regression
- Random Forest
- LGBM


### Final Model
Our [LGBM model](https://lightgbm.readthedocs.io/en/latest/Features.html) performed the best, with an AUC score of .744. AUC is a measure of True Positives and False Positives. A True Positive means that our model predicted that a student answered a question correctly, and their response was correct. A False Positive means our model predicted a student responded to a question correctly when their answer was incorrect. An AUC score ranges between 0 and 1, where the higher the number, the more accurate a classification model is.

> "Gradient boosting algorithm sequentially combines weak learners (decision tree) in way that each new tree fits to the residuals from the previous step so that the model improves. The final model aggregates the results from each step and a strong learner is achieved." [Source](https://towardsdatascience.com/gradient-boosted-decision-trees-explained-9259bd8205af)

<p align="center">
<img src="./images/bgdt.JPG"
	title="Gradient Boosting Model" width="650" height="300">
</p>
	
### 5. Conclusion

#### Key Findings
- Performance decreases on incomplete sentences and narration.
- Performance increases with explanations and visuals.
- Lectures don't significantly impact performance.

#### What was the best model?
- LightGBM: AUC score of .744
- The LightGBM model surpassed the baseline by 0.24, which is a 47% improvement (which is a comparison of the difference between the scores divided by the baseline).

#### Recommendations
- Offer more study material for difficult subjects and to develop students' ability to understand english based on context clues when photos aren't available
- Revise lectures to have a stronger impact over the course of students' studies
- Use model to tailor content to student's needs

#### Expectation
- Riiid's program will better prepare students for the TOEIC

### Future Investigations
#### What are your next steps?
- Use this predictive model on Riiid's other educational programs.
- Explore more features and different classification models like xgboost.
- Improve model to predict new student performance more effectively.

## How to Reproduce
All files are reproducible and available for download and use.
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, and Final-Report.ipynb files
- [ ] Run Final-Report.ipynb

## Contact Us 
[Daniella Bojado](https://github.com/dbojado)

[Samuel Davila](https://github.com/SamuelD-Data)

[Yongliang Shi](https://github.com/Yongliang-Shi)

[Christopher Logan Ortiz](https://github.com/Promeos)

