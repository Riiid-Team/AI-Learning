[![Header](https://github.com/Riiid-Team/Riiid-Project/blob/main/images/Riiid!%20Project.png "Header")](https://www.kaggle.com/c/riiid-test-answer-prediction/overview/description)

# Riiid AIEd Challenge 2020

## About the Project


## Goals
The goal of this project is to create a machine learning model that can predict whether or not a user will answer a problem correctly.

## Background

### About Riiid Labs
> “Riiid Labs, an AI solutions provider delivering creative disruption to the education market, empowers global education players to rethink traditional ways of learning leveraging AI. With a strong belief in equal opportunity in education, Riiid launched an AI tutor based on deep-learning algorithms in 2017 that attracted more than one million South Korean students. This year, the company released EdNet, the world’s largest open database for AI education containing more than 100 million student interactions.” [source](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)

<details>
  <summary>More Info:  Click to Expand </summary>
 
### Riiid Labs EdTech Application: Santa TOEIC
> “Santa TOEIC is the AI-based web/mobile learning platform for TOEIC. AI tutor provides a one-on-one curriculum, effectively increasing scores based on the essential questions and lectures for each user.” [source](https://www.riiid.co/en/product)

### TOEIC: Test of English for International Communication
> “The Test of English for International Communication (TOEIC) is an international standardized test of English language proficiency for non-native speakers. It is intentionally designed to measure the everyday English skills of people working in an international environment.” [source](https://en.wikipedia.org/wiki/TOEIC)

### TOEIC Listening and Reading: Exam used to build Santa TOEIC
> “The TOEIC Listening & Reading test is an objective test… There are 200 questions to answer in two hours in Listening (approximately 45 minutes, 100 questions) and Reading (75 minutes, 100 questions).” [source](https://www.iibc-global.org/english/toeic/test/lr/about/format.html)

</details>


## Deliverables
- [MVP Notebook](https://github.com/Riiid-Team/Riiid-Project/blob/main/MVP.ipynb)
- Final Notebook
- [Slide Presentation](https://www.canva.com/design/DAEQJVWzMfI/xLQVCWT7rMXS3qD22awfGA/view?utm_content=DAEQJVWzMfI&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)
- Video presentation
- Submission directly to Kaggle Kernels
- Submission file must be named submission.csv

## Project Management 
- [Trello Board](https://trello.com/b/HK21qlYW) 
- [Capstone Standup](https://docs.google.com/document/d/1tSexQKQZE7XicJyN401ZG8SlkKxIRokE44qmFS5kDZI/edit?usp=sharing)
- [Knowledge Share](https://docs.google.com/document/d/1W8FVh89gN6bMn85uHgqLIz50elTtU2H9-R7jwpOUBRw/edit?usp=sharing)


## Data Dictionary

### Original Dataset
 
**`train.csv`**
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| row_id                      | (int64) ID code for the row                                                                         |
| timestamp                   | (int64) The time in milliseconds between this user ineteraction and the first event completion from the user |
| user_id                     | (int32) ID code for the user                                                                        |
| content_id                  | (int16) ID Code for the user interaction                                                            |
| content_type_id             | (int8) 0 if the event was a question being posed to the user, 1 is the event was watching a lecture|
| task_container_id           | (int16) ID code for the batch of questions or lectures. -1 for lectures                             |
| user_answer                 | (int8) The user's answer to the question, if any. -1 for lectures                                  |
| answered_correctly          | (int8) If the user responded correctly, if any. -1 for lectures                                    |
| prior_question_elapsed_time | (float32) The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures inbetween. Null for a user's first question bundle or lecture. Note: The time is the average time a user took to solve each question in the previous bundle.|
| prior_question_had_explanation | (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback. |

**`lectures.csv`**  metadata for the lectures watched by users as they progress in their education.
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| lecture_id                  | Foreign key for the train/test content_id column, when the content type is lecture (1).     |
| part                        | Top level category code for the lecture.                                                    |
| tag                         | One tag code for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.  |
| type_of                     | Brief description of the core purpose of the lecture.                                       |

**`questions.csv`**  metadata for the __questions__ posed to users.
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| question_id                 | Foreign key for the train/test content_id column, when the content type is question (0).    |
| bundle_id                   | Code for which questions are served together.                                               |
| correct_answer              | The answer to the question. Can be compared with the train user_answer column to check if the user was right.  |
| part                        | The relevant section of the TOEIC test.                                                    |
| tags                        | One or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions together. |

### Feature Engineering
| Feature Name                | Description                                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------------|
| user_lectures_rt            | Placeholder                                                                                 |
| last_q_time                 | Placeholder                                                                                 |
| user_acc_mean               | Placeholder                                                                                 |
| avg_user_q_time             | Placeholder                                                                                 |
| mean_content_accuracy       | Placeholder                                                                                 |
| question_content_asked      | Placeholder                                                                                 |
| std_content_accuracy        | Placeholder                                                                                 |
| median_content_accuracy     | Placeholder                                                                                 |
| skew_content_accuracy       | Placeholder                                                                                 |
| mean_task_accuracy          | Placeholder                                                                                 |
| question_task_asked         | Placeholder                                                                                 |
| std_task_accuracy           | Placeholder                                                                                 |
| median_tesk_accuracy        | Placeholder                                                                                 |
| skew_task_accuracy          | Placeholder                                                                                 |
| mean_timestamp_accuracy     | Placeholder                                                                                 |
| question_timestamp_asked    | Placeholder                                                                                 |
| std_timestamp_accuracy      | Placeholder                                                                                 |
| median_timestamp_accuracy   | Placeholder                                                                                 |
| skew_timestamp_accuracy     | Placeholder                                                                                 |
| mean_priortime_accuracy     | Placeholder                                                                                 |
| question_priortime_asked    | Placeholder                                                                                 |
| std_priortime_accuracy      | Placeholder                                                                                 |
| median_priortime_accuracy   | Placeholder                                                                                 |
| skew_priortime_accuracy     | Placeholder                                                                                 |
 

## Project Steps
### Acquire
- Data is acquired from Kaggle - [Riiid Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction/data).
- Create an acquire.py file.  
- File is a reproducible component for gathering the data.

### Prepare
- Create a prepare.py file. 
- Clean dataset.
- Missing values are investigated and handled.
- Run train, validate, and test.
- File is a reproducible component that is ready for exploration.

### Explore
- Explore data
- Summarize takeaways and conclusions.   

### Model
- 

### Conclusions
#### What was best model?
- 

#### How did the findings compare with what is known?
- 


### Future Investigations
#### What are your next steps?
- 

## How to Reproduce
All files are reproducible and available for download and use.
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, and Final_Report.ipynb files

## Contact Us 
Daniella Bojado
- daniella.bojado@gmail.com 

Samuel Davila
- samuelrdavila@gmail.com

Yongliang Shi
- yongliang.michael.shi@gmail.com

Christopher Logan Ortiz
- christopher.logan.ortiz@gmail.com

