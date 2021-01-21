NOTE: the employee data is fake from [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# HR-Attrition 👔
Will they stay or will they go? Predicting whether employees will leave + why.

## Why this was done
I haven't done any HR analytics before and the idea of encorporating ML/DL with this domain excites me. The main problem is acquiring HR data. Thankfully IBM has produced a **synthentic dataset which I am utilising**. The other one is given the problem ("Who will leave in the next 6 months?") there aren't many people who have left historically in the last 6 months which leads to an **imbalanced class problem**. How can this potentially be solved? **generate more data**.

I feel that there are a lot of people within companies that add a ton of value to teams that may/may not go unnoticed and if they were to leave then progress within their teams could stall or worst case regress. If these individuals can be identified early and not just identified but understand WHY they might leave, that is the power of this.

Note: There is a lot of discussion about ML, DL and AI replacing jobs, this doesn't do so. I see the previously mentioned techniques allowing users to increase their productivity and in this particular use case, potentially saving employees leaving and improving the employees' work environment. If you had 3000 employees you could identify who might leave manually but not in a quick enough time nor efficiently. If you combine the knowledge of users and the model that is where this is powerful.

## Metric improvement incorporating synthetic data
| Metric      | w/o generated data | w/ generated data | 
| ----------- | ------------------ | ----------------- |
|   AUC       |        0.85        |       0.95        |
| Precision   |        0.73        |       0.92        |
| Recall      |        0.42        |       0.84        |

## Who this benefits
* 🕴️🕴️ **HR** - the obvious one. If employees respond to a survey and you have thousands of employees, without analysis you cannot efficiently find out who might be at risk of leaving. Secondly you will want to know employee's specific reasons as to why they might be at risk of leaving, it needs to be personal. Having a model that can tell you who might be at risk to leave the company and why they might leave saves HR time and offers a possible "save" of the employees at risk of leaving.
* 💁**Employees** - employees at risk of leaving the company will likely have a reason to leave e.g. working overtime often. Some may be on the fence about leaving and if the model can capture it early through using the inputs such as working overtime often this can be resolved. If the model can then be interpretted (using SHAP values) then HR can use the reasons to approach the employee and discuss these pain points that weren't necessarily obvious before. The employee will hopefully stay and have a better work environment and their talent retained.
* 🏭**Company** - there may be key members within a team that bring substantial value to their teams. If they were to leave then the company will potentially lose their value and may slow or lose progress from the employee's value and skills they bring. If the model captures them early, HR realise it and approach them with a personal discussion, they can potential "save" this employee from leaving and retain the value and skills they bring.

## Streamlit app
The final model was used as an experiment using `streamlit` to create a user based app. The app is split into two sections...
1. 1️⃣**Single prediction** - the user manually inputs values of an employee and finds out what the model predicts BUT it also includes a **reason plot** (highlighting what features contribute towards/away from the predicted value) using shap values.
![Single prediction](https://github.com/Lion-Mod/HR-Attrition/blob/main/single_prediction.gif)

### Example reason plot showing how features changed the prediction
![Example_reason](https://github.com/Lion-Mod/HR-Attrition/blob/main/example_reason_plot.PNG)

2. 1️⃣➕**Multi prediction** - upload a csv and get a dataframe back of the original data plus the **prediction** and **score**.
![Multi prediction](https://github.com/Lion-Mod/HR-Attrition/blob/main/multi_prediction.gif)

