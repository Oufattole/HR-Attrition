# HR-Attrition
Should I stay or should I go? Predicting whether employees will leave + why.

## Why this was done (more detail in the notebook)
I haven't done any HR analytics before and the idea of encorporating ML/DL excites me. The main problem is acquiring HR data. Thankfully IBM has produced a synthentic dataset which I am utilising.

There is a lot of discussion about ML, DL and AI replacing jobs, this doesn't do so. I see the previously mentioned techniques allowing for users to increase their productivity and in this particular use case, potentially prevent employees leaving and improve the employees work.

## Who this benefits
* 🕴️🕴️ **HR** - the obvious one. If employees respond to a survey and you have thousands of employees, without analysis you cannot efficiently find out who might be at risk of leaving. Secondly you will want to know employee's specific reasons as to why they might be at risk of leaving, it needs to be personal. Having a model that can tell you who might be at risk of leaving the company saves HR time and offers a possible "save" if there is interpretability of the model per employee. 
* 💁**Employees** - employees at risk of leaving the company will most likely have a reason, likely negative e.g. working overtime often. Some may be on the fence about leaving and if the model can capture it early, this can be resolved. If the model captures the reasons through model interpretability (using SHAP values) then HR can use them to approach the employee and discuss these pain points that weren't obvious before. The employee will hopefully stay and have a better work environment.
* 🏭**Company** - there may be key members within a team that bring substantial value to their teams. If they were to leave then the company will lose potential their value and may slow or lose progress from the employee's value and skills. If the model captures them early, HR realise it and approach them with a personal discussion, they can potential "save" this employee from leaving and retain the value and skills they bring.
